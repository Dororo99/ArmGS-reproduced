#!/usr/bin/env python3
"""Train ArmGS on one Waymo-v2 sequence using the StreetGS split.

The entry point intentionally keeps two initialization policies explicit.
``all-selected`` reproduces StreetGS' use of LiDAR from the complete selected
frame range, while ``train-only`` is the stricter held-out-sensor protocol.
COLMAP points are always expected to have been triangulated from training RGB
with known Waymo camera poses by :mod:`prepare_waymo_colmap`.  Legacy COLMAP
outputs use absolute Waymo world coordinates; the trainer aligns those points
to StreetGS full-context-centered coordinates exactly once.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for import_root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from armgs.backends import GsplatRasterizer
from armgs import materialize_actor_bbox_masks
from armgs.batching import canonical_frame_to_training_batch
from armgs.config import (
    build_core,
    build_density_controller,
    build_initialization_config,
    build_loss,
    build_sampler,
    build_sky,
    load_config,
)
from armgs.data import linspace_train_eval_split, periodic_train_eval_split
from armgs.data.schema import CanonicalDatasetManifest
from armgs.evaluation import (
    EvaluationAccumulator,
    LPIPSMetric,
    project_actor_boxes_to_mask,
)
from armgs.initialization import (
    load_colmap_points3d_text,
    preprocess_streetgs_waymo_background,
)
from armgs.pipeline import ArmGSCompositeRenderer
from armgs.scene_builder import (
    CanonicalScenePointClouds,
    ColoredPointCloud,
    build_scene_from_point_clouds,
    collect_colored_lidar_point_clouds,
)
from armgs.training import ArmGSTrainer
from train_armgs import (
    _atomic_write_text,
    _runtime_config,
    camera_scene_extent,
    conservative_scene_bounds,
    resolve_device,
    restore_training_checkpoint,
    save_training_checkpoint,
    validate_training_supervision,
)
from train_armgs_nuscenes import (
    _ACTOR_MASK_PROTOCOL,
    _LPIPS_PROTOCOL,
    _PSNR_PROTOCOL,
    _SSIM_PROTOCOL,
    _atomic_save_png,
    _execute_training_and_evaluation,
    _finish_wandb,
    _full_file_sha256,
    _gt_render_comparison,
    _handle_wandb_failure,
    _log_checkpoint_artifact,
    _single_hwc_rgb,
    _update_wandb_summary,
    _weighted_evaluation_summary,
    _without_rng_side_effects,
)


WAYMO_CAMERA_CHANNELS: tuple[str, ...] = (
    "FRONT",
    "FRONT_LEFT",
    "FRONT_RIGHT",
    "SIDE_LEFT",
    "SIDE_RIGHT",
)
PAPER_SPLIT_EVERY = 4
PAPER_SPLIT_OFFSET = 0
PAPER_SPLIT_START_POSITION = 4
STREETGS_PERIODIC_SPLIT = "streetgs-periodic"
SPLATAD_LINSPACE_SPLIT = "linspace"
PAPER_HEIGHT = 1066
PAPER_WIDTH = 1600
PAPER_ITERATIONS = 30_000
TRACKER_SOURCE = "waymo_gt"
PAPER_TRACKER_SOURCE = "streetgs_castrack"
STREETGS_ACTOR_MIN_POINTS = 2_000
STREETGS_ACTOR_MAX_POINTS: int | None = None
STREETGS_ACTOR_GRID_RESOLUTION = 20
STREETGS_ACTOR_RANDOM_SEED = 0
_DATASET_INPUT_IDENTITY_VERSION = 1
_SMALL_METADATA_CONTENT_HASH_MAX_BYTES = 8 * 1024 * 1024
_COLMAP_ABSOLUTE_WORLD_FRAME = "waymo_world"
_COLMAP_CENTERED_WORLD_FRAME = "waymo_world_centered"
_COLMAP_CENTERING_METHOD = "full_context_mean_vehicle_translation"


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _train_split_fraction(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must be finite and satisfy 0 < value < 1")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train ArmGS on one Waymo-v2 sequence with a StreetGS periodic or "
            "SplatAD LINSPACE split, LiDAR plus train-only COLMAP initialization, "
            "checkpoint-exact resume, held-out metrics, and W&B."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "armgs_waymo_streetgs.yaml",
        help="ArmGS Waymo YAML configuration",
    )
    parser.add_argument(
        "--waymo-root",
        "--root",
        dest="waymo_root",
        type=Path,
        required=True,
        help="Waymo-v2 dataset root",
    )
    parser.add_argument(
        "--parquet-dir",
        default="validation",
        help="Waymo split directory below the root (default: validation)",
    )
    parser.add_argument("--sequence", required=True, help="Waymo context name")
    parser.add_argument("--start-frame", type=_non_negative_int, default=0)
    parser.add_argument("--end-frame", type=_non_negative_int)
    parser.add_argument(
        "--split-type",
        choices=(STREETGS_PERIODIC_SPLIT, SPLATAD_LINSPACE_SPLIT),
        default=STREETGS_PERIODIC_SPLIT,
        help=(
            "streetgs-periodic holds out positions 4,8,12,...; linspace uses "
            "the SplatAD per-sensor selection (default: streetgs-periodic)"
        ),
    )
    parser.add_argument(
        "--train-split-fraction",
        type=_train_split_fraction,
        default=0.5,
        help="training fraction used by --split-type linspace (default: 0.5)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="decoded RGB/LiDAR cache (default: OUTPUT_DIR/waymo_cache)",
    )
    parser.add_argument(
        "--sky-mask-root",
        type=Path,
        help="Waymo sky-mask root consumed by the canonical loader",
    )
    parser.add_argument(
        "--colmap-points3d",
        type=Path,
        help=(
            "COLMAP points3D.txt produced from training RGB only, using known "
            "Waymo poses; mandatory in --paper-mode"
        ),
    )
    parser.add_argument(
        "--castrack-path",
        type=Path,
        help=(
            "StreetGS CAStrack JSON for this sequence (full validation JSON "
            "or an extracted per-scene JSON); mandatory in --paper-mode"
        ),
    )
    parser.add_argument(
        "--actor-box-scale",
        type=_positive_float,
        help=(
            "override initialization.actor_box_scale for this sequence; "
            "StreetGS applies this scale to actor length/width only"
        ),
    )
    parser.add_argument(
        "--camera",
        choices=("FRONT",),
        default="FRONT",
        help="paper evaluation camera (only FRONT is supported)",
    )
    parser.add_argument("--target-height", type=_positive_int, default=PAPER_HEIGHT)
    parser.add_argument("--target-width", type=_positive_int, default=PAPER_WIDTH)
    parser.add_argument(
        "--lidar-initialization-frames",
        choices=("all-selected", "train-only"),
        default="all-selected",
        help=(
            "all-selected matches StreetGS initialization; train-only avoids "
            "held-out LiDAR/RGB color seeding (default: all-selected)"
        ),
    )
    parser.add_argument(
        "--lidar-returns",
        choices=("first", "both"),
        default="first",
        help="range-image returns used for initialization (default: first)",
    )
    parser.add_argument(
        "--paper-mode",
        action="store_true",
        help=(
            "enforce 30k steps, 1600x1066 FRONT data, an explicit end frame, "
            "sky masks, and LiDAR+COLMAP initialization"
        ),
    )

    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="checkpoint to resume")
    parser.add_argument(
        "--iterations",
        type=_positive_int,
        help="total optimization steps, not additional resume steps",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=_positive_int,
        default=1000,
        help=(
            "legacy compatibility option; intermediate checkpoints are "
            "disabled and only final.pt is written"
        ),
    )
    parser.add_argument("--log-interval", type=_positive_int, default=100)
    parser.add_argument(
        "--image-log-interval",
        type=_non_negative_int,
        default=500,
        help="W&B GT/render logging interval; 0 disables images (default: 500)",
    )
    parser.add_argument(
        "--eval-interval",
        type=_non_negative_int,
        default=1000,
        help="held-out PSNR/SSIM evaluation interval; 0 disables periodic eval",
    )
    parser.add_argument(
        "--eval-at-end",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--eval-reconstruction-at-end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "score all training views with their exact appearance-embedding "
            "rows after training/eval-only (default: enabled)"
        ),
    )
    parser.add_argument(
        "--eval-lpips",
        action="store_true",
        help="also compute official LPIPS on the held-out split",
    )
    parser.add_argument(
        "--eval-lpips-net",
        choices=("alex", "vgg", "squeeze"),
        default="alex",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="restore and evaluate --resume without training/checkpoint writes",
    )

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY", "CamoSplat_ICLR_2027"),
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "Ours-ArmGS-Waymo"),
    )
    parser.add_argument(
        "--wandb-run-name",
        default=os.environ.get("WANDB_RUN_NAME") or os.environ.get("WANDB_NAME"),
    )
    parser.add_argument("--wandb-run-id", default=os.environ.get("WANDB_RUN_ID"))
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    default_wandb_dir = os.environ.get("WANDB_DIR")
    parser.add_argument(
        "--wandb-dir",
        type=Path,
        default=Path(default_wandb_dir) if default_wandb_dir else None,
    )
    parser.add_argument(
        "--wandb-fail-fast",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--wandb-log-checkpoint-artifact",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args(argv)
    if args.eval_only and args.resume is None:
        parser.error("--eval-only requires --resume")
    if args.end_frame is not None and args.end_frame < args.start_frame:
        parser.error("--end-frame must be greater than or equal to --start-frame")
    return args


def validate_paper_protocol(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> None:
    """Fail before loading data when a requested paper run is incomplete."""

    if not args.paper_mode:
        return
    errors: list[str] = []
    if args.colmap_points3d is None:
        errors.append("--colmap-points3d is required")
    if args.castrack_path is None:
        errors.append("--castrack-path is required")
    if args.sky_mask_root is None:
        errors.append("--sky-mask-root is required")
    if args.end_frame is None:
        errors.append("--end-frame is required")
    if args.camera != "FRONT":
        errors.append("--camera must be FRONT")
    if args.split_type != STREETGS_PERIODIC_SPLIT:
        errors.append("--split-type must be streetgs-periodic")
    if (args.target_height, args.target_width) != (PAPER_HEIGHT, PAPER_WIDTH):
        errors.append(f"resolution must be {PAPER_WIDTH}x{PAPER_HEIGHT}")
    if args.lidar_initialization_frames != "all-selected":
        errors.append("--lidar-initialization-frames must be all-selected")
    if args.lidar_returns != "first":
        errors.append("--lidar-returns must be first")
    if not args.eval_at_end:
        errors.append("--eval-at-end is required")
    if not args.eval_reconstruction_at_end:
        errors.append("--eval-reconstruction-at-end is required")
    if not args.eval_lpips:
        errors.append("--eval-lpips is required")
    if args.eval_lpips_net != "alex":
        errors.append("--eval-lpips-net must be alex")
    iterations = int(config["optimization"]["iterations"])
    if iterations != PAPER_ITERATIONS:
        errors.append(f"optimization iterations must be {PAPER_ITERATIONS}")
    initialization_config = config.get("initialization")
    if not isinstance(initialization_config, Mapping):
        errors.append("initialization config is required")
    elif initialization_config.get("voxel_size") is not None:
        errors.append("initialization.voxel_size must be null")
    try:
        actor_point_settings = (
            streetgs_waymo_actor_initialization_settings(config)
        )
    except (KeyError, TypeError, ValueError) as error:
        errors.append(str(error))
    else:
        if (
            actor_point_settings["minimum_lidar_points"]
            != STREETGS_ACTOR_MIN_POINTS
        ):
            errors.append(
                "initialization.streetgs_waymo.actor_min_lidar_points "
                f"must be {STREETGS_ACTOR_MIN_POINTS}"
            )
        if actor_point_settings["maximum_lidar_points"] is not None:
            errors.append(
                "initialization.streetgs_waymo.actor_max_lidar_points "
                "must be null (StreetGS does not cap dense actor LiDAR)"
            )
        if (
            actor_point_settings["fallback_grid_resolution"]
            != STREETGS_ACTOR_GRID_RESOLUTION
        ):
            errors.append(
                "initialization.streetgs_waymo."
                "actor_fallback_grid_resolution must be "
                f"{STREETGS_ACTOR_GRID_RESOLUTION}"
            )
    model_config = config.get("model")
    if not isinstance(model_config, Mapping) or int(
        model_config.get("sh_degree", -1)
    ) != 1:
        errors.append("model.sh_degree must be 1")
    data_config = config.get("data")
    waymo_config = (
        data_config.get("waymo") if isinstance(data_config, Mapping) else None
    )
    if not isinstance(waymo_config, Mapping) or float(
        waymo_config.get("scene_extent", float("nan"))
    ) != 20.0:
        errors.append("data.waymo.scene_extent must be 20")
    densification_config = config["optimization"].get("densification")
    if not isinstance(densification_config, Mapping):
        errors.append("optimization.densification config is required")
    elif densification_config.get("prune_actor_outside_box") is not True:
        errors.append(
            "optimization.densification.prune_actor_outside_box must be true"
        )
    if isinstance(densification_config, Mapping) and (
        "max_screen_radius" not in densification_config
        or densification_config.get("max_screen_radius") is not None
    ):
        errors.append(
            "optimization.densification.max_screen_radius must be null "
            "(StreetGS prunes background/actors by world scale, not screen size)"
        )
    if errors:
        raise ValueError("paper-mode protocol violation: " + "; ".join(errors))


def load_waymo_manifest(
    root: Path,
    *,
    sequence: str,
    parquet_dir: str,
    camera: str,
    start_frame: int,
    end_frame: int | None,
    target_size: tuple[int, int],
    lidar_returns: str,
    cache_dir: Path,
    sky_mask_root: Path | None,
    castrack_path: Path | None,
) -> CanonicalDatasetManifest:
    """Import the Waymo adapter lazily so CLI inspection stays lightweight."""

    try:
        from armgs.data.waymo import load_waymo_v2_manifest
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "Waymo canonical loader is unavailable; expected "
            "armgs.data.waymo.load_waymo_v2_manifest"
        ) from error
    return load_waymo_v2_manifest(
        root,
        sequence=sequence,
        parquet_dir=parquet_dir,
        camera_channels=(camera,),
        start_frame=start_frame,
        end_frame=end_frame,
        target_size=target_size,
        cache_dir=cache_dir,
        lidar_returns=lidar_returns,
        sky_mask_root=sky_mask_root,
        require_lidar=True,
        center_world=True,
        castrack_path=castrack_path,
    )


def load_waymo_context_center(
    root: Path,
    *,
    sequence: str,
    parquet_dir: str,
) -> torch.Tensor:
    """Load StreetGS full-context centering without selecting frame rows."""

    try:
        from armgs.data.waymo import load_waymo_world_center
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "Waymo world-center loader is unavailable; expected "
            "armgs.data.waymo.load_waymo_world_center"
        ) from error
    center = torch.as_tensor(
        load_waymo_world_center(
            root,
            sequence=sequence,
            parquet_dir=parquet_dir,
        )
    ).detach()
    if center.shape != (3,) or not center.is_floating_point():
        raise ValueError("Waymo world center must be a floating-point [3] tensor")
    if not torch.isfinite(center).all():
        raise ValueError("Waymo world center must be finite")
    return center.to(device="cpu", dtype=torch.float32)


def waymo_split_protocol(
    split_type: str,
    train_split_fraction: float = 0.5,
) -> dict[str, Any]:
    if split_type == STREETGS_PERIODIC_SPLIT:
        return {
            "type": "streetgs_periodic",
            "every": PAPER_SPLIT_EVERY,
            "offset": PAPER_SPLIT_OFFSET,
            "start_position": PAPER_SPLIT_START_POSITION,
            "held_out_relative_positions": "4,8,12,...",
        }
    if split_type == SPLATAD_LINSPACE_SPLIT:
        return {
            "type": "splatad_linspace",
            "train_fraction": float(train_split_fraction),
            "selection": "numpy.linspace(0,N-1,ceil(N*fraction),dtype=int64)",
            "per_sensor": True,
        }
    raise ValueError(f"unknown Waymo split type: {split_type!r}")


def split_waymo_manifest(
    manifest: CanonicalDatasetManifest,
    *,
    split_type: str = STREETGS_PERIODIC_SPLIT,
    train_split_fraction: float = 0.5,
) -> Any:
    """Apply the requested capture-atomic Waymo train/evaluation split."""

    if split_type == STREETGS_PERIODIC_SPLIT:
        return periodic_train_eval_split(
            manifest,
            every=PAPER_SPLIT_EVERY,
            offset=PAPER_SPLIT_OFFSET,
            start_position=PAPER_SPLIT_START_POSITION,
        )
    if split_type == SPLATAD_LINSPACE_SPLIT:
        return linspace_train_eval_split(
            manifest,
            train_fraction=train_split_fraction,
        )
    raise ValueError(f"unknown Waymo split type: {split_type!r}")


def lidar_initialization_manifest(
    source_manifest: CanonicalDatasetManifest,
    train_manifest: CanonicalDatasetManifest,
    policy: str,
) -> CanonicalDatasetManifest:
    if policy == "all-selected":
        return source_manifest
    if policy == "train-only":
        return train_manifest
    raise ValueError(f"unknown LiDAR initialization frame policy: {policy!r}")


def retain_training_actors(
    point_clouds: CanonicalScenePointClouds,
    train_manifest: CanonicalDatasetManifest,
) -> CanonicalScenePointClouds:
    """Discard clouds for actors with no optimizable training pose knots."""

    training_actor_ids = {track.actor_id for track in train_manifest.actor_tracks}
    return CanonicalScenePointClouds(
        background=point_clouds.background,
        actors={
            actor_id: cloud
            for actor_id, cloud in point_clouds.actors.items()
            if actor_id in training_actor_ids
        },
    )


def apply_streetgs_actor_point_policy(
    point_clouds: CanonicalScenePointClouds,
    train_manifest: CanonicalDatasetManifest,
    *,
    actor_box_scale: float,
    min_points: int = STREETGS_ACTOR_MIN_POINTS,
    max_points: int | None = STREETGS_ACTOR_MAX_POINTS,
    grid_resolution: int = STREETGS_ACTOR_GRID_RESOLUTION,
    seed: int = STREETGS_ACTOR_RANDOM_SEED,
) -> tuple[CanonicalScenePointClouds, dict[str, dict[str, Any]]]:
    """Apply StreetGS' sparse-cloud fallback without capping dense LiDAR.

    The official StreetGS Waymo actor initializer replaces a missing or
    sub-2,000-point cloud with a 20**3 bbox grid. Otherwise it retains the
    complete actor point cloud. max_points remains an explicit optional
    non-reference escape hatch for old experimental callers, but paper-mode
    validation requires it to be None.
    """

    if actor_box_scale <= 0.0:
        raise ValueError("actor_box_scale must be positive")
    if min_points <= 0 or (
        max_points is not None and max_points < min_points
    ):
        raise ValueError("actor point thresholds are invalid")
    if grid_resolution <= 1:
        raise ValueError("actor grid resolution must exceed one")
    if seed < 0:
        raise ValueError("actor random seed must be non-negative")

    reference = point_clouds.background.points
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    source_clouds = dict(point_clouds.actors)
    actor_clouds: dict[int, ColoredPointCloud] = {}
    records: dict[str, dict[str, Any]] = {}
    unit_axis = torch.linspace(
        -0.5,
        0.5,
        grid_resolution,
        dtype=reference.dtype,
        device=reference.device,
    )

    for track in sorted(train_manifest.actor_tracks, key=lambda item: item.actor_id):
        source = source_clouds.get(track.actor_id)
        source_count = 0 if source is None else int(source.points.shape[0])
        dimensions = track.dimensions_lwh.to(reference).clone()
        dimensions[:2] *= actor_box_scale
        if source is None or source_count < min_points:
            axes = tuple(unit_axis * dimensions[index] for index in range(3))
            grid = torch.stack(
                torch.meshgrid(*axes, indexing="ij"), dim=-1
            ).reshape(-1, 3)
            colors = torch.rand(
                (grid.shape[0], 3),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ).to(reference)
            cloud = ColoredPointCloud(grid, colors)
            strategy = (
                "bbox_grid_fallback_missing"
                if source is None
                else "bbox_grid_fallback_sparse"
            )
        elif max_points is not None and source_count > max_points:
            indices = torch.randperm(
                source_count,
                generator=generator,
                device="cpu",
            )[:max_points].to(device=source.points.device)
            cloud = ColoredPointCloud(
                source.points.index_select(0, indices),
                source.colors.index_select(0, indices),
            )
            strategy = "seeded_cap_non_reference"
        else:
            cloud = source
            strategy = "lidar_uncapped"

        used_lidar_point_count = (
            int(cloud.points.shape[0])
            if strategy in {"lidar_uncapped", "seeded_cap_non_reference"}
            else 0
        )
        generated_fallback_point_count = (
            int(cloud.points.shape[0])
            if strategy.startswith("bbox_grid_fallback_")
            else 0
        )

        actor_clouds[track.actor_id] = cloud
        records[str(track.actor_id)] = {
            "class_name": track.class_name,
            "source_point_count": source_count,
            "final_point_count": int(cloud.points.shape[0]),
            "used_lidar_point_count": used_lidar_point_count,
            "discarded_lidar_point_count": (
                source_count - used_lidar_point_count
            ),
            "generated_fallback_point_count": (
                generated_fallback_point_count
            ),
            "strategy": strategy,
            "dimensions_lwh": [
                float(value)
                for value in track.dimensions_lwh.detach().cpu().tolist()
            ],
            "effective_dimensions_lwh": [
                float(value) for value in dimensions.detach().cpu().tolist()
            ],
        }

    return (
        CanonicalScenePointClouds(
            background=point_clouds.background,
            actors=actor_clouds,
        ),
        records,
    )


def _waymo_component_paths(
    root: Path,
    parquet_dir: str,
    sequence: str,
) -> tuple[Path, ...]:
    split_directory = root / parquet_dir
    paths = tuple(sorted(split_directory.glob(f"*/{sequence}.parquet")))
    if not paths:
        raise FileNotFoundError(
            "no Waymo-v2 component parquet files were found for "
            f"{sequence!r} below {split_directory}"
        )
    return paths


def build_waymo_dataset_input_identity(
    manifest: Sequence[Any],
    *,
    root: Path,
    parquet_dir: str,
    sequence: str,
    colmap_points3d: Path | None = None,
    castrack_path: Path | None = None,
    small_metadata_content_hash_max_bytes: int = (
        _SMALL_METADATA_CONTENT_HASH_MAX_BYTES
    ),
) -> dict[str, Any]:
    """Fingerprint all component parquets and referenced decoded payloads."""

    if small_metadata_content_hash_max_bytes < 0:
        raise ValueError("small metadata content-hash limit must be non-negative")
    sources: list[tuple[str, Path, bool]] = [
        ("waymo_component", path, True)
        for path in _waymo_component_paths(root, parquet_dir, sequence)
    ]
    if colmap_points3d is not None:
        sources.append(("colmap_points3d", colmap_points3d, True))
    if castrack_path is not None:
        sources.append(("castrack", castrack_path, True))
    for frame in manifest:
        sources.append(("image", frame.image_path, False))
        if frame.lidar is not None:
            sources.append(("lidar", frame.lidar.source_path, False))
        if frame.sky_mask_path is not None:
            sources.append(("sky_mask", frame.sky_mask_path, False))
        if frame.actor_mask_path is not None:
            sources.append(("actor_mask", frame.actor_mask_path, False))

    aggregated: dict[str, dict[str, Any]] = {}
    for role, source_path, is_metadata in sources:
        resolved = source_path.resolve(strict=True)
        entry = aggregated.setdefault(
            str(resolved),
            {"path": resolved, "roles": set(), "is_metadata": False},
        )
        entry["roles"].add(role)
        entry["is_metadata"] = bool(entry["is_metadata"] or is_metadata)

    records: list[dict[str, Any]] = []
    for key in sorted(aggregated):
        source = aggregated[key]
        path = source["path"]
        before = path.stat()
        record: dict[str, Any] = {
            "resolved_path": key,
            "roles": sorted(source["roles"]),
            "size_bytes": before.st_size,
            "mtime_ns": before.st_mtime_ns,
            "verification": "stat_identity",
        }
        if source["is_metadata"] and before.st_size <= (
            small_metadata_content_hash_max_bytes
        ):
            record["content_sha256"] = _full_file_sha256(path)
            after = path.stat()
            if after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
                raise RuntimeError(f"dataset input changed while hashing: {path}")
            record["verification"] = "stat_identity+full_content_sha256"
        records.append(record)

    canonical = {
        "version": _DATASET_INPUT_IDENTITY_VERSION,
        "root": str(root.resolve()),
        "parquet_dir": parquet_dir,
        "sequence": sequence,
        "files": records,
    }
    serialized = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "version": _DATASET_INPUT_IDENTITY_VERSION,
        "digest_sha256": hashlib.sha256(serialized).hexdigest(),
        "file_count": len(records),
        "component_file_count": sum(
            "waymo_component" in record["roles"] for record in records
        ),
        "stat_identity_fields": ["resolved_path", "size_bytes", "mtime_ns"],
        "frame_payload_verification": "stat_identity",
        "small_metadata_verification": "stat_identity+full_content_sha256",
        "small_metadata_content_hash_max_bytes": (
            small_metadata_content_hash_max_bytes
        ),
    }


def _colmap_coordinate_frame_contract(
    payload: Mapping[str, Any],
    known_pose: Mapping[str, Any],
) -> dict[str, Any]:
    """Parse the optional centered-frame declaration in ``mapping.json``.

    Mapping files created before full-context centering had no declaration and
    are intentionally interpreted as absolute Waymo world coordinates.  A new
    mapping may opt into the centered frame only through this explicit schema::

        coordinate_frame:
          name: waymo_world_centered
          centering_method: full_context_mean_vehicle_translation
          world_center_m: [x, y, z]
    """

    declaration = payload.get("coordinate_frame")
    if declaration is None:
        if known_pose.get("world_frame") != _COLMAP_ABSOLUTE_WORLD_FRAME:
            raise ValueError(
                "COLMAP mapping without coordinate_frame must use waymo_world"
            )
        return {
            "name": _COLMAP_ABSOLUTE_WORLD_FRAME,
            "centered": False,
            "declaration": "implicit_legacy_absolute_world",
            "centering_method": None,
            "world_center_m": None,
        }
    if not isinstance(declaration, Mapping):
        raise ValueError("COLMAP coordinate_frame must be a mapping")
    if declaration.get("name") != _COLMAP_CENTERED_WORLD_FRAME:
        raise ValueError(
            "COLMAP coordinate_frame.name must be waymo_world_centered"
        )
    if declaration.get("centering_method") != _COLMAP_CENTERING_METHOD:
        raise ValueError(
            "COLMAP coordinate_frame has an incompatible centering method"
        )
    if known_pose.get("world_frame") != _COLMAP_CENTERED_WORLD_FRAME:
        raise ValueError(
            "centered COLMAP mapping known-pose world_frame must be "
            "waymo_world_centered"
        )
    declared_center = torch.as_tensor(declaration.get("world_center_m"))
    if declared_center.shape != (3,) or not declared_center.is_floating_point():
        raise ValueError(
            "COLMAP coordinate_frame.world_center_m must be floating-point [3]"
        )
    if not torch.isfinite(declared_center).all():
        raise ValueError("COLMAP coordinate-frame world center must be finite")
    return {
        "name": _COLMAP_CENTERED_WORLD_FRAME,
        "centered": True,
        "declaration": "mapping_json_explicit",
        "centering_method": _COLMAP_CENTERING_METHOD,
        "world_center_m": [
            float(value) for value in declared_center.to(torch.float64).tolist()
        ],
    }


def load_colmap_provenance(
    points3d: Path | None,
    *,
    sequence: str,
    train_source_indices: Sequence[int],
    eval_source_indices: Sequence[int],
    split_protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate prepare_waymo_colmap's train-only mapping when available."""

    if points3d is None:
        return {
            "verified": False,
            "mapping": None,
            "reason": "sfm_disabled",
            "coordinate_frame": None,
        }
    candidates = (
        points3d.parent / "mapping.json",
        points3d.parent.parent / "mapping.json",
    )
    mapping_path = next((path for path in candidates if path.is_file()), None)
    if mapping_path is None:
        return {
            "verified": False,
            "mapping": None,
            "reason": "prepare_waymo_colmap_mapping_not_found",
            "coordinate_frame": {
                "name": _COLMAP_ABSOLUTE_WORLD_FRAME,
                "centered": False,
                "declaration": "implicit_legacy_absolute_world",
                "centering_method": None,
                "world_center_m": None,
            },
        }
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if payload.get("dataset") != "waymo_v2":
        raise ValueError("COLMAP mapping dataset is not waymo_v2")
    if payload.get("sequence") != sequence:
        raise ValueError("COLMAP mapping sequence does not match training sequence")
    if payload.get("status") != "complete":
        raise ValueError("COLMAP mapping does not describe a complete reconstruction")
    final_points = payload.get("final_points3D_path")
    if not isinstance(final_points, str) or (
        Path(final_points).resolve() != points3d.resolve()
    ):
        raise ValueError("COLMAP mapping final points3D path does not match this run")
    if payload.get("camera_channels") != ["FRONT"]:
        raise ValueError("COLMAP mapping must contain FRONT training images only")
    known_pose = payload.get("known_pose_contract")
    if not isinstance(known_pose, Mapping):
        raise ValueError("COLMAP mapping has no known-pose contract")
    expected_known_pose = {
        "camera_model": "PINHOLE",
        "camera_convention": "opencv",
        "pose_refinement": False,
        "intrinsics_refinement": False,
    }
    for key, expected in expected_known_pose.items():
        if known_pose.get(key) != expected:
            raise ValueError(
                f"COLMAP mapping has incompatible known-pose field {key}"
            )
    coordinate_frame = _colmap_coordinate_frame_contract(payload, known_pose)
    split = payload.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("COLMAP mapping has no split metadata")
    expected_split = (
        dict(split_protocol)
        if split_protocol is not None
        else waymo_split_protocol(STREETGS_PERIODIC_SPLIT)
    )
    if expected_split["type"] == "streetgs_periodic":
        if split.get("type") not in {None, "periodic", "streetgs_periodic"}:
            raise ValueError("COLMAP mapping has incompatible split type")
        expected_protocol = {
            "every": PAPER_SPLIT_EVERY,
            "offset": PAPER_SPLIT_OFFSET,
            "start_position": PAPER_SPLIT_START_POSITION,
        }
        for key, expected in expected_protocol.items():
            if int(split.get(key, -1)) != expected:
                raise ValueError(f"COLMAP mapping has incompatible split {key}")
    elif expected_split["type"] == "splatad_linspace":
        if split.get("type") not in {"linspace", "splatad_linspace"}:
            raise ValueError("COLMAP mapping has incompatible split type")
        if float(split.get("train_fraction", float("nan"))) != float(
            expected_split["train_fraction"]
        ):
            raise ValueError("COLMAP mapping has incompatible train_fraction")
    else:
        raise ValueError("runtime Waymo split protocol is unsupported")
    if list(split.get("train_source_indices", ())) != list(train_source_indices):
        raise ValueError("COLMAP mapping training rows do not match this run")
    if list(split.get("eval_source_indices", ())) != list(eval_source_indices):
        raise ValueError("COLMAP mapping evaluation rows do not match this run")
    frame_rows = [int(frame["source_index"]) for frame in payload.get("frames", ())]
    if frame_rows != list(train_source_indices):
        raise ValueError("COLMAP mapping staged frames are not exactly training rows")
    return {
        "verified": True,
        "mapping": str(mapping_path.resolve()),
        "reason": "train_rows_match_runtime_split",
        "coordinate_frame": coordinate_frame,
    }


def align_colmap_points_to_centered_world(
    points: torch.Tensor,
    *,
    world_center: torch.Tensor,
    provenance: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Align SfM points to the centered manifest frame exactly once."""

    if points.ndim != 2 or points.shape[-1] != 3 or not points.is_floating_point():
        raise ValueError("COLMAP points must be a floating-point [N,3] tensor")
    if not torch.isfinite(points).all():
        raise ValueError("COLMAP points must be finite")
    center = torch.as_tensor(world_center).detach()
    if center.shape != (3,) or not center.is_floating_point():
        raise ValueError("Waymo world center must be a floating-point [3] tensor")
    if not torch.isfinite(center).all():
        raise ValueError("Waymo world center must be finite")
    coordinate_frame = provenance.get("coordinate_frame")
    if coordinate_frame is None:
        coordinate_frame = {
            "name": _COLMAP_ABSOLUTE_WORLD_FRAME,
            "centered": False,
            "declaration": "implicit_legacy_absolute_world",
            "world_center_m": None,
        }
    if not isinstance(coordinate_frame, Mapping):
        raise ValueError("COLMAP provenance coordinate_frame must be a mapping")
    centered = coordinate_frame.get("centered")
    if centered is True:
        declared_center = torch.as_tensor(
            coordinate_frame.get("world_center_m"), dtype=torch.float64
        )
        runtime_center = center.to(device="cpu", dtype=torch.float64)
        if declared_center.shape != (3,) or not torch.isfinite(
            declared_center
        ).all():
            raise ValueError("centered COLMAP provenance has no valid world center")
        if not torch.allclose(
            declared_center, runtime_center, rtol=0.0, atol=1.0e-4
        ):
            raise ValueError(
                "COLMAP centered-frame world center does not match this context"
            )
        aligned = points
        operation = "none_already_centered"
    elif centered is False:
        if coordinate_frame.get("name") != _COLMAP_ABSOLUTE_WORLD_FRAME:
            raise ValueError("unrecognized absolute COLMAP coordinate frame")
        aligned = points - center.to(points)[None]
        operation = "subtract_full_context_world_center_once"
    else:
        raise ValueError("COLMAP coordinate-frame centered flag must be boolean")
    alignment = {
        "input_coordinate_frame": coordinate_frame.get("name"),
        "input_frame_declaration": coordinate_frame.get("declaration"),
        "target_coordinate_frame": _COLMAP_CENTERED_WORLD_FRAME,
        "centering_method": _COLMAP_CENTERING_METHOD,
        "operation": operation,
        "world_center_m": [
            float(value) for value in center.to(torch.float64).tolist()
        ],
    }
    return aligned, alignment


def enforce_colmap_provenance(
    provenance: Mapping[str, Any],
    *,
    paper_mode: bool,
) -> None:
    """Require auditable train-only known-pose SfM for paper-mode runs."""

    if paper_mode and provenance.get("verified") is not True:
        raise ValueError(
            "paper-mode requires a verified train-only mapping.json from "
            "prepare_waymo_colmap"
        )


def enforce_paper_actor_mask_supervision(
    manifest: CanonicalDatasetManifest,
    *,
    paper_mode: bool,
) -> None:
    """Require attached masks, including explicit empty masks, in paper mode."""

    if not paper_mode:
        return
    missing = [
        row
        for row, frame in enumerate(manifest)
        if frame.actor_mask_path is None
    ]
    if missing:
        raise ValueError(
            "paper-mode requires an attached actor bbox mask on every "
            f"training row; missing rows {missing}"
        )


def streetgs_waymo_initialization_settings(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return validated settings for the effective StreetGS Waymo pass."""

    initialization = config.get("initialization")
    settings = (
        initialization.get("streetgs_waymo")
        if isinstance(initialization, Mapping)
        else None
    )
    if not isinstance(settings, Mapping):
        raise ValueError("initialization.streetgs_waymo config is required")
    result = {
        "background_voxel_size": float(settings["background_voxel_size"]),
        "radius_outlier_nb_points": int(settings["radius_outlier_nb_points"]),
        "radius_outlier_radius": float(settings["radius_outlier_radius"]),
        "sfm_extent_multiplier": float(settings["sfm_extent_multiplier"]),
        "filter_colmap_near_or_below_cameras": settings[
            "filter_colmap_near_or_below_cameras"
        ],
    }
    for name in (
        "background_voxel_size",
        "radius_outlier_radius",
        "sfm_extent_multiplier",
    ):
        value = result[name]
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"initialization.streetgs_waymo.{name} must be positive")
    if result["radius_outlier_nb_points"] <= 0:
        raise ValueError(
            "initialization.streetgs_waymo.radius_outlier_nb_points must be positive"
        )
    if not isinstance(result["filter_colmap_near_or_below_cameras"], bool):
        raise ValueError(
            "initialization.streetgs_waymo."
            "filter_colmap_near_or_below_cameras must be boolean"
        )
    return result


def streetgs_waymo_actor_initialization_settings(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return validated StreetGS Waymo actor point-cloud policy settings."""

    initialization = config.get("initialization")
    settings = (
        initialization.get("streetgs_waymo")
        if isinstance(initialization, Mapping)
        else None
    )
    if not isinstance(settings, Mapping):
        raise ValueError("initialization.streetgs_waymo config is required")

    required = (
        "actor_min_lidar_points",
        "actor_max_lidar_points",
        "actor_fallback_grid_resolution",
        "actor_fallback_random_seed",
    )
    missing = [name for name in required if name not in settings]
    if missing:
        raise ValueError(
            "initialization.streetgs_waymo is missing actor point setting(s): "
            + ", ".join(missing)
        )

    def integer(name: str, *, minimum: int) -> int:
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"initialization.streetgs_waymo.{name} must be an integer"
            )
        if value < minimum:
            raise ValueError(
                f"initialization.streetgs_waymo.{name} must be at least "
                f"{minimum}"
            )
        return value

    minimum_lidar_points = integer("actor_min_lidar_points", minimum=1)
    maximum_value = settings["actor_max_lidar_points"]
    if maximum_value is None:
        maximum_lidar_points = None
    else:
        maximum_lidar_points = integer(
            "actor_max_lidar_points", minimum=minimum_lidar_points
        )
    return {
        "minimum_lidar_points": minimum_lidar_points,
        "maximum_lidar_points": maximum_lidar_points,
        "fallback_grid_resolution": integer(
            "actor_fallback_grid_resolution", minimum=2
        ),
        "random_seed": integer("actor_fallback_random_seed", minimum=0),
    }


def gaussian_initial_scale_diagnostics(scene: Any) -> dict[str, Any]:
    """Summarize initialized physical max-axis scales before CUDA transfer."""

    def values(gaussians: Any, label: str) -> torch.Tensor:
        log_scales = gaussians.log_scales.detach()
        if not torch.isfinite(log_scales).all():
            raise ValueError(f"{label} initial log-scales contain non-finite values")
        physical = log_scales.exp().amax(dim=-1).to(
            device="cpu", dtype=torch.float64
        )
        if not torch.isfinite(physical).all():
            raise ValueError(f"{label} initial scales contain non-finite values")
        return physical

    def summary(scale_values: torch.Tensor) -> dict[str, Any]:
        if scale_values.numel() == 0:
            return {
                "gaussian_count": 0,
                "q50_m": None,
                "q90_m": None,
                "q99_m": None,
                "q99_9_m": None,
                "max_m": None,
            }
        quantiles = torch.quantile(
            scale_values,
            torch.tensor([0.5, 0.9, 0.99, 0.999], dtype=torch.float64),
        )
        return {
            "gaussian_count": int(scale_values.numel()),
            "q50_m": float(quantiles[0].item()),
            "q90_m": float(quantiles[1].item()),
            "q99_m": float(quantiles[2].item()),
            "q99_9_m": float(quantiles[3].item()),
            "max_m": float(scale_values.max().item()),
        }

    background = values(scene.background, "background")
    actor_values: list[torch.Tensor] = []
    per_actor: dict[str, Any] = {}
    for actor in scene.actors:
        current = values(actor.gaussians, f"actor {actor.actor_id}")
        actor_values.append(current)
        per_actor[str(actor.actor_id)] = summary(current)
    composite = torch.cat((background, *actor_values)) if actor_values else background
    actor_total = (
        torch.cat(actor_values)
        if actor_values
        else torch.empty(0, dtype=torch.float64)
    )
    return {
        "scale_definition": "maximum activated axis per Gaussian",
        "background": summary(background),
        "actors": {
            "actor_model_count": len(per_actor),
            **summary(actor_total),
            "per_actor": per_actor,
        },
        "composite": summary(composite),
    }


def _camera_channel_name(camera_id: int) -> str:
    if 0 <= camera_id < len(WAYMO_CAMERA_CHANNELS):
        return WAYMO_CAMERA_CHANNELS[camera_id]
    return f"CAMERA_{camera_id}"


_TRAIN_SCALAR_KEYS = frozenset(
    {
        "loss",
        "rgb_l1",
        "ssim_loss",
        "depth_loss",
        "sky_loss",
        "foreground_loss",
        "gaussians",
    }
)


def _log_to_wandb(
    run: Any,
    record: Mapping[str, Any],
    *,
    fail_fast: bool = False,
    commit: bool = True,
) -> bool:
    """Log Waymo train telemetry with controllable final-step commit."""

    step = int(record["step"])
    payload = {
        (f"train/{key}" if key in _TRAIN_SCALAR_KEYS else key): value
        for key, value in record.items()
        if key != "step"
    }
    if not payload:
        return True
    try:
        run.log(payload, step=step, commit=commit)
    except Exception as error:
        return _handle_wandb_failure(
            "metric/image logging", error, fail_fast=fail_fast
        )
    return True


def _wandb_image_payload_factory(
    batch: Any,
    output: Any,
    *,
    fail_fast: bool = False,
    training_manifest: Sequence[Any] | None = None,
    training_source_indices: Sequence[int] | None = None,
) -> Mapping[str, Any]:
    """Create the Waymo-named GT/render preview payload."""

    try:
        import wandb

        target = _single_hwc_rgb(batch.target_rgb, "target RGB")
        rendered = _single_hwc_rgb(output.rendering.rgb, "rendered RGB")
        if target.shape != rendered.shape:
            raise ValueError("GT and rendered RGB shapes must match")
        comparison = (
            torch.cat((target, rendered), dim=1)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .numpy()
        )
        camera_id = int(torch.as_tensor(batch.view.camera_id).detach().cpu().item())
        timestamp_ns = int(
            torch.as_tensor(batch.view.timestamp).detach().cpu().item()
        )
        training_row = (
            None
            if batch.view.training_row is None
            else int(
                torch.as_tensor(batch.view.training_row).detach().cpu().item()
            )
        )
        frame_index: int | None = None
        source_index: int | None = None
        if training_manifest is not None:
            if training_row is None or not 0 <= training_row < len(training_manifest):
                raise ValueError("train image has an invalid training row")
            frame_index = int(training_manifest[training_row].frame_index)
            if training_source_indices is not None:
                if len(training_source_indices) != len(training_manifest):
                    raise ValueError("training source-index mapping length mismatch")
                source_index = int(training_source_indices[training_row])
        channel = _camera_channel_name(camera_id)
        fields = [
            f"Step: {int(getattr(output, 'step', 0))}",
            f"Camera: {channel} ({camera_id})",
        ]
        if training_row is not None:
            fields.append(f"Training row: {training_row}")
        fields.append(f"Timestamp ns: {timestamp_ns}")
        if frame_index is not None:
            fields.append(f"Frame: {frame_index}")
        if source_index is not None:
            fields.append(f"Source row: {source_index}")
        fields.append("Left: GT | Right: Render")
        payload: dict[str, Any] = {
            "train/gt_vs_render": wandb.Image(
                comparison, caption=" | ".join(fields)
            ),
            "train/image_camera_id": camera_id,
            "train/image_camera": channel,
            "train/image_timestamp_ns": str(timestamp_ns),
        }
        if training_row is not None:
            payload["train/image_training_row"] = training_row
        if frame_index is not None:
            payload["train/image_frame_index"] = frame_index
        if source_index is not None:
            payload["train/image_source_index"] = source_index
        return payload
    except Exception as error:
        _handle_wandb_failure(
            "train image creation", error, fail_fast=fail_fast
        )
        return {}


def _initialize_wandb(
    args: argparse.Namespace,
    *,
    config: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
) -> Any | None:
    """Initialize W&B with Waymo-specific, checkpoint-relevant metadata."""

    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError:
        _handle_wandb_failure(
            "initialization",
            ImportError("W&B logging requested but 'wandb' is not installed"),
            fail_fast=bool(args.wandb_fail_fast),
        )
        return None
    fail_fast = bool(args.wandb_fail_fast)
    wandb_directory = args.wandb_dir or (args.output_dir / "wandb")
    try:
        wandb_directory.mkdir(parents=True, exist_ok=True)
    except Exception as error:
        _handle_wandb_failure(
            "local directory creation", error, fail_fast=fail_fast
        )
        return None

    identity = run_metadata.get("dataset_input_identity")
    identity_sha256 = (
        identity.get("digest_sha256") if isinstance(identity, Mapping) else None
    )
    wandb_config = {
        "armgs": copy.deepcopy(dict(config)),
        "dataset": {
            "type": "waymo_v2",
            "root": run_metadata["waymo_root"],
            "parquet_dir": run_metadata["parquet_dir"],
            "sequence": run_metadata["sequence"],
            "source_frame_range": run_metadata["source_frame_range"],
            "camera_channels": run_metadata["camera_channels"],
            "target_resolution": run_metadata["target_resolution"],
            "cache_dir": run_metadata["cache_dir"],
            "sky_mask_root": run_metadata.get("sky_mask_root"),
            "sky_mask_count": run_metadata.get("sky_mask_count", 0),
            "actor_mask_root": run_metadata.get("actor_mask_root"),
            "actor_mask_count": run_metadata.get("actor_mask_count", 0),
            "actor_mask_protocol": copy.deepcopy(
                run_metadata.get("actor_mask_protocol")
            ),
            "tracker_source": run_metadata["tracker_source"],
            "input_identity_sha256": identity_sha256,
        },
        "split": {
            **dict(run_metadata["split_protocol"]),
            "train_rows": len(run_metadata["train_source_indices"]),
            "eval_rows": len(run_metadata["eval_source_indices"]),
        },
        "initialization": copy.deepcopy(dict(run_metadata["initialization"])),
        "evaluation": {
            "interval": int(args.eval_interval),
            "at_end": bool(args.eval_at_end),
            "reconstruction_at_end": bool(
                args.eval_reconstruction_at_end
            ),
            "lpips": bool(args.eval_lpips),
            "lpips_net": str(args.eval_lpips_net),
            "eval_only": bool(args.eval_only),
            "metric_protocols": {
                "psnr": _PSNR_PROTOCOL,
                "ssim": _SSIM_PROTOCOL,
                "lpips": _LPIPS_PROTOCOL if args.eval_lpips else None,
                "actor_mask": _ACTOR_MASK_PROTOCOL,
            },
        },
        "logging": {
            "scalar_interval": int(args.log_interval),
            "image_interval": int(args.image_log_interval),
            "fail_fast": fail_fast,
            "checkpoint_artifact": bool(args.wandb_log_checkpoint_artifact),
        },
    }
    init_options: dict[str, Any] = {}
    requested_run_id = str(args.wandb_run_id).strip() if args.wandb_run_id else None
    resume_source = "explicit" if requested_run_id else "new"
    sidecar_path = args.output_dir / "wandb_run.json"
    if requested_run_id is None and sidecar_path.is_file():
        try:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar_id = payload.get("run_id") if isinstance(payload, Mapping) else None
            if not isinstance(sidecar_id, str) or not sidecar_id.strip():
                raise ValueError("sidecar run_id must be a non-empty string")
            requested_run_id = sidecar_id.strip()
            resume_source = "sidecar"
        except Exception as error:
            _handle_wandb_failure("run-ID sidecar read", error, fail_fast=fail_fast)
    if requested_run_id:
        init_options.update(id=requested_run_id, resume="allow")
    try:
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            dir=str(wandb_directory),
            config=wandb_config,
            **init_options,
        )
    except Exception as error:
        _handle_wandb_failure("initialization", error, fail_fast=fail_fast)
        return None
    if run is None:
        _handle_wandb_failure(
            "initialization",
            RuntimeError("W&B initialization returned no run"),
            fail_fast=fail_fast,
        )
        return None
    run_id = getattr(run, "id", None)
    if run_id:
        sidecar = {
            "format_version": 1,
            "run_id": str(run_id),
            "entity": str(getattr(run, "entity", None) or args.wandb_entity),
            "project": str(getattr(run, "project", None) or args.wandb_project),
            "name": getattr(run, "name", None) or args.wandb_run_name,
            "url": getattr(run, "url", None),
            "mode": args.wandb_mode,
            "resume_requested": bool(requested_run_id),
            "resume_source": resume_source,
        }
        try:
            _atomic_write_text(
                sidecar_path,
                json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
            )
        except Exception as error:
            _handle_wandb_failure(
                "run-ID sidecar write", error, fail_fast=fail_fast
            )
    return run


def _log_evaluation_to_wandb(
    run: Any,
    record: Mapping[str, Any],
    previews: Mapping[str, Any],
    *,
    fail_fast: bool = False,
    commit: bool = True,
) -> bool:
    """Log split-qualified Waymo metrics without nuScenes camera aliases."""

    try:
        import wandb

        prefix = str(record["split"])
        payload: dict[str, Any] = {}
        for metric, value in record["aggregate"].items():
            if value is not None:
                payload[f"{prefix}/{metric}"] = value
        for camera_name, summary in record["per_camera"].items():
            for metric, value in summary.items():
                if value is not None:
                    payload[f"{prefix}/{camera_name}/{metric}"] = value
        for camera_name, comparison in previews.items():
            payload[f"{prefix}/{camera_name}/gt_vs_render"] = wandb.Image(
                comparison,
                caption=f"{prefix} {camera_name}: Left GT | Right Render",
            )
        checkpoint_step = int(record["step"])
        log_step = checkpoint_step
        current_step = getattr(run, "step", None)
        if not isinstance(current_step, bool):
            try:
                current_step = int(current_step)
            except (TypeError, ValueError):
                current_step = None
        else:
            current_step = None
        if current_step is not None and current_step > checkpoint_step:
            # A completed run resumes at the next W&B history step.  Reusing
            # the checkpoint step would make W&B silently discard eval-only
            # history.  Preserve the model step explicitly while appending at
            # the first legal history step.
            payload["evaluation/checkpoint_step"] = checkpoint_step
            payload[f"{prefix}/checkpoint_step"] = checkpoint_step
            log_step = current_step
        run.log(payload, step=log_step, commit=commit)
    except Exception as error:
        return _handle_wandb_failure(
            "evaluation logging", error, fail_fast=fail_fast
        )
    return True


def _wandb_training_record_commit(
    *,
    step: int,
    total_iterations: int,
    evaluation_interval: int,
    final_evaluation_due: bool,
) -> bool:
    """Keep a same-step W&B row open until its evaluation is attached."""

    periodic_evaluation_due = bool(
        evaluation_interval > 0 and step % evaluation_interval == 0
    )
    final_step_evaluation_due = bool(
        step == total_iterations and final_evaluation_due
    )
    return not (periodic_evaluation_due or final_step_evaluation_due)


def evaluate_waymo_split(
    renderer: ArmGSCompositeRenderer,
    manifest: CanonicalDatasetManifest,
    *,
    device: torch.device | str,
    output_directory: Path,
    step: int,
    actor_box_scale: float = 1.0,
    lpips_metric: LPIPSMetric | None = None,
    wandb_run: Any | None = None,
    wandb_fail_fast: bool = False,
    wandb_commit: bool = True,
    evaluation_policy: Mapping[str, Any] | None = None,
    split_name: str = "novel_view",
    training_rows: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Score FRONT frames as novel views or exact-row reconstructions."""

    if not manifest:
        raise ValueError("evaluation manifest cannot be empty")
    if step < 0:
        raise ValueError("evaluation step must be non-negative")
    if split_name not in {"novel_view", "reconstruction"}:
        raise ValueError("split_name must be novel_view or reconstruction")
    if split_name == "reconstruction":
        if training_rows is None:
            raise ValueError("reconstruction evaluation requires training rows")
        resolved_training_rows: tuple[int | None, ...] = tuple(training_rows)
        if resolved_training_rows != tuple(range(len(manifest))):
            raise ValueError(
                "reconstruction training rows must exactly index the manifest"
            )
    else:
        if training_rows is not None:
            raise ValueError("novel-view evaluation cannot use training rows")
        resolved_training_rows = (None,) * len(manifest)
    accumulators: dict[str, EvaluationAccumulator] = {}
    previews: dict[str, Any] = {}
    preview_paths: dict[str, str] = {}
    previous_training = renderer.training
    renderer.eval()
    try:
        with torch.inference_mode():
            for frame, training_row in zip(manifest, resolved_training_rows):
                batch = canonical_frame_to_training_batch(
                    frame, training_row=training_row, device=device
                )
                prediction = renderer(batch.view).rgb.clamp(0.0, 1.0)
                camera_name = _camera_channel_name(int(frame.camera_id))
                accumulator = accumulators.setdefault(
                    camera_name,
                    EvaluationAccumulator(lpips_metric=lpips_metric),
                )
                projected = project_actor_boxes_to_mask(
                    frame,
                    manifest.actor_tracks,
                    box_scale=actor_box_scale,
                )
                accumulator.update(
                    prediction,
                    batch.target_rgb,
                    actor_mask=projected if projected.any() else None,
                )
                if camera_name not in previews:
                    comparison = _gt_render_comparison(batch.target_rgb, prediction)
                    preview_path = (
                        output_directory
                        / "evaluation"
                        / split_name
                        / f"step_{step:08d}_{camera_name}_gt_render.png"
                    )
                    _atomic_save_png(preview_path, comparison)
                    previews[camera_name] = comparison
                    preview_paths[camera_name] = str(preview_path)
    finally:
        renderer.train(previous_training)
    per_camera = {
        camera_name: accumulators[camera_name].summary()
        for camera_name in sorted(accumulators)
    }
    record: dict[str, Any] = {
        "event": f"{split_name}_evaluation",
        "step": int(step),
        "split": split_name,
        "aggregate": _weighted_evaluation_summary(per_camera),
        "per_camera": per_camera,
        "previews": {
            camera_name: preview_paths[camera_name]
            for camera_name in sorted(preview_paths)
        },
        "policy": dict(evaluation_policy or {}),
    }
    _atomic_write_text(
        output_directory
        / "evaluation"
        / split_name
        / f"step_{step:08d}.json",
        json.dumps(record, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(record, sort_keys=True), flush=True)
    if wandb_run is not None:
        _log_evaluation_to_wandb(
            wandb_run,
            record,
            previews,
            fail_fast=wandb_fail_fast,
            commit=wandb_commit,
        )
    return record


def run(args: argparse.Namespace) -> Path:
    config = _runtime_config(load_config(args.config), args.iterations)
    if args.split_type == SPLATAD_LINSPACE_SPLIT:
        config.setdefault("data", {})["split"] = waymo_split_protocol(
            args.split_type,
            args.train_split_fraction,
        )
    validate_paper_protocol(args, config)
    total_iterations = int(config["optimization"]["iterations"])
    device = resolve_device(args.device)
    root = args.waymo_root.resolve(strict=True)
    output_directory = args.output_dir.resolve()
    cache_dir = (
        args.cache_dir.resolve()
        if args.cache_dir is not None
        else output_directory / "waymo_cache"
    )
    sky_mask_root = (
        args.sky_mask_root.resolve(strict=True)
        if args.sky_mask_root is not None
        else None
    )
    colmap_points3d = (
        args.colmap_points3d.resolve(strict=True)
        if args.colmap_points3d is not None
        else None
    )
    castrack_path = (
        args.castrack_path.resolve(strict=True)
        if args.castrack_path is not None
        else None
    )
    world_center = load_waymo_context_center(
        root,
        sequence=args.sequence,
        parquet_dir=args.parquet_dir,
    )
    tracker_source = (
        PAPER_TRACKER_SOURCE if castrack_path is not None else TRACKER_SOURCE
    )
    configured_actor_box_scale = float(
        config["initialization"].get("actor_box_scale", 1.0)
    )
    actor_box_scale = (
        float(args.actor_box_scale)
        if args.actor_box_scale is not None
        else configured_actor_box_scale
    )
    if not math.isfinite(actor_box_scale) or actor_box_scale <= 0.0:
        raise ValueError("actor box scale must be finite and positive")
    if castrack_path is None:
        print(
            "warning: tracker_source=waymo_gt; ArmGS reports StreetGS CAStrack "
            "boxes, so this run cannot claim exact tracker parity",
            file=sys.stderr,
        )

    manifest = load_waymo_manifest(
        root,
        sequence=args.sequence,
        parquet_dir=args.parquet_dir,
        camera=args.camera,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        target_size=(args.target_height, args.target_width),
        lidar_returns=args.lidar_returns,
        cache_dir=cache_dir,
        sky_mask_root=sky_mask_root,
        castrack_path=castrack_path,
    )
    actor_mask_root = output_directory / "prepared_actor_masks"
    manifest = materialize_actor_bbox_masks(
        manifest,
        actor_mask_root,
        box_scale=actor_box_scale,
    )
    split_protocol = waymo_split_protocol(
        args.split_type,
        args.train_split_fraction,
    )
    split = split_waymo_manifest(
        manifest,
        split_type=args.split_type,
        train_split_fraction=args.train_split_fraction,
    )
    enforce_paper_actor_mask_supervision(
        split.train_manifest,
        paper_mode=args.paper_mode,
    )
    colmap_provenance = load_colmap_provenance(
        colmap_points3d,
        sequence=args.sequence,
        train_source_indices=split.train_source_indices,
        eval_source_indices=split.eval_source_indices,
        split_protocol=split_protocol,
    )
    enforce_colmap_provenance(colmap_provenance, paper_mode=args.paper_mode)
    if colmap_points3d is not None and not colmap_provenance["verified"]:
        print(
            "warning: COLMAP points are accepted as train-only by contract, but "
            "prepare_waymo_colmap mapping.json provenance was not found",
            file=sys.stderr,
        )
    loss = build_loss(config)
    validate_training_supervision(split.train_manifest, loss)

    initialization = build_initialization_config(config)
    background_initialization = replace(initialization, voxel_size=None)
    actor_initialization = replace(initialization, voxel_size=None)
    waymo_data_config = config.get("data", {}).get("waymo")
    if not isinstance(waymo_data_config, Mapping):
        raise ValueError("data.waymo config is required")
    scene_extent = float(waymo_data_config["scene_extent"])
    if not math.isfinite(scene_extent) or scene_extent <= 0.0:
        raise ValueError("data.waymo.scene_extent must be finite and positive")
    measured_camera_extent = camera_scene_extent(split.train_manifest)
    preprocessing_settings = streetgs_waymo_initialization_settings(config)
    actor_point_settings = streetgs_waymo_actor_initialization_settings(config)
    lidar_manifest = lidar_initialization_manifest(
        manifest,
        split.train_manifest,
        args.lidar_initialization_frames,
    )
    point_clouds = collect_colored_lidar_point_clouds(
        lidar_manifest,
        actor_box_scale=actor_box_scale,
    )
    point_clouds = retain_training_actors(point_clouds, split.train_manifest)
    point_clouds, actor_initialization_records = (
        apply_streetgs_actor_point_policy(
            point_clouds,
            split.train_manifest,
            actor_box_scale=actor_box_scale,
            min_points=actor_point_settings["minimum_lidar_points"],
            max_points=actor_point_settings["maximum_lidar_points"],
            grid_resolution=actor_point_settings[
                "fallback_grid_resolution"
            ],
            seed=actor_point_settings["random_seed"],
        )
    )
    actor_strategy_counts: dict[str, int] = {}
    for record in actor_initialization_records.values():
        strategy = str(record["strategy"])
        actor_strategy_counts[strategy] = (
            actor_strategy_counts.get(strategy, 0) + 1
        )
    actor_point_totals = {
        name: sum(
            int(record[name])
            for record in actor_initialization_records.values()
        )
        for name in (
            "source_point_count",
            "final_point_count",
            "used_lidar_point_count",
            "discarded_lidar_point_count",
            "generated_fallback_point_count",
        )
    }
    raw_lidar_background_point_count = int(
        point_clouds.background.points.shape[0]
    )
    sfm_points: torch.Tensor | None = None
    sfm_colors: torch.Tensor | None = None
    sfm_alignment = {
        "input_coordinate_frame": None,
        "input_frame_declaration": None,
        "target_coordinate_frame": _COLMAP_CENTERED_WORLD_FRAME,
        "centering_method": _COLMAP_CENTERING_METHOD,
        "operation": "sfm_disabled",
        "world_center_m": [
            float(value) for value in world_center.to(torch.float64).tolist()
        ],
    }
    if colmap_points3d is not None:
        sfm_points, sfm_colors = load_colmap_points3d_text(colmap_points3d)
        sfm_points, sfm_alignment = align_colmap_points_to_centered_world(
            sfm_points,
            world_center=world_center,
            provenance=colmap_provenance,
        )
    camera_centers = torch.stack(
        [frame.camera_to_world[:3, 3] for frame in split.train_manifest], dim=0
    )
    preprocessed_background = preprocess_streetgs_waymo_background(
        point_clouds.background.points,
        point_clouds.background.colors,
        sfm_points,
        sfm_colors,
        camera_centers=camera_centers,
        voxel_size=preprocessing_settings["background_voxel_size"],
        radius_outlier_nb_points=preprocessing_settings[
            "radius_outlier_nb_points"
        ],
        radius_outlier_radius=preprocessing_settings["radius_outlier_radius"],
        sfm_extent_multiplier=preprocessing_settings["sfm_extent_multiplier"],
        filter_sfm_near_or_below_cameras=preprocessing_settings[
            "filter_colmap_near_or_below_cameras"
        ],
        camera_extent=scene_extent,
    )
    point_clouds = CanonicalScenePointClouds(
        background=ColoredPointCloud(
            preprocessed_background.points,
            preprocessed_background.colors,
        ),
        actors=point_clouds.actors,
    )

    bounds = conservative_scene_bounds(
        split.train_manifest,
        point_clouds,
        minimum_padding=max(3.0 * initialization.initial_scale, 1.0e-3),
    )
    scene = build_scene_from_point_clouds(
        split.train_manifest,
        point_clouds,
        initialization=background_initialization,
        actor_initialization=actor_initialization,
        sky=build_sky(config),
        require_all_actor_points=True,
    )
    initial_scale_diagnostics = gaussian_initial_scale_diagnostics(scene)
    print(
        json.dumps(
            {
                "event": "initial_gaussian_scale_diagnostics",
                **initial_scale_diagnostics,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    core = build_core(
        config,
        num_training_frames=len(split.train_manifest),
        training_camera_ids=split.training_camera_ids,
        training_timestamps=split.training_timestamps,
        scene_aabb_min=bounds.aabb_min,
        scene_aabb_max=bounds.aabb_max,
    )
    renderer = ArmGSCompositeRenderer(core, scene, GsplatRasterizer()).to(device)
    sampler = build_sampler(config, dataset_size=len(split.train_manifest))
    density_controller = build_density_controller(
        config,
        renderer.scene,
        scene_scale=scene_extent,
        actor_box_scale=actor_box_scale,
    )
    trainer = ArmGSTrainer.from_config(
        renderer,
        loss,
        config,
        sampler=sampler,
        density_controller=density_controller,
        background_extent=scene_extent,
        actor_box_scale=actor_box_scale,
    )

    initialization_metadata = {
        "modalities": ["lidar", "sfm"] if colmap_points3d else ["lidar"],
        "lidar_frames": args.lidar_initialization_frames,
        "lidar_returns": args.lidar_returns,
        "streetgs_reference_lidar_returns": "first",
        "lidar_capture_count": len({frame.frame_index for frame in lidar_manifest}),
        "lidar_image_count": len(lidar_manifest),
        "lidar_uses_held_out_sensor_data": (
            args.lidar_initialization_frames == "all-selected"
        ),
        "lidar_uses_held_out_rgb_for_color": (
            args.lidar_initialization_frames == "all-selected"
        ),
        "sfm_frames": "train-only",
        "sfm_provenance": colmap_provenance,
        "sfm_alignment": sfm_alignment,
        "colmap_points3d": str(colmap_points3d) if colmap_points3d else None,
        "world_coordinate_frame": _COLMAP_CENTERED_WORLD_FRAME,
        "full_context_world_center_m": [
            float(value) for value in world_center.to(torch.float64).tolist()
        ],
        "lidar_background_input_point_count": raw_lidar_background_point_count,
        "lidar_background_retained_point_count": (
            preprocessed_background.lidar_point_count
        ),
        "sfm_input_point_count": preprocessed_background.sfm_input_point_count,
        "sfm_retained_point_count": (
            preprocessed_background.sfm_retained_point_count
        ),
        "merged_background_point_count": int(
            point_clouds.background.points.shape[0]
        ),
        "streetgs_waymo_preprocessing": {
            **preprocessing_settings,
            "operation_order": [
                "lidar_voxel_down_sample",
                "lidar_radius_outlier_removal",
                "lidar_aabb_sphere_filter_sfm",
                "concatenate_without_revoxel",
            ],
            "lidar_aabb_center_m": [
                float(value)
                for value in preprocessed_background.lidar_aabb_center.to(
                    torch.float64
                ).tolist()
            ],
            "lidar_aabb_half_diagonal_m": float(
                preprocessed_background.lidar_aabb_half_diagonal.item()
            ),
            "camera_extent_m": scene_extent,
        },
        "configured_gaussian_voxel_size_m": initialization.voxel_size,
        "effective_background_gaussian_voxel_size_m": (
            background_initialization.voxel_size
        ),
        "sfm_merge_voxel_size_m": None,
        "actor_voxel_size_m": actor_initialization.voxel_size,
        "initial_scale_diagnostics": initial_scale_diagnostics,
        "actor_point_policy": {
            "name": (
                "streetgs_bbox_grid_fallback_and_uncapped_lidar"
                if actor_point_settings["maximum_lidar_points"] is None
                else "bbox_grid_fallback_and_non_reference_seeded_cap"
            ),
            "minimum_lidar_points": actor_point_settings[
                "minimum_lidar_points"
            ],
            "maximum_lidar_points": actor_point_settings[
                "maximum_lidar_points"
            ],
            "lidar_point_cap_enabled": (
                actor_point_settings["maximum_lidar_points"] is not None
            ),
            "fallback_grid_resolution": actor_point_settings[
                "fallback_grid_resolution"
            ],
            "fallback_point_count": (
                actor_point_settings["fallback_grid_resolution"] ** 3
            ),
            "random_seed": actor_point_settings["random_seed"],
            "configured_actor_box_scale": configured_actor_box_scale,
            "actor_box_scale": actor_box_scale,
            "actor_box_scale_cli_override": args.actor_box_scale is not None,
            "strategy_counts": actor_strategy_counts,
            "point_totals": actor_point_totals,
            "actor_count": len(actor_initialization_records),
            "actors": actor_initialization_records,
        },
    }
    paper_protocol_deviations: list[str] = []
    if args.split_type != STREETGS_PERIODIC_SPLIT:
        paper_protocol_deviations.append("split_not_streetgs_periodic")
    if args.lidar_initialization_frames != "all-selected":
        paper_protocol_deviations.append("lidar_initialization_not_all_selected")
    if castrack_path is None:
        paper_protocol_deviations.append(
            "tracker_source_waymo_gt_not_streetgs_castrack"
        )
    if args.lidar_returns != "first":
        paper_protocol_deviations.append("lidar_returns_not_streetgs_first")
    if initialization.voxel_size is not None:
        paper_protocol_deviations.append(
            "gaussian_initialization_revoxel_was_configured"
        )
    if (
        actor_point_settings["minimum_lidar_points"]
        != STREETGS_ACTOR_MIN_POINTS
    ):
        paper_protocol_deviations.append(
            "actor_min_lidar_points_not_streetgs_2000"
        )
    if actor_point_settings["maximum_lidar_points"] is not None:
        paper_protocol_deviations.append(
            "dense_actor_lidar_points_capped_not_streetgs"
        )
    if (
        actor_point_settings["fallback_grid_resolution"]
        != STREETGS_ACTOR_GRID_RESOLUTION
    ):
        paper_protocol_deviations.append(
            "actor_fallback_grid_resolution_not_streetgs_20"
        )
    run_metadata = {
        "dataset": "waymo_v2",
        "waymo_root": str(root),
        "parquet_dir": args.parquet_dir,
        "sequence": args.sequence,
        "source_frame_range": {
            "start": args.start_frame,
            "end_inclusive": args.end_frame,
        },
        "cache_dir": str(cache_dir),
        "castrack_path": str(castrack_path) if castrack_path else None,
        "sky_mask_root": str(sky_mask_root) if sky_mask_root else None,
        "sky_mask_count": sum(frame.sky_mask_path is not None for frame in manifest),
        "actor_mask_root": str(actor_mask_root.resolve()),
        "actor_mask_count": sum(
            frame.actor_mask_path is not None for frame in manifest
        ),
        "actor_mask_train_count": sum(
            frame.actor_mask_path is not None for frame in split.train_manifest
        ),
        "actor_mask_protocol": {
            "source": f"projected_{tracker_source}_cuboid_union",
            "format": "grayscale_png_actor_255_background_0",
            "box_scale": actor_box_scale,
            "box_scale_axes": "length_width_only",
            "fully_empty_masks_remain_attached": True,
        },
        "camera_channels": [args.camera],
        "camera_ids": sorted({frame.camera_id for frame in manifest}),
        "target_resolution": [args.target_height, args.target_width],
        "split_protocol": split_protocol,
        "train_source_indices": list(split.train_source_indices),
        "eval_source_indices": list(split.eval_source_indices),
        "tracker_source": tracker_source,
        "paper_tracker_source": PAPER_TRACKER_SOURCE,
        "paper_mode_requested": bool(args.paper_mode),
        "paper_protocol_compliant": bool(
            args.paper_mode and not paper_protocol_deviations
        ),
        "paper_protocol_deviations": paper_protocol_deviations,
        "require_all_actor_models": True,
        "actors_without_lidar_use_bbox_fallback": True,
        "initialization": initialization_metadata,
        "background_scene_extent": scene_extent,
        "background_scene_extent_source": "data.waymo.scene_extent",
        "measured_camera_scene_extent": measured_camera_extent,
        "actor_scene_extents": {
            str(actor.actor_id): actor.density_extent(
                actor_box_scale=actor_box_scale
            )
            for actor in renderer.scene.actors
            if actor.dimensions_lwh is not None
        },
        "hash_grid_aabb": {
            "min": list(bounds.aabb_min),
            "max": list(bounds.aabb_max),
        },
        "dataset_input_identity": build_waymo_dataset_input_identity(
            manifest,
            root=root,
            parquet_dir=args.parquet_dir,
            sequence=args.sequence,
            colmap_points3d=colmap_points3d,
            castrack_path=castrack_path,
        ),
    }
    if args.resume is not None:
        restore_training_checkpoint(
            trainer,
            args.resume,
            map_location=device,
            config=config,
            run_metadata=run_metadata,
        )

    checkpoints = output_directory / "checkpoints"
    _atomic_write_text(
        output_directory / "resolved_config.yaml",
        yaml.safe_dump(config, sort_keys=False),
    )
    _atomic_write_text(
        output_directory / "run_metadata.json",
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n",
    )

    wandb_run = _initialize_wandb(
        args,
        config=config,
        run_metadata=run_metadata,
    )
    evaluation_policy = {
        "views": ["reconstruction", "novel_view"],
        "interval": int(args.eval_interval),
        "at_end": bool(args.eval_at_end),
        "reconstruction_at_end": bool(args.eval_reconstruction_at_end),
        "lpips": bool(args.eval_lpips),
        "lpips_net": args.eval_lpips_net,
        "eval_only": bool(args.eval_only),
        "resolution": [args.target_height, args.target_width],
        "metric_protocols": {
            "psnr": _PSNR_PROTOCOL,
            "ssim": _SSIM_PROTOCOL,
            "lpips": _LPIPS_PROTOCOL if args.eval_lpips else None,
            "actor_mask": _ACTOR_MASK_PROTOCOL,
        },
    }
    _atomic_write_text(
        output_directory / "evaluation_policy.json",
        json.dumps(evaluation_policy, indent=2, sort_keys=True) + "\n",
    )
    latest_evaluations: dict[str, dict[str, Any]] = {}
    final_novel_evaluation_due = bool(
        args.eval_only
        or args.eval_at_end
        or (args.eval_interval and total_iterations % args.eval_interval == 0)
    )
    final_wandb_evaluation_due = bool(
        final_novel_evaluation_due or args.eval_reconstruction_at_end
    )
    final_path = (
        args.resume
        if args.eval_only and args.resume is not None
        else checkpoints / "final.pt"
    )
    active_error = False
    try:
        lpips_metric = (
            _without_rng_side_effects(
                lambda: LPIPSMetric(net=args.eval_lpips_net, device=device),
                device=device,
            )
            if args.eval_lpips
            else None
        )

        def evaluation_callback(step: int) -> None:
            latest_evaluations["novel_view"] = _without_rng_side_effects(
                lambda: evaluate_waymo_split(
                    renderer,
                    split.eval_manifest,
                    device=device,
                    output_directory=output_directory,
                    step=step,
                    actor_box_scale=actor_box_scale,
                    lpips_metric=lpips_metric,
                    wandb_run=wandb_run,
                    wandb_fail_fast=args.wandb_fail_fast,
                    evaluation_policy=evaluation_policy,
                    wandb_commit=not (
                        args.eval_reconstruction_at_end
                        and (args.eval_only or step == total_iterations)
                    ),
                    split_name="novel_view",
                ),
                device=device,
            )

        def after_training_callback() -> None:
            save_training_checkpoint(final_path, trainer, config, run_metadata)

        _execute_training_and_evaluation(
            trainer,
            split.train_manifest,
            total_iterations=total_iterations,
            device=device,
            checkpoint_interval=args.checkpoint_interval,
            log_interval=args.log_interval,
            checkpoint_callback=None,
            log_callback=(
                (
                    lambda record: _log_to_wandb(
                        wandb_run,
                        record,
                        fail_fast=args.wandb_fail_fast,
                        commit=_wandb_training_record_commit(
                            step=int(record["step"]),
                            total_iterations=total_iterations,
                            evaluation_interval=int(args.eval_interval),
                            final_evaluation_due=final_wandb_evaluation_due,
                        ),
                    )
                )
                if wandb_run is not None
                else None
            ),
            log_payload_factory=(
                (
                    lambda batch, output: _wandb_image_payload_factory(
                        batch,
                        output,
                        fail_fast=args.wandb_fail_fast,
                        training_manifest=split.train_manifest,
                        training_source_indices=split.train_source_indices,
                    )
                )
                if (
                    wandb_run is not None
                    and args.wandb_mode != "disabled"
                    and args.image_log_interval > 0
                )
                else None
            ),
            image_log_interval=(
                args.image_log_interval
                if (
                    wandb_run is not None
                    and args.wandb_mode != "disabled"
                    and args.image_log_interval > 0
                )
                else 0
            ),
            evaluation_interval=args.eval_interval,
            evaluate_at_end=args.eval_at_end,
            eval_only=args.eval_only,
            evaluation_callback=evaluation_callback,
            after_training_callback=after_training_callback,
        )
        if args.eval_reconstruction_at_end:
            latest_evaluations["reconstruction"] = _without_rng_side_effects(
                lambda: evaluate_waymo_split(
                    renderer,
                    split.train_manifest,
                    device=device,
                    output_directory=output_directory,
                    step=trainer.step,
                    actor_box_scale=actor_box_scale,
                    lpips_metric=lpips_metric,
                    wandb_run=wandb_run,
                    wandb_fail_fast=args.wandb_fail_fast,
                    evaluation_policy=evaluation_policy,
                    split_name="reconstruction",
                    training_rows=range(len(split.train_manifest)),
                ),
                device=device,
            )
        if wandb_run is not None:
            summary: dict[str, Any] = {
                "checkpoint": str(final_path),
                "completed_steps": trainer.step,
            }
            for split_name, evaluation in latest_evaluations.items():
                for metric, value in evaluation["aggregate"].items():
                    if value is not None:
                        summary[f"{split_name}/{metric}"] = value
            _update_wandb_summary(
                wandb_run,
                summary,
                fail_fast=args.wandb_fail_fast,
            )
            if args.wandb_log_checkpoint_artifact and not args.eval_only:
                artifact = _log_checkpoint_artifact(
                    wandb_run,
                    final_path,
                    metadata={
                        "dataset": "waymo_v2",
                        "scene": args.sequence,
                        "step": int(trainer.step),
                        "dataset_input_identity_sha256": run_metadata[
                            "dataset_input_identity"
                        ]["digest_sha256"],
                    },
                    fail_fast=args.wandb_fail_fast,
                )
                if artifact is not None:
                    _update_wandb_summary(
                        wandb_run,
                        {
                            "checkpoint_artifact/sha256": artifact["sha256"],
                            "checkpoint_artifact/size_bytes": artifact[
                                "size_bytes"
                            ],
                        },
                        fail_fast=args.wandb_fail_fast,
                    )
        return final_path
    except BaseException:
        active_error = True
        raise
    finally:
        if wandb_run is not None:
            _finish_wandb(
                wandb_run,
                exit_code=1 if active_error else 0,
                fail_fast=args.wandb_fail_fast and not active_error,
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        final_path = run(args)
        print(json.dumps({"checkpoint": str(final_path)}, sort_keys=True), flush=True)
        return 0
    except (
        FileNotFoundError,
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
