#!/usr/bin/env python3
"""Train ArmGS on one canonicalized nuScenes scene.

This entry point mirrors the checkpoint-exact KITTI training path while keeping
nuScenes conversion and W&B integration explicit. Scene initialization consumes
training captures only, so held-out RGB and LiDAR cannot seed Gaussians.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
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
from armgs.data import periodic_train_eval_split
from armgs.data.schema import CanonicalDatasetManifest
from armgs.data.nuscenes import (
    NUSCENES_CAMERA_CHANNELS,
    load_nuscenes_manifest,
    normalize_nuscenes_scene_name,
    parse_nuscenes_sky_mask_reject_list,
)
from armgs.evaluation import (
    EvaluationAccumulator,
    LPIPSMetric,
    project_actor_boxes_to_mask,
)
from armgs.initialization import load_colmap_points3d_text
from armgs.pipeline import ArmGSCompositeRenderer
from armgs.scene_builder import (
    build_scene_from_point_clouds,
    collect_colored_lidar_point_clouds,
    merge_sfm_background,
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
    train_until,
    validate_training_supervision,
)


_NUSCENES_DATASET_INPUT_IDENTITY_VERSION = 3
_SMALL_METADATA_CONTENT_HASH_MAX_BYTES = 8 * 1024 * 1024
_PSNR_PROTOCOL = "mean-per-image-rgb-mse-data-range-1"
_SSIM_PROTOCOL = "3dgs-gaussian-11x11-sigma-1.5-data-range-1"
_LPIPS_PROTOCOL = "official-lpips-v0.1-minus-one-to-one"
_ACTOR_MASK_PROTOCOL = "streetgs-projected-cuboid-silhouette-union"


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



def parse_camera_channels(value: str) -> tuple[str, ...]:
    """Parse all or a unique comma-separated nuScenes camera list."""

    stripped = value.strip()
    if stripped.lower() == "all":
        return tuple(NUSCENES_CAMERA_CHANNELS)
    components = tuple(
        component.strip().upper() for component in stripped.split(",")
    )
    if not components or any(not component for component in components):
        raise argparse.ArgumentTypeError(
            "cameras must be 'all' or comma-separated channel names"
        )
    unknown = set(components) - set(NUSCENES_CAMERA_CHANNELS)
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown nuScenes camera channels: " + ", ".join(sorted(unknown))
        )
    if len(components) != len(set(components)):
        raise argparse.ArgumentTypeError("camera channels must be unique")
    return components


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train ArmGS on a nuScenes scene with capture-safe holdout, "
            "colored-LiDAR initialization, exact checkpoint resume, and W&B."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "armgs_default.yaml",
        help="ArmGS YAML configuration",
    )
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    parser.add_argument(
        "--sky-mask-root",
        type=Path,
        help="directory containing <camera_channel>/<sample_data_token>.png masks",
    )
    parser.add_argument(
        "--sky-mask-reject-list",
        type=Path,
        help=(
            "UTF-8 file of sample_data tokens whose raw sky masks are kept "
            "but excluded from sky BCE supervision"
        ),
    )
    parser.add_argument(
        "--colmap-points3d",
        type=Path,
        help=(
            "optional COLMAP points3D.txt triangulated with the dataset camera "
            "poses and already expressed in the nuScenes world frame"
        ),
    )
    parser.add_argument(
        "--scene",
        default="0061",
        help="scene selector: 61, 0061, or scene-0061 (default: 0061)",
    )
    parser.add_argument(
        "--version",
        default="v1.0-trainval",
        help="nuScenes metadata version directory (default: v1.0-trainval)",
    )
    parser.add_argument(
        "--cameras",
        type=parse_camera_channels,
        default=tuple(NUSCENES_CAMERA_CHANNELS),
        help="all or comma-separated nuScenes camera channels (default: all)",
    )
    parser.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="checkpoint to resume")
    parser.add_argument(
        "--iterations",
        type=_positive_int,
        help="override optimization.iterations (total, not additional, steps)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=_positive_int,
        default=1000,
    )
    parser.add_argument("--log-interval", type=_positive_int, default=100)
    parser.add_argument(
        "--image-log-interval",
        type=_non_negative_int,
        default=500,
        help=(
            "W&B train GT/render image interval, independent of scalar logs; "
            "0 disables train image logging (default: 500)"
        ),
    )

    parser.add_argument(
        "--eval-interval",
        type=_non_negative_int,
        default=1000,
        help="held-out evaluation interval; 0 disables periodic evaluation",
    )
    parser.add_argument(
        "--eval-at-end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="evaluate the held-out split after training (default: enabled)",
    )
    parser.add_argument(
        "--eval-lpips",
        action="store_true",
        help="also compute held-out LPIPS with one shared model",
    )
    parser.add_argument(
        "--eval-lpips-net",
        choices=("alex", "vgg", "squeeze"),
        default="alex",
        help="LPIPS backbone (default: alex)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "evaluate --resume at its restored step without training or "
            "writing checkpoints"
        ),
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="enable Weights & Biases metric logging",
    )
    parser.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY", "CamoSplat"),
        help="W&B entity/team (default: WANDB_ENTITY or CamoSplat)",
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "ArmGS-nuScenes"),
        help="W&B project (default: WANDB_PROJECT or ArmGS-nuScenes)",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=os.environ.get("WANDB_RUN_NAME")
        or os.environ.get("WANDB_NAME"),
        help="W&B display name (default: WANDB_RUN_NAME or WANDB_NAME)",
    )
    parser.add_argument(
        "--wandb-run-id",
        default=os.environ.get("WANDB_RUN_ID"),
        help=(
            "existing W&B run ID to resume with resume='allow' "
            "(default: WANDB_RUN_ID)"
        ),
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
        help="W&B mode (default: WANDB_MODE or online)",
    )
    default_wandb_dir = os.environ.get("WANDB_DIR")
    parser.add_argument(
        "--wandb-dir",
        type=Path,
        default=Path(default_wandb_dir) if default_wandb_dir else None,
        help="local W&B directory (default: WANDB_DIR or OUTPUT_DIR/wandb)",
    )
    parser.add_argument(
        "--wandb-fail-fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "abort training on a W&B initialization/logging error "
            "(default: fail soft and continue training)"
        ),
    )
    parser.add_argument(
        "--wandb-log-checkpoint-artifact",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "upload the final training checkpoint as a checksummed W&B "
            "Artifact (default: disabled)"
        ),
    )
    args = parser.parse_args(argv)
    if args.eval_only and args.resume is None:
        parser.error("--eval-only requires --resume")
    return args


def _full_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_nuscenes_dataset_input_identity(
    manifest: Sequence[Any],
    *,
    root: Path,
    version: str,
    sky_mask_reject_list: Path | None = None,
    sky_mask_rejected_count: int = 0,
    colmap_points3d: Path | None = None,
    small_metadata_content_hash_max_bytes: int = (
        _SMALL_METADATA_CONTENT_HASH_MAX_BYTES
    ),
) -> dict[str, Any]:
    """Fingerprint nuScenes tables and all payloads referenced by the manifest."""

    if small_metadata_content_hash_max_bytes < 0:
        raise ValueError("small metadata content-hash limit must be non-negative")
    if sky_mask_rejected_count < 0:
        raise ValueError("sky mask rejected count must be non-negative")
    if sky_mask_reject_list is None and sky_mask_rejected_count:
        raise ValueError("sky mask rejected count requires a reject-list file")
    nested_version_directory = root / version
    version_directory = (
        root if root.name == version else nested_version_directory
    )
    metadata_paths = sorted(version_directory.glob("*.json"))
    if not metadata_paths:
        raise FileNotFoundError(
            f"nuScenes metadata JSON files were not found: {version_directory}"
        )

    sources: list[tuple[str, Path, bool]] = [
        ("metadata", path, True) for path in metadata_paths
    ]
    resolved_reject_list: Path | None = None
    if sky_mask_reject_list is not None:
        resolved_reject_list = sky_mask_reject_list.resolve(strict=True)
        sources.append(("sky_mask_reject_list", resolved_reject_list, True))
    resolved_colmap_points3d: Path | None = None
    if colmap_points3d is not None:
        resolved_colmap_points3d = colmap_points3d.resolve(strict=True)
        sources.append(("colmap_points3d", resolved_colmap_points3d, True))
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
        key = str(resolved)
        entry = aggregated.setdefault(
            key,
            {"path": resolved, "roles": set(), "is_metadata": False},
        )
        entry["roles"].add(role)
        entry["is_metadata"] = entry["is_metadata"] or is_metadata

    file_records: list[dict[str, Any]] = []
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
            if (
                after.st_size != before.st_size
                or after.st_mtime_ns != before.st_mtime_ns
            ):
                raise RuntimeError(
                    f"dataset metadata changed while fingerprinting: {path}"
                )
            record["verification"] = "stat_identity+full_content_sha256"
        file_records.append(record)

    canonical_payload = {
        "version": _NUSCENES_DATASET_INPUT_IDENTITY_VERSION,
        "small_metadata_content_hash_max_bytes": (
            small_metadata_content_hash_max_bytes
        ),
        "files": file_records,
    }
    if resolved_reject_list is not None:
        canonical_payload["sky_mask_reject_list"] = str(resolved_reject_list)
        canonical_payload["sky_mask_rejected_count"] = sky_mask_rejected_count
    if resolved_colmap_points3d is not None:
        canonical_payload["colmap_points3d"] = str(resolved_colmap_points3d)
    serialized = json.dumps(
        canonical_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    identity = {
        "version": _NUSCENES_DATASET_INPUT_IDENTITY_VERSION,
        "digest_sha256": hashlib.sha256(serialized).hexdigest(),
        "file_count": len(file_records),
        "metadata_file_count": len(metadata_paths),
        "stat_identity_fields": ["resolved_path", "size_bytes", "mtime_ns"],
        "frame_payload_verification": "stat_identity",
        "small_metadata_verification": "stat_identity+full_content_sha256",
        "small_metadata_content_hash_max_bytes": (
            small_metadata_content_hash_max_bytes
        ),
    }
    if resolved_reject_list is not None:
        identity["sky_mask_reject_list"] = str(resolved_reject_list)
        identity["sky_mask_rejected_count"] = sky_mask_rejected_count
    if resolved_colmap_points3d is not None:
        identity["colmap_points3d"] = str(resolved_colmap_points3d)
    return identity


def _initialize_wandb(
    args: argparse.Namespace,
    *,
    config: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
) -> Any | None:
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError:
        _handle_wandb_failure(
            "initialization",
            ImportError("W&B logging requested but 'wandb' is not installed"),
            fail_fast=bool(getattr(args, "wandb_fail_fast", False)),
        )
        return None

    fail_fast = bool(getattr(args, "wandb_fail_fast", False))
    wandb_directory = args.wandb_dir or (args.output_dir / "wandb")
    try:
        wandb_directory.mkdir(parents=True, exist_ok=True)
    except Exception as error:
        _handle_wandb_failure(
            "local directory creation", error, fail_fast=fail_fast
        )
        return None
    dataset_input_identity = run_metadata.get("dataset_input_identity")
    dataset_input_identity_sha256 = (
        dataset_input_identity.get("digest_sha256")
        if isinstance(dataset_input_identity, Mapping)
        else None
    )
    wandb_config = {
        "armgs": copy.deepcopy(dict(config)),
        "dataset": {
            "type": "nuscenes",
            "root": run_metadata["nuscenes_root"],
            "version": run_metadata["version"],
            "scene": run_metadata["scene"],
            "sky_mask_root": run_metadata.get("sky_mask_root"),
            "sky_mask_count": run_metadata.get("sky_mask_count", 0),
            "sky_mask_reject_list": run_metadata.get("sky_mask_reject_list"),
            "sky_mask_rejected_count": run_metadata.get(
                "sky_mask_rejected_count", 0
            ),
            "camera_channels": run_metadata["camera_channels"],
            "input_identity_sha256": dataset_input_identity_sha256,
        },
        "split": {
            "train_rows": len(run_metadata["train_source_indices"]),
            "eval_rows": len(run_metadata["eval_source_indices"]),
        },
        "evaluation": {
            "interval": int(getattr(args, "eval_interval", 1000)),
            "at_end": bool(getattr(args, "eval_at_end", True)),
            "lpips": bool(getattr(args, "eval_lpips", False)),
            "lpips_net": str(getattr(args, "eval_lpips_net", "alex")),
            "eval_only": bool(getattr(args, "eval_only", False)),
            "metric_protocols": {
                "psnr": _PSNR_PROTOCOL,
                "ssim": _SSIM_PROTOCOL,
                "lpips": (
                    _LPIPS_PROTOCOL
                    if bool(getattr(args, "eval_lpips", False))
                    else None
                ),
                "actor_mask": _ACTOR_MASK_PROTOCOL,
            },
        },
        "logging": {
            "scalar_interval": int(getattr(args, "log_interval", 100)),
            "image_interval": int(getattr(args, "image_log_interval", 500)),
            "fail_fast": bool(getattr(args, "wandb_fail_fast", False)),
            "checkpoint_artifact": bool(
                getattr(args, "wandb_log_checkpoint_artifact", False)
            ),
        },
    }
    init_options: dict[str, Any] = {}
    explicit_run_id = getattr(args, "wandb_run_id", None)
    requested_run_id = str(explicit_run_id).strip() if explicit_run_id else None
    resume_source = "explicit" if requested_run_id else "new"
    existing_sidecar = args.output_dir / "wandb_run.json"
    if requested_run_id is None and existing_sidecar.is_file():
        try:
            sidecar_payload = json.loads(existing_sidecar.read_text(encoding="utf-8"))
            if not isinstance(sidecar_payload, Mapping):
                raise ValueError("sidecar root must be a mapping")
            sidecar_run_id = sidecar_payload.get("run_id")
            if not isinstance(sidecar_run_id, str) or not sidecar_run_id.strip():
                raise ValueError("sidecar run_id must be a non-empty string")
            requested_run_id = sidecar_run_id.strip()
            resume_source = "sidecar"
        except Exception as error:
            _handle_wandb_failure(
                "run-ID sidecar read", error, fail_fast=fail_fast
            )
    if requested_run_id:
        init_options.update(id=str(requested_run_id), resume="allow")
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
        _handle_wandb_failure(
            "initialization",
            error,
            fail_fast=fail_fast,
        )
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
                args.output_dir / "wandb_run.json",
                json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
            )
        except Exception as error:
            _handle_wandb_failure(
                "run-ID sidecar write",
                error,
                fail_fast=fail_fast,
            )
    else:
        _handle_wandb_failure(
            "run-ID sidecar write",
            RuntimeError("W&B run has no run ID"),
            fail_fast=fail_fast,
        )
    return run


def _handle_wandb_failure(
    operation: str,
    error: BaseException,
    *,
    fail_fast: bool,
) -> bool:
    message = f"W&B {operation} failed: {error}"
    if fail_fast:
        raise RuntimeError(message) from error
    print(f"warning: {message}; continuing without aborting training", file=sys.stderr)
    return False


_LEGACY_TRAIN_SCALAR_KEYS = frozenset(
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
) -> bool:
    step = int(record["step"])
    payload = {
        (f"train/{key}" if key in _LEGACY_TRAIN_SCALAR_KEYS else key): value
        for key, value in record.items()
        if key != "step"
    }
    if not payload:
        return True
    try:
        run.log(payload, step=step)
    except Exception as error:
        return _handle_wandb_failure(
            "metric/image logging", error, fail_fast=fail_fast
        )
    return True


def _single_hwc_rgb(image: Any, label: str) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{label} must be a Tensor")
    value = image.detach()
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError(f"{label} preview requires a singleton batch")
        value = value[0]
    if value.ndim != 3:
        raise ValueError(f"{label} must have shape [H,W,3] or [1,H,W,3]")
    if value.shape[-1] == 3:
        pass
    elif value.shape[0] == 3:
        value = value.permute(1, 2, 0)
    else:
        raise ValueError(f"{label} must contain exactly three RGB channels")
    value = value.to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} contains non-finite values")
    return value.clamp(0.0, 1.0)


def _wandb_gt_render_payload(
    batch: Any,
    output: Any,
    *,
    frame_index: int | None = None,
    source_index: int | None = None,
) -> dict[str, Any]:
    try:
        import wandb
    except ImportError as error:
        raise ImportError(
            "W&B image logging requested but 'wandb' is not installed"
        ) from error

    target = _single_hwc_rgb(batch.target_rgb, "target RGB")
    rendered = _single_hwc_rgb(output.rendering.rgb, "rendered RGB")
    if target.shape != rendered.shape:
        raise ValueError(
            "GT and rendered RGB shapes must match for W&B comparison: "
            f"{list(target.shape)} != {list(rendered.shape)}"
        )
    comparison = torch.cat((target, rendered), dim=1)
    comparison = comparison.mul(255.0).round().to(torch.uint8).numpy()
    view = batch.view
    camera_id = int(torch.as_tensor(view.camera_id).detach().cpu().item())
    timestamp_ns = int(torch.as_tensor(view.timestamp).detach().cpu().item())
    training_row = (
        None
        if view.training_row is None
        else int(torch.as_tensor(view.training_row).detach().cpu().item())
    )
    if frame_index is None:
        frame_index = getattr(view, "frame_index", None)
    step = int(getattr(output, "step", 0))
    caption_fields = [
        f"Step: {step}",
        f"Camera: {_camera_channel_name(camera_id)} ({camera_id})",
        f"Training row: {training_row}",
        f"Timestamp ns: {timestamp_ns}",
    ]
    if frame_index is not None:
        caption_fields.append(f"Frame: {int(frame_index)}")
    if source_index is not None:
        caption_fields.append(f"Source row: {int(source_index)}")
    caption_fields.append("Left: GT | Right: Render")
    payload: dict[str, Any] = {
        "train/gt_vs_render": wandb.Image(
            comparison,
            caption=" | ".join(caption_fields),
        ),
        "train/image_camera_id": camera_id,
        "train/image_camera": _camera_channel_name(camera_id),
        "train/image_timestamp_ns": str(timestamp_ns),
    }
    if training_row is not None:
        payload["train/image_training_row"] = training_row
    if frame_index is not None:
        payload["train/image_frame_index"] = int(frame_index)
    if source_index is not None:
        payload["train/image_source_index"] = int(source_index)
    return payload


def _wandb_image_payload_factory(
    batch: Any,
    output: Any,
    *,
    fail_fast: bool = False,
    frame_index: int | None = None,
    source_index: int | None = None,
    training_manifest: Sequence[Any] | None = None,
    training_source_indices: Sequence[int] | None = None,
) -> Mapping[str, Any]:
    try:
        if training_manifest is not None:
            if batch.view.training_row is None:
                raise ValueError("train image batch has no training row")
            training_row = int(
                torch.as_tensor(batch.view.training_row).detach().cpu().item()
            )
            if not 0 <= training_row < len(training_manifest):
                raise IndexError("train image training row is out of range")
            frame_index = int(training_manifest[training_row].frame_index)
            if training_source_indices is not None:
                if len(training_source_indices) != len(training_manifest):
                    raise ValueError(
                        "training source-index mapping length does not match manifest"
                    )
                source_index = int(training_source_indices[training_row])
        return _wandb_gt_render_payload(
            batch,
            output,
            frame_index=frame_index,
            source_index=source_index,
        )
    except Exception as error:
        _handle_wandb_failure(
            "train image creation", error, fail_fast=fail_fast
        )
        return {}


def _finish_wandb(
    run: Any,
    *,
    exit_code: int = 0,
    fail_fast: bool = False,
) -> bool:
    try:
        run.finish(exit_code=int(exit_code))
    except Exception as error:
        return _handle_wandb_failure(
            "run finalization", error, fail_fast=fail_fast
        )
    return True


def _update_wandb_summary(
    run: Any,
    values: Mapping[str, Any],
    *,
    fail_fast: bool = False,
) -> bool:
    try:
        for key, value in values.items():
            run.summary[key] = value
    except Exception as error:
        return _handle_wandb_failure(
            "summary update", error, fail_fast=fail_fast
        )
    return True


def _log_checkpoint_artifact(
    run: Any,
    checkpoint: Path,
    *,
    metadata: Mapping[str, Any],
    fail_fast: bool = False,
) -> dict[str, Any] | None:
    try:
        import wandb

        resolved = checkpoint.resolve(strict=True)
        artifact_metadata = {
            **dict(metadata),
            "sha256": _full_file_sha256(resolved),
            "size_bytes": resolved.stat().st_size,
        }
        raw_name = (
            f"armgs-{artifact_metadata.get('scene', 'run')}-"
            f"{getattr(run, 'id', 'unknown')}-checkpoint"
        )
        artifact_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw_name).strip("-")
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            metadata=artifact_metadata,
        )
        artifact.add_file(str(resolved), name="final.pt")
        run.log_artifact(artifact)
    except Exception as error:
        _handle_wandb_failure(
            "checkpoint Artifact upload", error, fail_fast=fail_fast
        )
        return None
    return artifact_metadata


def _without_rng_side_effects(
    callback: Any,
    *,
    device: torch.device | str,
) -> Any:
    resolved_device = torch.device(device)
    cuda_devices: list[int] = []
    if resolved_device.type == "cuda":
        cuda_devices.append(
            torch.cuda.current_device()
            if resolved_device.index is None
            else resolved_device.index
        )
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        return callback()



def _camera_channel_name(camera_id: int) -> str:
    if 0 <= camera_id < len(NUSCENES_CAMERA_CHANNELS):
        return NUSCENES_CAMERA_CHANNELS[camera_id]
    return f"CAMERA_{camera_id}"


def _gt_render_comparison(target: Any, rendered: Any) -> Any:
    target_rgb = _single_hwc_rgb(target, "target RGB")
    rendered_rgb = _single_hwc_rgb(rendered, "rendered RGB")
    if target_rgb.shape != rendered_rgb.shape:
        raise ValueError(
            "GT and rendered RGB shapes must match for evaluation preview: "
            f"{list(target_rgb.shape)} != {list(rendered_rgb.shape)}"
        )
    return (
        torch.cat((target_rgb, rendered_rgb), dim=1)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )


def _atomic_save_png(path: Path, pixels: Any) -> None:
    try:
        from PIL import Image
    except ImportError as error:
        raise ImportError(
            "evaluation preview export requires the optional Pillow dependency"
        ) from error

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            Image.fromarray(pixels, mode="RGB").save(handle, format="PNG")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _weighted_evaluation_summary(
    per_camera: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_images": sum(int(item["num_images"]) for item in per_camera.values()),
        "num_actor_images": sum(
            int(item["num_actor_images"]) for item in per_camera.values()
        ),
        "valid_pixels": sum(
            int(item["valid_pixels"]) for item in per_camera.values()
        ),
        "actor_pixels": sum(
            int(item["actor_pixels"]) for item in per_camera.values()
        ),
    }
    for metric, count_key in (
        ("psnr", "num_images"),
        ("ssim", "num_images"),
        ("lpips", "num_images"),
        ("actor_psnr", "num_actor_images"),
    ):
        weighted_sum = 0.0
        weight_sum = 0
        for item in per_camera.values():
            value = item[metric]
            weight = int(item[count_key])
            if value is not None and weight:
                weighted_sum += float(value) * weight
                weight_sum += weight
        summary[metric] = weighted_sum / weight_sum if weight_sum else None
    return summary


def _log_evaluation_to_wandb(
    run: Any,
    record: Mapping[str, Any],
    previews: Mapping[str, Any],
    *,
    fail_fast: bool = False,
) -> bool:
    try:
        import wandb
    except ImportError:
        return _handle_wandb_failure(
            "evaluation logging",
            ImportError(
                "W&B evaluation image logging requested but 'wandb' is not installed"
            ),
            fail_fast=fail_fast,
        )

    try:
        payload: dict[str, Any] = {}
        for metric, value in record["aggregate"].items():
            if value is not None:
                payload[f"eval/{metric}"] = value
        for camera_name, summary in record["per_camera"].items():
            for metric, value in summary.items():
                if value is not None:
                    payload[f"eval/{camera_name}/{metric}"] = value
        for camera_name, comparison in previews.items():
            payload[f"eval/{camera_name}/gt_vs_render"] = wandb.Image(
                comparison,
                caption=f"{camera_name}: Left GT | Right Render",
            )
        run.log(payload, step=int(record["step"]))
    except Exception as error:
        return _handle_wandb_failure(
            "evaluation logging", error, fail_fast=fail_fast
        )
    return True


def evaluate_nuscenes_split(
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
    evaluation_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Render and score every held-out frame using novel-view embeddings."""

    if not manifest:
        raise ValueError("held-out evaluation manifest cannot be empty")
    if step < 0:
        raise ValueError("evaluation step must be non-negative")
    if actor_box_scale <= 0:
        raise ValueError("actor_box_scale must be positive")

    accumulators: dict[str, EvaluationAccumulator] = {}
    preview_arrays: dict[str, Any] = {}
    preview_paths: dict[str, str] = {}
    previous_training = renderer.training
    renderer.eval()
    try:
        with torch.inference_mode():
            for frame in manifest:
                batch = canonical_frame_to_training_batch(
                    frame, training_row=None, device=device
                )
                prediction = renderer(batch.view).rgb.clamp(0.0, 1.0)
                camera_name = _camera_channel_name(int(frame.camera_id))
                accumulator = accumulators.setdefault(
                    camera_name,
                    EvaluationAccumulator(lpips_metric=lpips_metric),
                )
                projected_actor_mask = project_actor_boxes_to_mask(
                    frame,
                    manifest.actor_tracks,
                    box_scale=actor_box_scale,
                )
                actor_mask = projected_actor_mask if projected_actor_mask.any() else None
                accumulator.update(
                    prediction,
                    batch.target_rgb,
                    actor_mask=actor_mask,
                )
                if camera_name not in preview_arrays:
                    comparison = _gt_render_comparison(
                        batch.target_rgb, prediction
                    )
                    preview_path = (
                        output_directory
                        / "evaluation"
                        / f"step_{step:08d}_{camera_name}_gt_render.png"
                    )
                    _atomic_save_png(preview_path, comparison)
                    preview_arrays[camera_name] = comparison
                    preview_paths[camera_name] = str(preview_path)
    finally:
        renderer.train(previous_training)

    per_camera = {
        camera_name: accumulators[camera_name].summary()
        for camera_name in sorted(accumulators)
    }
    record: dict[str, Any] = {
        "event": "held_out_evaluation",
        "step": int(step),
        "split": "eval",
        "aggregate": _weighted_evaluation_summary(per_camera),
        "per_camera": per_camera,
        "previews": {
            camera_name: preview_paths[camera_name]
            for camera_name in sorted(preview_paths)
        },
        "policy": dict(evaluation_policy or {}),
    }
    json_path = output_directory / "evaluation" / f"step_{step:08d}.json"
    _atomic_write_text(
        json_path,
        json.dumps(record, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(record, sort_keys=True), flush=True)
    if wandb_run is not None:
        _log_evaluation_to_wandb(
            wandb_run,
            record,
            preview_arrays,
            fail_fast=wandb_fail_fast,
        )
    return record


def _execute_training_and_evaluation(
    trainer: ArmGSTrainer,
    manifest: Any,
    *,
    total_iterations: int,
    device: torch.device | str,
    checkpoint_interval: int,
    log_interval: int,
    checkpoint_callback: Any,
    log_callback: Any | None,
    log_payload_factory: Any | None,
    image_log_interval: int | None = None,
    evaluation_interval: int,
    evaluate_at_end: bool,
    eval_only: bool,
    evaluation_callback: Any,
    after_training_callback: Any,
) -> int | None:
    """Run training/evaluation without re-firing callbacks for restored steps."""

    if evaluation_interval < 0:
        raise ValueError("evaluation_interval must be non-negative")
    if image_log_interval is not None and image_log_interval < 0:
        raise ValueError("image_log_interval must be non-negative")
    last_evaluation_step: int | None = None

    def completed_step_callback(step: int) -> None:
        nonlocal last_evaluation_step
        if evaluation_interval and step % evaluation_interval == 0:
            evaluation_callback(step)
            last_evaluation_step = step

    if eval_only:
        evaluation_callback(trainer.step)
        return trainer.step

    train_until(
        trainer,
        manifest,
        total_iterations=total_iterations,
        device=device,
        checkpoint_interval=checkpoint_interval,
        log_interval=log_interval,
        checkpoint_callback=checkpoint_callback,
        log_callback=log_callback,
        log_payload_factory=log_payload_factory,
        payload_interval=image_log_interval,
        completed_step_callback=completed_step_callback,
    )
    after_training_callback()
    if evaluate_at_end and last_evaluation_step != trainer.step:
        evaluation_callback(trainer.step)
        last_evaluation_step = trainer.step
    return last_evaluation_step




def run(args: argparse.Namespace) -> Path:
    config = _runtime_config(load_config(args.config), args.iterations)
    if args.eval_only and args.resume is None:
        raise ValueError("--eval-only requires --resume")
    total_iterations = int(config["optimization"]["iterations"])
    device = resolve_device(args.device)
    root = args.nuscenes_root.resolve(strict=True)
    sky_mask_root = (
        args.sky_mask_root.resolve(strict=True)
        if args.sky_mask_root is not None else None
    )
    sky_mask_reject_list = (
        args.sky_mask_reject_list.resolve(strict=True)
        if args.sky_mask_reject_list is not None
        else None
    )
    colmap_points3d = (
        args.colmap_points3d.resolve(strict=True)
        if args.colmap_points3d is not None
        else None
    )
    sky_mask_reject_tokens = (
        parse_nuscenes_sky_mask_reject_list(sky_mask_reject_list)
        if sky_mask_reject_list is not None
        else frozenset()
    )
    camera_channels = tuple(args.cameras)
    normalized_scene = normalize_nuscenes_scene_name(args.scene)

    manifest = load_nuscenes_manifest(
        root,
        scene=args.scene,
        version=args.version,
        camera_channels=camera_channels,
        sky_mask_root=sky_mask_root,
        sky_mask_reject_tokens=sky_mask_reject_tokens,
        require_lidar=True,
    )
    split_config = config.get("data", {}).get("split")
    if not isinstance(split_config, dict):
        raise ValueError("configuration is missing 'data.split'")
    split = periodic_train_eval_split(
        manifest,
        every=int(split_config["every"]),
        offset=int(split_config.get("offset", 0)),
        start_position=int(split_config.get("start_position", 0)),
    )
    loss = build_loss(config)
    validate_training_supervision(split.train_manifest, loss)

    initialization = build_initialization_config(config)
    actor_box_scale = float(config["initialization"].get("actor_box_scale", 1.0))
    point_clouds = collect_colored_lidar_point_clouds(
        split.train_manifest,
        actor_box_scale=actor_box_scale,
    )
    lidar_background_point_count = int(point_clouds.background.points.shape[0])
    sfm_point_count = 0
    if colmap_points3d is not None:
        sfm_points, sfm_colors = load_colmap_points3d_text(colmap_points3d)
        sfm_point_count = int(sfm_points.shape[0])
        point_clouds = merge_sfm_background(
            point_clouds,
            sfm_points,
            sfm_colors,
            voxel_size=initialization.voxel_size,
        )
    bounds = conservative_scene_bounds(
        split.train_manifest,
        point_clouds,
        minimum_padding=max(3.0 * initialization.initial_scale, 1.0e-3),
    )
    scene_extent = camera_scene_extent(split.train_manifest)
    scene = build_scene_from_point_clouds(
        split.train_manifest,
        point_clouds,
        initialization=initialization,
        sky=build_sky(config),
        require_all_actor_points=False,
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

    sky_mask_rejected_count = sum(
        not frame.sky_supervision_valid for frame in manifest
    )
    run_metadata = {
        "dataset": "nuscenes",
        "nuscenes_root": str(root),
        "version": args.version,
        "sky_mask_root": (
            str(sky_mask_root) if sky_mask_root is not None else None
        ),
        "sky_mask_count": sum(frame.sky_mask_path is not None for frame in manifest),
        "sky_mask_reject_list": (
            str(sky_mask_reject_list)
            if sky_mask_reject_list is not None
            else None
        ),
        "sky_mask_rejected_count": sky_mask_rejected_count,
        "scene": normalized_scene,
        "camera_channels": list(camera_channels),
        "camera_ids": sorted({frame.camera_id for frame in manifest}),
        "train_source_indices": list(split.train_source_indices),
        "eval_source_indices": list(split.eval_source_indices),
        "allow_actors_without_lidar_points": True,
        "initialization_modalities": (
            ["lidar", "sfm"] if colmap_points3d is not None else ["lidar"]
        ),
        "paper_lidar_plus_sfm_initialization": colmap_points3d is not None,
        "colmap_points3d": (
            str(colmap_points3d) if colmap_points3d is not None else None
        ),
        "lidar_background_point_count_before_merge": (
            lidar_background_point_count
        ),
        "sfm_point_count_before_merge": sfm_point_count,
        "merged_background_point_count": int(
            point_clouds.background.points.shape[0]
        ),
        "background_scene_extent": scene_extent,
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
        "dataset_input_identity": build_nuscenes_dataset_input_identity(
            manifest,
            root=root,
            version=args.version,
            sky_mask_reject_list=sky_mask_reject_list,
            sky_mask_rejected_count=sky_mask_rejected_count,
            colmap_points3d=colmap_points3d,
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

    output_directory = args.output_dir
    checkpoints = output_directory / "checkpoints"
    _atomic_write_text(
        output_directory / "resolved_config.yaml",
        yaml.safe_dump(config, sort_keys=False),
    )
    _atomic_write_text(
        output_directory / "run_metadata.json",
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n",
    )

    def checkpoint_callback(step: int) -> None:
        save_training_checkpoint(
            checkpoints / f"step_{step:08d}.pt",
            trainer,
            config,
            run_metadata,
        )

    wandb_run = _initialize_wandb(
        args,
        config=config,
        run_metadata=run_metadata,
    )
    evaluation_policy = {
        "interval": int(args.eval_interval),
        "at_end": bool(args.eval_at_end),
        "lpips": bool(args.eval_lpips),
        "lpips_net": args.eval_lpips_net,
        "eval_only": bool(args.eval_only),
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
    latest_evaluation: dict[str, Any] | None = None
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
            nonlocal latest_evaluation

            def evaluate() -> dict[str, Any]:
                return evaluate_nuscenes_split(
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
                )

            latest_evaluation = _without_rng_side_effects(
                evaluate,
                device=device,
            )

        def after_training_callback() -> None:
            save_training_checkpoint(
                final_path,
                trainer,
                config,
                run_metadata,
            )

        _execute_training_and_evaluation(
            trainer,
            split.train_manifest,
            total_iterations=total_iterations,
            device=device,
            checkpoint_interval=args.checkpoint_interval,
            log_interval=args.log_interval,
            checkpoint_callback=checkpoint_callback,
            log_callback=(
                (
                    lambda record: _log_to_wandb(
                        wandb_run,
                        record,
                        fail_fast=args.wandb_fail_fast,
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
        if wandb_run is not None:
            summary_values: dict[str, Any] = {
                "checkpoint": str(final_path),
                "completed_steps": trainer.step,
            }
            if latest_evaluation is not None:
                for metric, value in latest_evaluation["aggregate"].items():
                    if value is not None:
                        summary_values[f"eval/{metric}"] = value
            _update_wandb_summary(
                wandb_run,
                summary_values,
                fail_fast=args.wandb_fail_fast,
            )
            if args.wandb_log_checkpoint_artifact and not args.eval_only:
                artifact_metadata = _log_checkpoint_artifact(
                    wandb_run,
                    final_path,
                    metadata={
                        "dataset": "nuscenes",
                        "scene": normalized_scene,
                        "step": int(trainer.step),
                        "dataset_input_identity_sha256": run_metadata[
                            "dataset_input_identity"
                        ]["digest_sha256"],
                    },
                    fail_fast=args.wandb_fail_fast,
                )
                if artifact_metadata is not None:
                    _update_wandb_summary(
                        wandb_run,
                        {
                            "checkpoint_artifact/sha256": artifact_metadata[
                                "sha256"
                            ],
                            "checkpoint_artifact/size_bytes": artifact_metadata[
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
        print(
            json.dumps({"checkpoint": str(final_path)}, sort_keys=True),
            flush=True,
        )
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
