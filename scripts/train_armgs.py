#!/usr/bin/env python3
"""Train ArmGS on one canonicalized KITTI sequence.

The CLI deliberately keeps dataset conversion, split construction, scene
initialization, and optimization in one auditable vertical slice. Scene
initialization consumes only training captures; evaluation RGB and LiDAR are
never used to seed Gaussians.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import torch
import yaml

# Make a source checkout directly runnable without requiring an editable install.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

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
from armgs.data import (
    CanonicalDatasetManifest,
    load_kitti_manifest,
    periodic_train_eval_split,
)
from armgs.geometry import quaternion_to_rotation_matrix
from armgs.losses import ArmGSLoss
from armgs.pipeline import ArmGSCompositeRenderer
from armgs.scene_builder import (
    CanonicalScenePointClouds,
    build_scene_from_point_clouds,
    collect_colored_lidar_point_clouds,
)
from armgs.training import ArmGSTrainer


_CHECKPOINT_FORMAT_VERSION = 2
_DATASET_INPUT_IDENTITY_VERSION = 1
_SMALL_METADATA_CONTENT_HASH_MAX_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class SceneBounds:
    """World-space hash-grid bounds and their geometry-only scale."""

    aabb_min: tuple[float, float, float]
    aabb_max: tuple[float, float, float]
    scene_scale: float


def camera_scene_extent(
    manifest: CanonicalDatasetManifest,
    *,
    radius_padding: float = 1.1,
    minimum_extent: float = 1.0e-3,
) -> float:
    """Return the official 3DGS camera-normalization radius.

    Density control and mean-position learning rates use this value.  The
    geometry AABB remains a separate contract used only by spatial encoders.
    """

    if not math.isfinite(radius_padding) or radius_padding <= 0.0:
        raise ValueError("radius_padding must be finite and positive")
    if not math.isfinite(minimum_extent) or minimum_extent <= 0.0:
        raise ValueError("minimum_extent must be finite and positive")
    centers = torch.stack(
        [frame.camera_to_world[:3, 3].detach().to(torch.float64) for frame in manifest]
    )
    center = centers.mean(dim=0)
    radius = float(torch.linalg.vector_norm(centers - center, dim=-1).amax().item())
    extent = max(radius * radius_padding, minimum_extent)
    if not math.isfinite(extent):
        raise ValueError("computed camera scene extent must be finite")
    return extent


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_camera_ids(value: str) -> tuple[int, ...]:
    """Parse a comma-separated, unique list of non-negative camera ids."""

    components = [component.strip() for component in value.split(",")]
    if not components or any(not component for component in components):
        raise argparse.ArgumentTypeError(
            "camera ids must be comma-separated integers, for example 2,3"
        )
    try:
        camera_ids = tuple(int(component) for component in components)
    except ValueError as error:
        raise argparse.ArgumentTypeError("camera ids must be integers") from error
    if any(camera_id < 0 for camera_id in camera_ids):
        raise argparse.ArgumentTypeError("camera ids must be non-negative")
    if len(camera_ids) != len(set(camera_ids)):
        raise argparse.ArgumentTypeError("camera ids must be unique")
    return camera_ids


def parse_camera_directory_mappings(
    entries: Sequence[str] | None,
    camera_ids: Sequence[int],
    *,
    option_name: str,
) -> dict[int, Path]:
    """Parse repeated CAMERA_ID=DIR mask-directory arguments."""

    requested = set(camera_ids)
    mappings: dict[int, Path] = {}
    for entry in entries or ():
        camera_text, separator, directory_text = entry.partition("=")
        if not separator or not camera_text.strip() or not directory_text.strip():
            raise ValueError(f"{option_name} entries must use CAMERA_ID=DIR")
        try:
            camera_id = int(camera_text)
        except ValueError as error:
            raise ValueError(
                f"{option_name} camera id must be an integer: {camera_text!r}"
            ) from error
        if camera_id not in requested:
            raise ValueError(
                f"{option_name} references unrequested camera id {camera_id}"
            )
        if camera_id in mappings:
            raise ValueError(f"{option_name} repeats camera id {camera_id}")
        mappings[camera_id] = Path(directory_text)
    return mappings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train ArmGS on a KITTI sequence with leak-free periodic holdout, "
            "colored-LiDAR initialization, gsplat, and exact checkpoint resume."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "armgs_default.yaml",
        help="ArmGS YAML configuration",
    )
    parser.add_argument("--kitti-root", type=Path, required=True)
    parser.add_argument(
        "--camera-ids",
        type=parse_camera_ids,
        default=(2,),
        help="comma-separated KITTI camera ids (default: 2)",
    )
    parser.add_argument(
        "--tracklets",
        type=Path,
        help="optional tracklet_labels.xml (relative paths use --kitti-root)",
    )
    parser.add_argument(
        "--sky-mask-dir",
        action="append",
        default=[],
        metavar="CAMERA_ID=DIR",
        help="repeat once per camera requiring sky supervision",
    )
    parser.add_argument(
        "--actor-mask-dir",
        action="append",
        default=[],
        metavar="CAMERA_ID=DIR",
        help="repeat once per camera requiring actor supervision",
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
        help=(
            "legacy compatibility option; intermediate checkpoints are "
            "disabled and only final.pt is written"
        ),
    )
    parser.add_argument("--log-interval", type=_positive_int, default=100)
    return parser.parse_args(argv)


def _resolve_under_root(root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else root / path


def _resolve_mapping_under_root(
    root: Path, mappings: Mapping[int, Path]
) -> dict[int, Path]:
    return {
        camera_id: path if path.is_absolute() else root / path
        for camera_id, path in mappings.items()
    }

def _full_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_input_identity(
    manifest: CanonicalDatasetManifest,
    *,
    calibration_path: Path,
    poses_path: Path,
    times_path: Path,
    tracklet_path: Path | None = None,
    small_metadata_content_hash_max_bytes: int = (
        _SMALL_METADATA_CONTENT_HASH_MAX_BYTES
    ),
) -> dict[str, Any]:
    """Fingerprint every file that determines the canonical KITTI manifest.

    Every file contributes its resolved path, byte size, and nanosecond mtime.
    Small calibration/pose/time/tracklet metadata also contributes a full-file
    SHA-256. Image, LiDAR, and mask payloads deliberately use stat identity only
    so constructing a checkpoint does not reread the complete dataset. Entries
    and their roles are canonically sorted before the aggregate SHA-256, making
    the result independent of manifest or mapping enumeration order.
    """

    if small_metadata_content_hash_max_bytes < 0:
        raise ValueError("small metadata content-hash limit must be non-negative")

    sources: list[tuple[str, Path, bool]] = [
        ("calibration", calibration_path, True),
        ("poses", poses_path, True),
        ("timestamps", times_path, True),
    ]
    if tracklet_path is not None:
        sources.append(("tracklets", tracklet_path, True))
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
        "version": _DATASET_INPUT_IDENTITY_VERSION,
        "small_metadata_content_hash_max_bytes": (
            small_metadata_content_hash_max_bytes
        ),
        "files": file_records,
    }
    serialized = json.dumps(
        canonical_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "version": _DATASET_INPUT_IDENTITY_VERSION,
        "digest_sha256": hashlib.sha256(serialized).hexdigest(),
        "file_count": len(file_records),
        "stat_identity_fields": ["resolved_path", "size_bytes", "mtime_ns"],
        "frame_payload_verification": "stat_identity",
        "small_metadata_verification": "stat_identity+full_content_sha256",
        "small_metadata_content_hash_max_bytes": (
            small_metadata_content_hash_max_bytes
        ),
    }



def resolve_device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"invalid Torch device {value!r}") from error
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable; gsplat training requires a CUDA build"
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {device.index} is unavailable")
    return device


def validate_training_supervision(
    manifest: CanonicalDatasetManifest,
    loss: ArmGSLoss,
) -> None:
    """Fail before training when strict non-RGB objectives lack labels.

    ArmGS foreground regularization is entropy on rendered actor alpha, so it
    requires actor tracks but no external per-pixel actor target.
    """

    if not loss.require_auxiliary:
        return

    def rows_missing(predicate: Any) -> list[int]:
        return [row for row, frame in enumerate(manifest.frames) if predicate(frame)]

    if loss.lambda_depth > 0.0:
        missing_depth = rows_missing(
            lambda frame: frame.lidar_projection is None
            or frame.lidar_projection.depths.numel() == 0
        )
        if missing_depth:
            raise ValueError(
                "strict depth loss requires projected LiDAR on every training row; "
                f"missing rows {missing_depth}"
            )
    if loss.lambda_sky > 0.0:
        missing_sky = rows_missing(lambda frame: frame.sky_mask_path is None)
        if missing_sky:
            raise ValueError(
                "strict sky loss requires a sky mask on every training row; "
                f"missing rows {missing_sky}"
            )
    if loss.lambda_foreground > 0.0:
        if not manifest.actor_tracks:
            raise ValueError(
                "strict foreground loss requires at least one dynamic actor track"
            )


def conservative_scene_bounds(
    manifest: CanonicalDatasetManifest,
    point_clouds: CanonicalScenePointClouds,
    *,
    padding_fraction: float = 0.05,
    minimum_padding: float = 1.0e-3,
) -> SceneBounds:
    """Bound all initialized geometry with an explicit conservative assumption.

    The raw bound contains background world points and the eight corners of
    every actor-local point-cloud AABB transformed through every track sample.
    It is padded on every side by five percent of the longest raw axis (or
    minimum_padding). ``scene_scale`` is retained as the padded AABB diagonal
    for backward compatibility, but must not be used for density control or
    mean-position learning-rate scaling; use :func:`camera_scene_extent`.
    """

    if not math.isfinite(padding_fraction) or padding_fraction < 0.0:
        raise ValueError("padding_fraction must be finite and non-negative")
    if not math.isfinite(minimum_padding) or minimum_padding <= 0.0:
        raise ValueError("minimum_padding must be finite and positive")

    background = point_clouds.background.points.detach()
    minima = [background.amin(dim=0)]
    maxima = [background.amax(dim=0)]
    tracks = {track.actor_id: track for track in manifest.actor_tracks}
    unknown = set(point_clouds.actors) - set(tracks)
    if unknown:
        raise ValueError(
            f"actor point clouds have no matching tracks: {sorted(unknown)}"
        )

    corner_bits = tuple(itertools.product((0, 1), repeat=3))
    for actor_id, cloud in point_clouds.actors.items():
        local_points = cloud.points.detach()
        local_min = local_points.amin(dim=0)
        local_max = local_points.amax(dim=0)
        corners = torch.stack(
            [
                torch.stack(
                    [
                        local_max[axis] if bit else local_min[axis]
                        for axis, bit in enumerate(bits)
                    ]
                )
                for bits in corner_bits
            ]
        )
        for sample in tracks[actor_id].samples:
            rotation = quaternion_to_rotation_matrix(
                sample.quaternion_wxyz.to(corners)
            )
            translation = sample.translation.to(corners)
            world_corners = corners @ rotation.transpose(-1, -2) + translation
            minima.append(world_corners.amin(dim=0).to(background))
            maxima.append(world_corners.amax(dim=0).to(background))

    raw_min = torch.stack(minima).amin(dim=0)
    raw_max = torch.stack(maxima).amax(dim=0)
    longest_axis = float((raw_max - raw_min).amax().item())
    padding = max(longest_axis * padding_fraction, minimum_padding)
    aabb_min_tensor = raw_min - padding
    aabb_max_tensor = raw_max + padding
    scene_scale = float(
        torch.linalg.vector_norm(aabb_max_tensor - aabb_min_tensor).item()
    )
    if not math.isfinite(scene_scale) or scene_scale <= 0.0:
        raise ValueError("computed scene scale must be finite and positive")
    return SceneBounds(
        tuple(float(value) for value in aabb_min_tensor.tolist()),
        tuple(float(value) for value in aabb_max_tensor.tolist()),
        scene_scale,
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_training_checkpoint(
    path: Path,
    trainer: ArmGSTrainer,
    config: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
) -> None:
    """Atomically persist model/optimizer/RNG/sampler/density and run identity."""

    payload = {
        "format_version": _CHECKPOINT_FORMAT_VERSION,
        "trainer": trainer.state_dict(),
        "config": copy.deepcopy(dict(config)),
        "run_metadata": copy.deepcopy(dict(run_metadata)),
    }
    _atomic_torch_save(payload, path)


def load_training_checkpoint(
    path: Path, *, map_location: torch.device | str
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    try:
        try:
            payload = torch.load(
                path, map_location=map_location, weights_only=True
            )
        except TypeError:
            payload = torch.load(path, map_location=map_location)
    except Exception as error:
        raise ValueError(f"failed to load checkpoint {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a mapping")
    if int(payload.get("format_version", -1)) != _CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported training checkpoint format")
    for key in ("trainer", "config", "run_metadata"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"checkpoint {key!r} must be a mapping")
    return payload


def _config_without_iteration(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(config))
    optimization = normalized.get("optimization")
    if isinstance(optimization, dict):
        optimization["iterations"] = "<total-iteration-override>"
    return normalized


def restore_training_checkpoint(
    trainer: ArmGSTrainer,
    path: Path,
    *,
    map_location: torch.device | str,
    config: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
) -> None:
    checkpoint = load_training_checkpoint(path, map_location=map_location)
    if _config_without_iteration(checkpoint["config"]) != _config_without_iteration(
        config
    ):
        raise ValueError(
            "checkpoint configuration differs from the current run "
            "(only total iterations may change)"
        )
    if checkpoint["run_metadata"] != dict(run_metadata):
        raise ValueError("checkpoint dataset/split identity differs from the current run")
    trainer.load_state_dict(checkpoint["trainer"])


_LOSS_TELEMETRY_FIELDS = (
    ("train/loss", "total"),
    ("train/rgb_l1", "rgb"),
    ("train/ssim_loss", "ssim"),
    ("train/depth_loss", "depth"),
    ("train/sky_loss", "sky"),
    ("train/foreground_loss", "foreground"),
)

_PSNR_MSE_FLOOR = 1.0e-10


def _accumulate_loss_telemetry(
    sums: dict[str, torch.Tensor], output: Any, batch: Any
) -> None:
    """Accumulate detached losses and train-view quality on the active device."""

    for metric, attribute in _LOSS_TELEMETRY_FIELDS:
        value = getattr(output.losses, attribute).detach()
        if value.numel() != 1:
            raise ValueError(f"training loss {attribute!r} must be scalar")
        value = value.reshape(()).to(dtype=torch.float64)
        if metric in sums:
            sums[metric].add_(value)
        else:
            sums[metric] = value.clone()

    rendered_rgb = output.rendering.rgb.detach()
    target_rgb = batch.target_rgb.detach().to(rendered_rgb)
    if rendered_rgb.shape != target_rgb.shape:
        raise ValueError(
            "rendered and target RGB must have matching shapes for telemetry"
        )
    if rendered_rgb.numel() == 0:
        raise ValueError("training RGB tensors must not be empty")

    squared_error = (rendered_rgb - target_rgb).square()
    mse = squared_error.mean(dtype=torch.float64)
    quality_values = {
        "train/psnr": -10.0
        * torch.log10(mse.clamp_min(_PSNR_MSE_FLOOR)),
        "train/ssim": 1.0
        - output.losses.ssim.detach().reshape(()).to(dtype=torch.float64),
    }
    for metric, value in quality_values.items():
        if metric in sums:
            sums[metric].add_(value)
        else:
            sums[metric] = value.clone()


def _accumulate_density_telemetry(
    totals: dict[str, int], output: Any
) -> bool:
    """Add topology-update counts and report whether results were available."""

    updates = getattr(output, "density_updates", None)
    if updates is None:
        return False
    if not isinstance(updates, Mapping):
        raise TypeError("density_updates must be a mapping or None")
    totals["train/density/steps_with_results"] += 1
    totals["train/density/updated_groups"] += len(updates)
    for result in updates.values():
        totals["train/density/topology_changed_groups"] += int(
            bool(result.topology_changed)
        )
        totals["train/density/duplicated_gaussians"] += int(
            result.duplicated_count
        )
        totals["train/density/split_parent_gaussians"] += int(
            result.split_parent_count
        )
        totals["train/density/split_child_gaussians"] += int(
            result.split_child_count
        )
        totals["train/density/pruned_gaussians"] += int(result.pruned_count)
        totals["train/density/opacity_reset_groups"] += int(
            bool(result.opacity_was_reset)
        )
    return True


def _empty_density_telemetry() -> dict[str, int]:
    return {
        "train/density/steps_with_results": 0,
        "train/density/updated_groups": 0,
        "train/density/topology_changed_groups": 0,
        "train/density/duplicated_gaussians": 0,
        "train/density/split_parent_gaussians": 0,
        "train/density/split_child_gaussians": 0,
        "train/density/pruned_gaussians": 0,
        "train/density/opacity_reset_groups": 0,
    }


def _metric_key_component(value: Any, fallback: str) -> str:
    text = str(value).strip()
    normalized = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in text
    ).strip("_")
    return normalized or fallback


def _optimizer_learning_rate_telemetry(trainer: ArmGSTrainer) -> dict[str, float]:
    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is None:
        return {}
    records: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        component = _metric_key_component(group.get("name"), f"group_{index}")
        key = f"train/lr/{component}"
        if key in records:
            key = f"{key}_{index}"
        records[key] = float(group["lr"])
    return records


def _gaussian_count_telemetry(trainer: ArmGSTrainer) -> dict[str, int]:
    scene = trainer.renderer.scene
    background_count = int(scene.background.count)
    actor_count = sum(int(actor.gaussians.count) for actor in scene.actors)
    records = {
        "train/gaussians/total": background_count + actor_count,
        "train/gaussians/background": background_count,
        "train/gaussians/actors": actor_count,
    }
    for index, actor in enumerate(scene.actors):
        actor_id = _metric_key_component(
            getattr(actor, "actor_id", index), f"index_{index}"
        )
        records[f"train/gaussians/actor/{actor_id}"] = int(
            actor.gaussians.count
        )
    return records


def _cuda_memory_telemetry(device: torch.device) -> dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    bytes_per_mib = float(1024**2)
    return {
        "train/cuda/memory_allocated_mib": (
            torch.cuda.memory_allocated(device) / bytes_per_mib
        ),
        "train/cuda/memory_reserved_mib": (
            torch.cuda.memory_reserved(device) / bytes_per_mib
        ),
        "train/cuda/max_memory_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / bytes_per_mib
        ),
        "train/cuda/max_memory_reserved_mib": (
            torch.cuda.max_memory_reserved(device) / bytes_per_mib
        ),
    }


def _loss_log_record(
    trainer: ArmGSTrainer,
    *,
    loss_sums: Mapping[str, torch.Tensor],
    interval_steps: int,
    interval_seconds: float,
    density_totals: Mapping[str, int] | None,
    device: torch.device,
) -> dict[str, Any]:
    if interval_steps <= 0:
        raise ValueError("telemetry interval must contain at least one step")
    if not math.isfinite(interval_seconds) or interval_seconds < 0.0:
        raise ValueError("telemetry interval time must be finite and non-negative")

    record: dict[str, Any] = {
        "step": trainer.step,
        **{
            metric: float((total / interval_steps).cpu().item())
            for metric, total in loss_sums.items()
        },
        **_gaussian_count_telemetry(trainer),
        **_optimizer_learning_rate_telemetry(trainer),
        "train/telemetry/window_steps": interval_steps,
        "train/performance/step_time_seconds": interval_seconds
        / interval_steps,
        "train/performance/steps_per_second": interval_steps
        / max(interval_seconds, sys.float_info.epsilon),
        **_cuda_memory_telemetry(device),
    }
    if density_totals is not None:
        record.update(density_totals)
    return record


def train_until(
    trainer: ArmGSTrainer,
    manifest: CanonicalDatasetManifest,
    *,
    total_iterations: int,
    device: torch.device | str,
    checkpoint_interval: int,
    log_interval: int,
    checkpoint_callback: Any | None = None,
    log_callback: Any | None = None,
    log_payload_factory: Any | None = None,
    payload_interval: int | None = None,
    completed_step_callback: Any | None = None,
) -> None:
    """Advance exactly to total_iterations using checkpointed sampler rows.

    Loss scalars are interval means over newly executed steps. A positive
    ``payload_interval`` invokes ``log_payload_factory`` independently at that
    cadence; zero disables payloads. ``None`` preserves the legacy behavior of
    attaching a payload to each scalar record. Coincident scalar and payload
    records are delivered through one callback.

    ``completed_step_callback`` runs once after each newly completed step and
    after that step's checkpoint callback, if one is due. Restored steps never
    trigger it again.
    """

    if trainer.sampler is None:
        raise ValueError("training CLI requires a stateful sampler")
    if log_payload_factory is not None and log_callback is None:
        raise ValueError("log_payload_factory requires log_callback")
    for name, interval in (
        ("checkpoint_interval", checkpoint_interval),
        ("log_interval", log_interval),
    ):
        if (
            isinstance(interval, bool)
            or not isinstance(interval, int)
            or interval <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    if payload_interval is not None and (
        isinstance(payload_interval, bool)
        or not isinstance(payload_interval, int)
        or payload_interval < 0
    ):
        raise ValueError("payload_interval must be a non-negative integer or None")
    if trainer.step > total_iterations:
        raise ValueError(
            f"checkpoint step {trainer.step} exceeds target iterations {total_iterations}"
        )
    runtime_device = torch.device(device)
    loss_sums: dict[str, torch.Tensor] = {}
    interval_steps = 0
    interval_seconds = 0.0
    density_totals = _empty_density_telemetry()
    density_results_available = False
    empty_passes = 0
    while trainer.step < total_iterations:
        made_progress = False
        for training_row in trainer.sampler:
            made_progress = True
            step_started = time.perf_counter()
            batch = canonical_frame_to_training_batch(
                manifest[training_row], training_row, device=device
            )
            output = trainer.train_step(batch)
            scalar_due = (
                trainer.step % log_interval == 0
                or trainer.step == total_iterations
            )
            if log_payload_factory is None or payload_interval == 0:
                payload_due = False
            elif payload_interval is None:
                payload_due = scalar_due
            else:
                payload_due = trainer.step % payload_interval == 0
            if (
                (scalar_due or payload_due)
                and runtime_device.type == "cuda"
                and torch.cuda.is_available()
            ):
                torch.cuda.synchronize(runtime_device)
            interval_seconds += time.perf_counter() - step_started
            interval_steps += 1
            _accumulate_loss_telemetry(loss_sums, output, batch)
            density_results_available = (
                _accumulate_density_telemetry(density_totals, output)
                or density_results_available
            )

            record: dict[str, Any] | None = None
            if scalar_due:
                record = _loss_log_record(
                    trainer,
                    loss_sums=loss_sums,
                    interval_steps=interval_steps,
                    interval_seconds=interval_seconds,
                    density_totals=(
                        density_totals if density_results_available else None
                    ),
                    device=runtime_device,
                )
                print(json.dumps(record, sort_keys=True), flush=True)
                loss_sums = {}
                interval_steps = 0
                interval_seconds = 0.0
                density_totals = _empty_density_telemetry()
                density_results_available = False
            if payload_due:
                assert log_payload_factory is not None
                extra_payload = log_payload_factory(batch, output)
                if not isinstance(extra_payload, Mapping):
                    raise TypeError("log_payload_factory must return a mapping")
                callback_base = record or {"step": trainer.step}
                duplicate_keys = set(callback_base).intersection(extra_payload)
                if duplicate_keys:
                    raise ValueError(
                        "log payload duplicates scalar keys: "
                        + ", ".join(sorted(duplicate_keys))
                    )
                record = {**callback_base, **dict(extra_payload)}
            if record is not None and log_callback is not None:
                log_callback(record)
            if checkpoint_callback is not None and trainer.step % checkpoint_interval == 0:
                checkpoint_callback(trainer.step)
            if completed_step_callback is not None:
                completed_step_callback(trainer.step)
            if trainer.step == total_iterations:
                return
        if made_progress:
            empty_passes = 0
        else:
            # An exactly exhausted mid-epoch checkpoint needs one empty pass so
            # StatefulShuffleSampler can roll over and materialize the next epoch.
            empty_passes += 1
            if empty_passes > 1:
                raise RuntimeError("stateful sampler produced no rows")


def _runtime_config(
    config: Mapping[str, Any], iteration_override: int | None
) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(config))
    if iteration_override is not None:
        resolved["optimization"]["iterations"] = iteration_override
    iterations = int(resolved["optimization"]["iterations"])
    if iterations <= 0:
        raise ValueError("optimization.iterations must be positive")
    return resolved


def run(args: argparse.Namespace) -> Path:
    config = _runtime_config(load_config(args.config), args.iterations)
    total_iterations = int(config["optimization"]["iterations"])
    device = resolve_device(args.device)
    root = args.kitti_root.resolve()
    camera_ids = tuple(args.camera_ids)
    sky_directories = _resolve_mapping_under_root(
        root,
        parse_camera_directory_mappings(
            args.sky_mask_dir, camera_ids, option_name="--sky-mask-dir"
        ),
    )
    actor_directories = _resolve_mapping_under_root(
        root,
        parse_camera_directory_mappings(
            args.actor_mask_dir, camera_ids, option_name="--actor-mask-dir"
        ),
    )
    tracklet_path = _resolve_under_root(root, args.tracklets)

    manifest = load_kitti_manifest(
        root,
        camera_ids=camera_ids,
        tracklet_path=tracklet_path,
        sky_mask_dirs=sky_directories,
        actor_mask_dirs=actor_directories,
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
    actor_box_scale = float(
        config["initialization"].get("actor_box_scale", 1.0)
    )
    point_clouds = collect_colored_lidar_point_clouds(
        split.train_manifest, actor_box_scale=actor_box_scale
    )
    bounds = conservative_scene_bounds(
        split.train_manifest,
        point_clouds,
        minimum_padding=max(3.0 * initialization.initial_scale, 1.0e-3),
    )
    scene_extent = camera_scene_extent(split.train_manifest)
    sky = build_sky(config)
    scene = build_scene_from_point_clouds(
        split.train_manifest,
        point_clouds,
        initialization=initialization,
        sky=sky,
        require_all_actor_points=True,
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

    run_metadata = {
        "kitti_root": str(root),
        "camera_ids": list(camera_ids),
        "tracklets": str(tracklet_path) if tracklet_path is not None else None,
        "sky_mask_dirs": {
            str(camera_id): str(path)
            for camera_id, path in sorted(sky_directories.items())
        },
        "actor_mask_dirs": {
            str(camera_id): str(path)
            for camera_id, path in sorted(actor_directories.items())
        },
        "train_source_indices": list(split.train_source_indices),
        "eval_source_indices": list(split.eval_source_indices),
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
        "dataset_input_identity": build_dataset_input_identity(
            manifest,
            calibration_path=root / "calib.txt",
            poses_path=root / "poses.txt",
            times_path=root / "times.txt",
            tracklet_path=tracklet_path,
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
    resolved_config = yaml.safe_dump(config, sort_keys=False)
    _atomic_write_text(
        output_directory / "resolved_config.yaml", resolved_config
    )
    _atomic_write_text(
        output_directory / "run_metadata.json",
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n",
    )

    train_until(
        trainer,
        split.train_manifest,
        total_iterations=total_iterations,
        device=device,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        checkpoint_callback=None,
    )
    final_path = checkpoints / "final.pt"
    save_training_checkpoint(final_path, trainer, config, run_metadata)
    return final_path


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
