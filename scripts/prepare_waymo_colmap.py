#!/usr/bin/env python3
"""Prepare paper-resolution Waymo-v2 images for known-pose COLMAP SfM.

The pipeline follows the ArmGS/Street Gaussians initialization contract:
only training views are staged, moving actor cuboids are excluded from SIFT,
dataset PINHOLE intrinsics and OpenCV camera-to-world poses are written into a
known COLMAP model, and only 3D points are triangulated.  CPU SIFT extraction
and matching are the default so the command works with Ubuntu's COLMAP 3.9.1.

``--dry-run`` never writes the COLMAP output directory, although the Waymo
loader may populate its separate decoded-image cache. ``--skip-execution`` is
an inspection-only staging mode; because non-empty outputs are protected, use
a fresh output directory for the subsequent real run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPT_ROOT = REPOSITORY_ROOT / "scripts"
for import_root in (SOURCE_ROOT, SCRIPT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from armgs.data.schema import (  # noqa: E402
    ActorTrack,
    CanonicalDatasetManifest,
    CanonicalFrame,
)
from armgs.data.split import periodic_train_eval_split  # noqa: E402
import prepare_nuscenes_colmap as _known_pose  # noqa: E402


WAYMO_CAMERA_CHANNELS: tuple[str, ...] = (
    "FRONT",
    "FRONT_LEFT",
    "FRONT_RIGHT",
    "SIDE_LEFT",
    "SIDE_RIGHT",
)
_MAPPING_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StagedFrame:
    """One selected Waymo training observation and COLMAP-relative names."""

    source_index: int
    channel: str
    frame: CanonicalFrame
    image_name: str
    mask_name: str
    visible_actor_ids: tuple[int, ...] = ()
    dynamic_pixel_count: int = 0

    def __post_init__(self) -> None:
        if self.source_index < 0:
            raise ValueError("source_index must be non-negative")
        if self.channel not in WAYMO_CAMERA_CHANNELS:
            raise ValueError(f"unknown Waymo camera channel: {self.channel}")
        image_path = Path(self.image_name)
        mask_path = Path(self.mask_name)
        if image_path.is_absolute() or mask_path.is_absolute():
            raise ValueError("staged image and mask names must be relative")
        if not image_path.parts or image_path.parts[0] != self.channel:
            raise ValueError("staged image must be inside its camera folder")
        if self.mask_name != f"{self.image_name}.png":
            raise ValueError("COLMAP mask name must equal image name plus '.png'")
        if self.dynamic_pixel_count < 0:
            raise ValueError("dynamic_pixel_count must be non-negative")


def parse_camera_channels(value: str) -> tuple[str, ...]:
    stripped = value.strip()
    if stripped.lower() == "all":
        return WAYMO_CAMERA_CHANNELS
    channels = tuple(component.strip().upper() for component in stripped.split(","))
    if not channels or any(not channel for channel in channels):
        raise argparse.ArgumentTypeError(
            "cameras must be 'all' or comma-separated Waymo channels"
        )
    unknown = set(channels) - set(WAYMO_CAMERA_CHANNELS)
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown Waymo camera channels: " + ", ".join(sorted(unknown))
        )
    if len(channels) != len(set(channels)):
        raise argparse.ArgumentTypeError("camera channels must be unique")
    return channels


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _nonnegative_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def _positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a floating-point value") from error
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a StreetGS-style known-pose COLMAP model from the "
            "paper-protocol training split of one Waymo-v2 sequence."
        )
    )
    parser.add_argument("--waymo-root", type=Path, required=True)
    parser.add_argument("--parquet-dir", default="training")
    parser.add_argument("--sequence", required=True)
    parser.add_argument(
        "--cameras",
        type=parse_camera_channels,
        default=("FRONT",),
        help="all or comma-separated channels (paper default: FRONT)",
    )
    parser.add_argument("--start-frame", type=_nonnegative_int, default=0)
    parser.add_argument("--end-frame", type=_nonnegative_int)
    parser.add_argument("--target-height", type=_positive_int, default=1066)
    parser.add_argument("--target-width", type=_positive_int, default=1600)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help=(
            "decoded Waymo cache (default: sibling '<output>_waymo_cache'); "
            "dry-run may populate this directory"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-every", type=_positive_int, default=4)
    parser.add_argument("--split-offset", type=_nonnegative_int, default=0)
    parser.add_argument("--split-start-position", type=_nonnegative_int, default=4)
    parser.add_argument("--actor-box-scale", type=_positive_float, default=1.0)
    parser.add_argument(
        "--castrack-path",
        type=Path,
        help=(
            "optional scene-extracted or full CAStrack JSON; when omitted, "
            "the generic Waymo GT actor-track path remains available"
        ),
    )
    parser.add_argument("--colmap-binary", default="colmap")
    parser.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use GPU SIFT extraction/matching (default: disabled for apt COLMAP)",
    )
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--dry-run",
        action="store_true",
        help="validate loader/split/commands without writing COLMAP output",
    )
    execution.add_argument(
        "--skip-execution",
        action="store_true",
        help="inspection-only image/mask staging without invoking COLMAP",
    )
    return parser.parse_args(argv)


def _load_waymo_v2_manifest(
    root: Path,
    *,
    sequence: str,
    parquet_dir: str,
    camera_channels: Sequence[str],
    start_frame: int,
    end_frame: int | None,
    target_size: tuple[int, int],
    cache_dir: Path,
    castrack_path: Path | None,
) -> CanonicalDatasetManifest:
    """Resolve the canonical Waymo loader lazily while its API lands."""

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
        camera_channels=tuple(camera_channels),
        start_frame=start_frame,
        end_frame=end_frame,
        target_size=target_size,
        cache_dir=cache_dir,
        sky_mask_root=None,
        require_lidar=False,
        center_world=True,
        castrack_path=castrack_path,
    )


def _load_waymo_world_center(
    root: Path,
    *,
    sequence: str,
    parquet_dir: str,
) -> torch.Tensor:
    """Load the exact full-context center used by the centered manifest."""

    try:
        from armgs.data.waymo import load_waymo_world_center
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "Waymo world-center loader is unavailable; expected "
            "armgs.data.waymo.load_waymo_world_center"
        ) from error
    center = load_waymo_world_center(
        root,
        sequence=sequence,
        parquet_dir=parquet_dir,
    )
    if center.shape != (3,) or not center.is_floating_point():
        raise ValueError("Waymo full-context world center must be floating [3]")
    if not torch.isfinite(center).all():
        raise ValueError("Waymo full-context world center must be finite")
    return center.detach().to(device="cpu", dtype=torch.float64)


def _camera_channel(camera_id: int) -> str:
    if camera_id < 0 or camera_id >= len(WAYMO_CAMERA_CHANNELS):
        raise ValueError(f"Waymo manifest has unknown camera_id: {camera_id}")
    return WAYMO_CAMERA_CHANNELS[camera_id]


def _staged_image_name(frame: CanonicalFrame, channel: str) -> str:
    suffix = frame.image_path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(
            f"Waymo staged RGB must be JPEG or PNG, found: {frame.image_path}"
        )
    return (Path(channel) / f"{frame.frame_index:06d}{suffix}").as_posix()


def select_training_frames(
    manifest: CanonicalDatasetManifest,
    *,
    every: int = 4,
    offset: int = 0,
    start_position: int = 4,
    target_size: tuple[int, int] = (1066, 1600),
) -> tuple[Any, tuple[StagedFrame, ...]]:
    split = periodic_train_eval_split(
        manifest,
        every=every,
        offset=offset,
        start_position=start_position,
    )
    selected: list[StagedFrame] = []
    names: set[str] = set()
    for source_index in split.train_source_indices:
        frame = manifest.frames[source_index]
        if frame.image_size != target_size:
            raise ValueError(
                f"Waymo frame {frame.frame_index}/{frame.camera_id} has size "
                f"{frame.image_size}, expected paper target {target_size}"
            )
        channel = _camera_channel(frame.camera_id)
        image_name = _staged_image_name(frame, channel)
        if image_name in names:
            raise ValueError(f"duplicate staged COLMAP image name: {image_name}")
        names.add(image_name)
        _known_pose._validate_pinhole_intrinsics(frame)
        selected.append(
            StagedFrame(
                source_index=source_index,
                channel=channel,
                frame=frame,
                image_name=image_name,
                mask_name=f"{image_name}.png",
            )
        )
    return split, tuple(selected)


def stage_training_frames(
    staged_frames: Sequence[StagedFrame],
    actor_tracks: Sequence[ActorTrack],
    *,
    images_dir: Path,
    masks_dir: Path,
    actor_box_scale: float,
) -> tuple[StagedFrame, ...]:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Pillow is required for Waymo RGB staging") from error
    result: list[StagedFrame] = []
    for record in staged_frames:
        destination = images_dir / record.image_name
        mask_destination = masks_dir / record.mask_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        mask_destination.parent.mkdir(parents=True, exist_ok=True)
        height, width = record.frame.image_size
        with Image.open(record.frame.image_path) as image:
            if image.size != (width, height):
                raise ValueError(
                    f"decoded Waymo image {record.frame.image_path} has size "
                    f"{image.size}, expected {(width, height)}"
                )
        shutil.copy2(record.frame.image_path, destination)
        mask, actor_ids, dynamic_pixels = _known_pose.render_static_feature_mask(
            record.frame,
            actor_tracks,
            actor_box_scale=actor_box_scale,
        )
        mask.save(mask_destination, format="PNG")
        result.append(
            StagedFrame(
                source_index=record.source_index,
                channel=record.channel,
                frame=record.frame,
                image_name=record.image_name,
                mask_name=record.mask_name,
                visible_actor_ids=actor_ids,
                dynamic_pixel_count=dynamic_pixels,
            )
        )
    return tuple(result)


def build_colmap_commands(
    output_dir: Path, colmap_binary: str, *, use_gpu: bool = False
) -> dict[str, list[str]]:
    gpu_value = "1" if use_gpu else "0"
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    database_path = output_dir / "database.db"
    known_model = output_dir / "known_pose_model"
    triangulated_model = output_dir / "triangulated_model"
    text_model = output_dir / "triangulated_text"
    return {
        "feature_extractor": [
            colmap_binary,
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(images_dir),
            "--ImageReader.mask_path",
            str(masks_dir),
            "--ImageReader.camera_model",
            "PINHOLE",
            "--ImageReader.single_camera_per_folder",
            "1",
            "--SiftExtraction.use_gpu",
            gpu_value,
        ],
        "exhaustive_matcher": [
            colmap_binary,
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
            "--SiftMatching.use_gpu",
            gpu_value,
        ],
        "point_triangulator": [
            colmap_binary,
            "point_triangulator",
            "--database_path",
            str(database_path),
            "--image_path",
            str(images_dir),
            "--input_path",
            str(known_model),
            "--output_path",
            str(triangulated_model),
            "--Mapper.ba_refine_focal_length",
            "0",
            "--Mapper.ba_refine_principal_point",
            "0",
            "--Mapper.ba_refine_extra_params",
            "0",
            "--Mapper.fix_existing_images",
            "1",
            "--Mapper.filter_max_reproj_error",
            "4",
            "--Mapper.filter_min_tri_angle",
            "0.5",
            "--Mapper.tri_min_angle",
            "0.5",
            "--Mapper.tri_ignore_two_view_tracks",
            "1",
        ],
        "model_converter": [
            colmap_binary,
            "model_converter",
            "--input_path",
            str(triangulated_model),
            "--output_path",
            str(text_model),
            "--output_type",
            "TXT",
        ],
    }


def run_checked(command: Sequence[str]) -> None:
    _known_pose.run_checked(command)


def update_database_known_intrinsics(
    database_path: Path, staged_frames: Sequence[StagedFrame]
) -> tuple[Any, ...]:
    return _known_pose.update_database_known_intrinsics(database_path, staged_frames)


def write_known_pose_model(
    model_dir: Path,
    staged_frames: Sequence[StagedFrame],
    database_images: Sequence[Any],
) -> None:
    _known_pose.write_known_pose_model(model_dir, staged_frames, database_images)


def _frame_mapping(record: StagedFrame) -> dict[str, Any]:
    frame = record.frame
    height, width = frame.image_size
    capture_timestamp = frame.capture_timestamp
    assert capture_timestamp is not None
    return {
        "source_index": record.source_index,
        "frame_index": frame.frame_index,
        "camera_id": frame.camera_id,
        "camera_channel": record.channel,
        "observation_timestamp_ns": int(frame.timestamp.item()),
        "capture_timestamp_ns": int(capture_timestamp.item()),
        "original_image_path": str(frame.image_path.resolve()),
        "image_name": record.image_name,
        "mask_name": record.mask_name,
        "image_size": [height, width],
        "intrinsics": frame.intrinsics.detach().cpu().tolist(),
        "camera_to_world": frame.camera_to_world.detach().cpu().tolist(),
        "visible_dynamic_actor_ids": list(record.visible_actor_ids),
        "dynamic_pixel_count": record.dynamic_pixel_count,
        "dynamic_fraction": record.dynamic_pixel_count / float(height * width),
    }


def _mapping_payload(
    *,
    args: argparse.Namespace,
    cache_dir: Path,
    manifest: CanonicalDatasetManifest,
    split: Any,
    staged_frames: Sequence[StagedFrame],
    commands: dict[str, list[str]],
    status: str,
    world_center: torch.Tensor,
) -> dict[str, Any]:
    return {
        "schema_version": _MAPPING_SCHEMA_VERSION,
        "status": status,
        "dataset": "waymo_v2",
        "sequence": args.sequence,
        "waymo_root": str(args.waymo_root.resolve()),
        "parquet_dir": args.parquet_dir,
        "cache_dir": str(cache_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "camera_channels": list(args.cameras),
        "source_frame_range": {
            "start": args.start_frame,
            "end_inclusive": args.end_frame,
        },
        "paper_resolution": [args.target_height, args.target_width],
        "coordinate_frame": {
            "name": "waymo_world_centered",
            "centered": True,
            "centering_method": "full_context_mean_vehicle_translation",
            "world_center_m": world_center.detach().cpu().tolist(),
        },
        "split": {
            "type": "periodic",
            "every": args.split_every,
            "offset": args.split_offset,
            "start_position": args.split_start_position,
            "train_source_indices": list(split.train_source_indices),
            "eval_source_indices": list(split.eval_source_indices),
        },
        "actor_mask": {
            "source": "projected_dynamic_actor_cuboids",
            "tracker_source": (
                "castrack" if args.castrack_path is not None else "waymo_gt"
            ),
            "castrack_path": (
                str(args.castrack_path.resolve())
                if args.castrack_path is not None
                else None
            ),
            "static_value": 255,
            "dynamic_value": 0,
            "box_scale": args.actor_box_scale,
            "track_count": len(manifest.actor_tracks),
        },
        "known_pose_contract": {
            "camera_model": "PINHOLE",
            "camera_convention": "opencv",
            "world_frame": "waymo_world_centered",
            "pose_refinement": False,
            "intrinsics_refinement": False,
            "sift_gpu": bool(args.use_gpu),
        },
        "commands": commands,
        "frames": [_frame_mapping(record) for record in staged_frames],
        "summary": {
            "source_image_count": len(manifest),
            "train_image_count": len(staged_frames),
            "eval_image_count": len(split.eval_manifest),
            "train_capture_count": len(
                {record.frame.frame_index for record in staged_frames}
            ),
            "dynamic_mask_pixel_count": sum(
                record.dynamic_pixel_count for record in staged_frames
            ),
        },
    }


def _write_mapping(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _point_count(points3d_path: Path) -> int:
    return _known_pose._point_count(points3d_path)


def prepare_waymo_colmap(args: argparse.Namespace) -> dict[str, Any]:
    if args.end_frame is not None and args.end_frame < args.start_frame:
        raise ValueError("end_frame must be greater than or equal to start_frame")
    if args.split_offset >= args.split_every:
        raise ValueError("split_offset must satisfy 0 <= offset < split_every")
    output_dir = args.output_dir.resolve()
    cache_dir = (
        args.cache_dir.resolve()
        if args.cache_dir is not None
        else output_dir.with_name(f"{output_dir.name}_waymo_cache")
    )
    target_size = (args.target_height, args.target_width)
    if args.castrack_path is not None:
        args.castrack_path = args.castrack_path.resolve(strict=True)
    manifest = _load_waymo_v2_manifest(
        args.waymo_root,
        sequence=args.sequence,
        parquet_dir=args.parquet_dir,
        camera_channels=tuple(args.cameras),
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        target_size=target_size,
        cache_dir=cache_dir,
        castrack_path=args.castrack_path,
    )
    world_center = _load_waymo_world_center(
        args.waymo_root,
        sequence=args.sequence,
        parquet_dir=args.parquet_dir,
    )
    split, selected = select_training_frames(
        manifest,
        every=args.split_every,
        offset=args.split_offset,
        start_position=args.split_start_position,
        target_size=target_size,
    )
    commands = build_colmap_commands(
        output_dir, args.colmap_binary, use_gpu=args.use_gpu
    )
    payload = _mapping_payload(
        args=args,
        cache_dir=cache_dir,
        manifest=manifest,
        split=split,
        staged_frames=selected,
        commands=commands,
        status="dry_run" if args.dry_run else "planned",
        world_center=world_center,
    )
    if args.dry_run:
        payload["output_directory_nonempty"] = (
            output_dir.is_dir() and any(output_dir.iterdir())
        )
        return payload

    _known_pose._initialize_output_directory(output_dir)
    staged = stage_training_frames(
        selected,
        manifest.actor_tracks,
        images_dir=output_dir / "images",
        masks_dir=output_dir / "masks",
        actor_box_scale=args.actor_box_scale,
    )
    payload = _mapping_payload(
        args=args,
        cache_dir=cache_dir,
        manifest=manifest,
        split=split,
        staged_frames=staged,
        commands=commands,
        status="staged",
        world_center=world_center,
    )
    mapping_path = output_dir / "mapping.json"
    _write_mapping(mapping_path, payload)
    if args.skip_execution:
        return payload

    run_checked(commands["feature_extractor"])
    database_images = update_database_known_intrinsics(
        output_dir / "database.db", staged
    )
    write_known_pose_model(
        output_dir / "known_pose_model", staged, database_images
    )
    (output_dir / "triangulated_model").mkdir()
    run_checked(commands["exhaustive_matcher"])
    run_checked(commands["point_triangulator"])
    (output_dir / "triangulated_text").mkdir()
    run_checked(commands["model_converter"])

    database_by_name = {image.name: image for image in database_images}
    for frame_payload in payload["frames"]:
        image = database_by_name[str(frame_payload["image_name"])]
        frame_payload["database_image_id"] = image.image_id
        frame_payload["database_camera_id"] = image.camera_id
    points_path = output_dir / "triangulated_text" / "points3D.txt"
    point_count = _point_count(points_path)
    if point_count <= 0:
        payload["status"] = "failed_empty_points3D"
        payload["final_points3D_path"] = str(points_path.resolve())
        payload["summary"]["sfm_point_count"] = point_count
        _write_mapping(mapping_path, payload)
        raise RuntimeError(
            "known-pose COLMAP triangulation produced an empty points3D.txt"
        )
    payload["status"] = "complete"
    payload["final_points3D_path"] = str(points_path.resolve())
    payload["summary"]["sfm_point_count"] = point_count
    _write_mapping(mapping_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = prepare_waymo_colmap(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
