#!/usr/bin/env python3
"""Triangulate nuScenes background points from known dataset camera poses.

This follows the Street Gaussians/ArmGS initialization route:

* only images in the training split are given to COLMAP;
* projected dynamic-actor cuboids are blacked out in feature masks;
* feature extraction creates COLMAP's image and camera identifiers;
* nuScenes PINHOLE intrinsics and world-aligned camera poses replace COLMAP's
  provisional camera model before matching and point triangulation.

``--dry-run`` performs metadata/split validation without writing anything.
``--skip-execution`` stages images, masks, and mapping metadata without
requiring a COLMAP installation. A real run deliberately refuses to reuse a
non-empty output directory, because mixing databases or staged images from two
splits silently corrupts known-pose triangulation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import sqlite3
import struct
import subprocess
import sys
from typing import Any, Sequence

import torch
from torch import Tensor


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from armgs.data.nuscenes import (  # noqa: E402
    NUSCENES_CAMERA_CHANNELS,
    load_nuscenes_manifest,
    normalize_nuscenes_scene_name,
)
from armgs.data.schema import (  # noqa: E402
    ActorTrack,
    CanonicalDatasetManifest,
    CanonicalFrame,
)
from armgs.data.split import periodic_train_eval_split  # noqa: E402
from armgs.geometry import quaternion_slerp, quaternion_to_rotation_matrix  # noqa: E402


_PINHOLE_MODEL_ID = 1
_MAPPING_SCHEMA_VERSION = 1
_CUBOID_FACES: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 3, 2),
    (4, 5, 7, 6),
    (0, 1, 5, 4),
    (2, 3, 7, 6),
    (0, 2, 6, 4),
    (1, 3, 7, 5),
)


@dataclass(frozen=True)
class StagedFrame:
    """One training observation and its COLMAP-relative file names."""

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
        if self.channel not in NUSCENES_CAMERA_CHANNELS:
            raise ValueError(f"unknown nuScenes camera channel: {self.channel}")
        image_path = Path(self.image_name)
        mask_path = Path(self.mask_name)
        if image_path.is_absolute() or mask_path.is_absolute():
            raise ValueError("staged image and mask names must be relative")
        if image_path.parts[0] != self.channel:
            raise ValueError("staged image must be inside its camera folder")
        if self.mask_name != f"{self.image_name}.png":
            raise ValueError("COLMAP mask name must equal image name plus '.png'")
        if self.dynamic_pixel_count < 0:
            raise ValueError("dynamic_pixel_count must be non-negative")


@dataclass(frozen=True)
class DatabaseImage:
    image_id: int
    name: str
    camera_id: int


def parse_camera_channels(value: str) -> tuple[str, ...]:
    """Parse ``all`` or a unique comma-separated nuScenes camera list."""

    stripped = value.strip()
    if stripped.lower() == "all":
        return tuple(NUSCENES_CAMERA_CHANNELS)
    channels = tuple(component.strip().upper() for component in stripped.split(","))
    if not channels or any(not channel for channel in channels):
        raise argparse.ArgumentTypeError(
            "cameras must be 'all' or comma-separated channel names"
        )
    unknown = set(channels) - set(NUSCENES_CAMERA_CHANNELS)
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown nuScenes camera channels: " + ", ".join(sorted(unknown))
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
            "Prepare a StreetGS-style, known-pose COLMAP background model "
            "from the training split of one nuScenes scene."
        )
    )
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--scene", default="0061")
    parser.add_argument(
        "--cameras",
        type=parse_camera_channels,
        default=tuple(NUSCENES_CAMERA_CHANNELS),
        help="all or comma-separated channels (default: all six cameras)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split-every",
        type=_positive_int,
        default=8,
        help="periodic evaluation interval (default: 8)",
    )
    parser.add_argument(
        "--split-offset",
        type=_nonnegative_int,
        default=0,
        help="periodic evaluation offset (default: 0)",
    )
    parser.add_argument(
        "--split-start-position",
        type=_nonnegative_int,
        default=1,
        help="keep earlier captures in train before periodic holdout (default: 1)",
    )
    parser.add_argument(
        "--actor-box-scale",
        type=_positive_float,
        default=1.0,
        help="uniform multiplier for dynamic actor cuboids (default: 1.0)",
    )
    parser.add_argument("--colmap-binary", default="colmap")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the split/commands without writing or running COLMAP",
    )
    execution.add_argument(
        "--skip-execution",
        action="store_true",
        help=(
            "stage images, masks, and mapping JSON for inspection without "
            "invoking COLMAP; a real run requires a fresh empty output directory"
        ),
    )
    return parser.parse_args(argv)


def _camera_channel(camera_id: int) -> str:
    if camera_id < 0 or camera_id >= len(NUSCENES_CAMERA_CHANNELS):
        raise ValueError(f"nuScenes manifest has an unknown camera_id: {camera_id}")
    return NUSCENES_CAMERA_CHANNELS[camera_id]


def _validate_pinhole_intrinsics(frame: CanonicalFrame) -> tuple[float, float, float, float]:
    if frame.camera_convention != "opencv":
        raise ValueError("known-pose COLMAP preparation requires OpenCV cameras")
    intrinsics = frame.intrinsics.detach().to(device="cpu", dtype=torch.float64)
    expected_bottom = intrinsics.new_tensor([0.0, 0.0, 1.0])
    if not torch.allclose(intrinsics[2], expected_bottom, atol=1.0e-9, rtol=0.0):
        raise ValueError("PINHOLE intrinsics must end with [0,0,1]")
    if abs(float(intrinsics[0, 1])) > 1.0e-9 or abs(float(intrinsics[1, 0])) > 1.0e-9:
        raise ValueError("COLMAP PINHOLE cannot represent skewed intrinsics")
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    if fx <= 0.0 or fy <= 0.0 or not all(
        math.isfinite(value) for value in (fx, fy, cx, cy)
    ):
        raise ValueError("PINHOLE parameters must be finite with positive focal lengths")
    return fx, fy, cx, cy


def _actor_pose_at(
    track: ActorTrack, timestamp_ns: int
) -> tuple[Tensor, Tensor] | None:
    start, end = track.lifecycle_timestamps
    if timestamp_ns < int(start.item()) or timestamp_ns > int(end.item()):
        return None
    samples = track.samples
    if len(samples) == 1:
        return samples[0].quaternion_wxyz, samples[0].translation
    times = torch.tensor(
        [int(sample.timestamp.item()) for sample in samples], dtype=torch.float64
    )
    query = torch.tensor(float(timestamp_ns), dtype=torch.float64)
    upper = int(torch.searchsorted(times, query, right=False).item())
    upper = max(1, min(upper, len(samples) - 1))
    lower = upper - 1
    denominator = times[upper] - times[lower]
    weight64 = ((query - times[lower]) / denominator).clamp(0.0, 1.0)
    reference = samples[lower].translation
    weight = weight64.to(reference)
    translation = torch.lerp(
        samples[lower].translation, samples[upper].translation, weight
    )
    quaternion = quaternion_slerp(
        samples[lower].quaternion_wxyz,
        samples[upper].quaternion_wxyz,
        weight,
    )
    return quaternion, translation


def _cuboid_corners(dimensions_lwh: Tensor, scale: float) -> Tensor:
    half = dimensions_lwh.detach().to(device="cpu", dtype=torch.float64) * (0.5 * scale)
    length, width, height = half.unbind()
    return torch.stack(
        tuple(
            torch.stack((sx * length, sy * width, sz * height))
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        )
    )


def _clip_polygon_near(points: list[Tensor], near: float) -> list[Tensor]:
    if not points:
        return []
    result: list[Tensor] = []
    previous = points[-1]
    previous_inside = float(previous[2]) >= near
    for current in points:
        current_inside = float(current[2]) >= near
        if current_inside != previous_inside:
            weight = (near - float(previous[2])) / float(current[2] - previous[2])
            result.append(previous + (current - previous) * weight)
        if current_inside:
            result.append(current)
        previous = current
        previous_inside = current_inside
    return result


def _clip_polygon_axis(
    points: list[tuple[float, float]],
    *,
    axis: int,
    bound: float,
    keep_greater: bool,
) -> list[tuple[float, float]]:
    if not points:
        return []

    def inside(point: tuple[float, float]) -> bool:
        return point[axis] >= bound if keep_greater else point[axis] <= bound

    output: list[tuple[float, float]] = []
    previous = points[-1]
    previous_inside = inside(previous)
    for current in points:
        current_inside = inside(current)
        if current_inside != previous_inside:
            delta = current[axis] - previous[axis]
            if abs(delta) > 1.0e-12:
                weight = (bound - previous[axis]) / delta
                intersection = (
                    previous[0] + weight * (current[0] - previous[0]),
                    previous[1] + weight * (current[1] - previous[1]),
                )
                output.append(intersection)
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
    return output


def _clip_polygon_image(
    points: list[tuple[float, float]], height: int, width: int
) -> list[tuple[float, float]]:
    result = points
    result = _clip_polygon_axis(result, axis=0, bound=0.0, keep_greater=True)
    result = _clip_polygon_axis(result, axis=0, bound=float(width - 1), keep_greater=False)
    result = _clip_polygon_axis(result, axis=1, bound=0.0, keep_greater=True)
    result = _clip_polygon_axis(result, axis=1, bound=float(height - 1), keep_greater=False)
    return result


def _draw_actor_cuboid(
    draw: Any,
    frame: CanonicalFrame,
    track: ActorTrack,
    *,
    actor_box_scale: float,
    near: float = 1.0e-3,
) -> bool:
    pose = _actor_pose_at(track, int(frame.timestamp.item()))
    if pose is None:
        return False
    quaternion, translation = pose
    corners_actor = _cuboid_corners(track.dimensions_lwh, actor_box_scale)
    rotation = quaternion_to_rotation_matrix(
        quaternion.detach().to(device="cpu", dtype=torch.float64)
    )
    corners_world = corners_actor @ rotation.T + translation.detach().to(
        device="cpu", dtype=torch.float64
    )
    world_to_camera = torch.linalg.inv(
        frame.camera_to_world.detach().to(device="cpu", dtype=torch.float64)
    )
    corners_camera = corners_world @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    intrinsics = frame.intrinsics.detach().to(device="cpu", dtype=torch.float64)
    height, width = frame.image_size
    drew_polygon = False
    for face in _CUBOID_FACES:
        camera_polygon = _clip_polygon_near(
            [corners_camera[index] for index in face], near
        )
        if len(camera_polygon) < 3:
            continue
        image_polygon: list[tuple[float, float]] = []
        for point in camera_polygon:
            projected = intrinsics @ point
            image_polygon.append(
                (float(projected[0] / projected[2]), float(projected[1] / projected[2]))
            )
        image_polygon = _clip_polygon_image(image_polygon, height, width)
        if len(image_polygon) < 3:
            continue
        draw.polygon(image_polygon, fill=0)
        drew_polygon = True
    return drew_polygon


def render_static_feature_mask(
    frame: CanonicalFrame,
    actor_tracks: Sequence[ActorTrack],
    *,
    actor_box_scale: float = 1.0,
) -> tuple[Any, tuple[int, ...], int]:
    """Return a COLMAP mask: static pixels 255, dynamic cuboids 0."""

    if not math.isfinite(actor_box_scale) or actor_box_scale <= 0.0:
        raise ValueError("actor_box_scale must be finite and positive")
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:  # pragma: no cover - dependency error is explicit.
        raise RuntimeError(
            "Pillow is required; install the ArmGS 'data' optional dependencies"
        ) from error
    height, width = frame.image_size
    mask = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(mask)
    visible: list[int] = []
    for track in actor_tracks:
        if _draw_actor_cuboid(
            draw, frame, track, actor_box_scale=actor_box_scale
        ):
            visible.append(track.actor_id)
    histogram = mask.histogram()
    dynamic_pixels = int(histogram[0])
    return mask, tuple(visible), dynamic_pixels


def _staged_image_name(frame: CanonicalFrame, channel: str) -> str:
    suffix = frame.image_path.suffix.lower()
    if not suffix:
        raise ValueError(f"nuScenes image has no file extension: {frame.image_path}")
    basename = frame.image_path.name
    if Path(basename).name != basename:
        raise ValueError(f"unsafe nuScenes image basename: {basename}")
    return (Path(channel) / basename).as_posix()


def select_training_frames(
    manifest: CanonicalDatasetManifest,
    *,
    every: int,
    offset: int,
    start_position: int,
) -> tuple[Any, tuple[StagedFrame, ...]]:
    split = periodic_train_eval_split(
        manifest,
        every=every,
        offset=offset,
        start_position=start_position,
    )
    staged: list[StagedFrame] = []
    seen_names: set[str] = set()
    for source_index in split.train_source_indices:
        frame = manifest.frames[source_index]
        channel = _camera_channel(frame.camera_id)
        image_name = _staged_image_name(frame, channel)
        if image_name in seen_names:
            raise ValueError(f"duplicate staged COLMAP image name: {image_name}")
        seen_names.add(image_name)
        _validate_pinhole_intrinsics(frame)
        staged.append(
            StagedFrame(
                source_index=source_index,
                channel=channel,
                frame=frame,
                image_name=image_name,
                mask_name=f"{image_name}.png",
            )
        )
    return split, tuple(staged)


def _initialize_output_directory(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"COLMAP output path is not a directory: {output_dir}")
        try:
            next(output_dir.iterdir())
        except StopIteration:
            pass
        else:
            raise FileExistsError(
                "refusing to overwrite non-empty COLMAP output directory: "
                f"{output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def stage_training_frames(
    staged_frames: Sequence[StagedFrame],
    actor_tracks: Sequence[ActorTrack],
    *,
    images_dir: Path,
    masks_dir: Path,
    actor_box_scale: float,
) -> tuple[StagedFrame, ...]:
    """Copy training images and generate COLMAP-compatible feature masks."""

    result: list[StagedFrame] = []
    for record in staged_frames:
        destination = images_dir / record.image_name
        mask_destination = masks_dir / record.mask_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        mask_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.frame.image_path, destination)
        mask, actor_ids, dynamic_pixels = render_static_feature_mask(
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


def _same_camera_calibration(left: CanonicalFrame, right: CanonicalFrame) -> bool:
    return left.image_size == right.image_size and torch.allclose(
        left.intrinsics.detach().to(device="cpu", dtype=torch.float64),
        right.intrinsics.detach().to(device="cpu", dtype=torch.float64),
        atol=1.0e-9,
        rtol=1.0e-9,
    )


def update_database_known_intrinsics(
    database_path: Path, staged_frames: Sequence[StagedFrame]
) -> tuple[DatabaseImage, ...]:
    """Read COLMAP IDs and replace every provisional camera with PINHOLE K."""

    if not database_path.is_file():
        raise FileNotFoundError(f"COLMAP database was not created: {database_path}")
    expected = {record.image_name: record for record in staged_frames}
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT image_id, name, camera_id FROM images ORDER BY image_id"
        ).fetchall()
        images = tuple(
            DatabaseImage(image_id=int(row[0]), name=str(row[1]), camera_id=int(row[2]))
            for row in rows
        )
        actual_names = {image.name for image in images}
        if actual_names != set(expected):
            missing = sorted(set(expected) - actual_names)
            extra = sorted(actual_names - set(expected))
            raise ValueError(
                f"COLMAP database image names differ from staged images; "
                f"missing={missing}, extra={extra}"
            )
        if len(images) != len(actual_names):
            raise ValueError("COLMAP database contains duplicate image names")

        records_by_camera: dict[int, list[StagedFrame]] = {}
        channel_to_camera: dict[str, int] = {}
        for image in images:
            record = expected[image.name]
            previous = channel_to_camera.setdefault(record.channel, image.camera_id)
            if previous != image.camera_id:
                raise ValueError(
                    f"ImageReader.single_camera_per_folder did not share camera for "
                    f"{record.channel}"
                )
            records_by_camera.setdefault(image.camera_id, []).append(record)
        if len(channel_to_camera) != len(set(channel_to_camera.values())):
            raise ValueError("different camera folders unexpectedly share a camera_id")

        database_camera_ids = {
            int(row[0])
            for row in connection.execute("SELECT camera_id FROM cameras").fetchall()
        }
        if database_camera_ids != set(records_by_camera):
            raise ValueError(
                "COLMAP database camera rows do not match image camera identifiers"
            )
        for camera_id, records in records_by_camera.items():
            reference = records[0].frame
            if any(
                not _same_camera_calibration(reference, record.frame)
                for record in records[1:]
            ):
                raise ValueError(
                    f"nuScenes camera folder {records[0].channel} has varying intrinsics"
                )
            fx, fy, cx, cy = _validate_pinhole_intrinsics(reference)
            height, width = reference.image_size
            params = struct.pack("<4d", fx, fy, cx, cy)
            cursor = connection.execute(
                "UPDATE cameras SET model = ?, width = ?, height = ?, params = ?, "
                "prior_focal_length = 1 WHERE camera_id = ?",
                (_PINHOLE_MODEL_ID, width, height, params, camera_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"failed to update COLMAP camera_id {camera_id}")
        connection.commit()
    return images


def _rotation_matrix_to_qvec(rotation: Tensor) -> tuple[float, float, float, float]:
    """Convert a proper 3x3 rotation matrix to deterministic COLMAP wxyz."""

    matrix = rotation.detach().to(device="cpu", dtype=torch.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape [3,3]")
    values = matrix.tolist()
    trace = values[0][0] + values[1][1] + values[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (values[2][1] - values[1][2]) / scale
        qy = (values[0][2] - values[2][0]) / scale
        qz = (values[1][0] - values[0][1]) / scale
    elif values[0][0] > values[1][1] and values[0][0] > values[2][2]:
        scale = math.sqrt(1.0 + values[0][0] - values[1][1] - values[2][2]) * 2.0
        qw = (values[2][1] - values[1][2]) / scale
        qx = 0.25 * scale
        qy = (values[0][1] + values[1][0]) / scale
        qz = (values[0][2] + values[2][0]) / scale
    elif values[1][1] > values[2][2]:
        scale = math.sqrt(1.0 + values[1][1] - values[0][0] - values[2][2]) * 2.0
        qw = (values[0][2] - values[2][0]) / scale
        qx = (values[0][1] + values[1][0]) / scale
        qy = 0.25 * scale
        qz = (values[1][2] + values[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + values[2][2] - values[0][0] - values[1][1]) * 2.0
        qw = (values[1][0] - values[0][1]) / scale
        qx = (values[0][2] + values[2][0]) / scale
        qy = (values[1][2] + values[2][1]) / scale
        qz = 0.25 * scale
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qvec = (qw / norm, qx / norm, qy / norm, qz / norm)
    if qvec[0] < 0.0:
        qvec = tuple(-value for value in qvec)  # type: ignore[assignment]
    return qvec


def write_known_pose_model(
    model_dir: Path,
    staged_frames: Sequence[StagedFrame],
    database_images: Sequence[DatabaseImage],
) -> None:
    """Write an empty COLMAP text model in the canonical nuScenes world frame."""

    records = {record.image_name: record for record in staged_frames}
    if {image.name for image in database_images} != set(records):
        raise ValueError("database images must match staged frames exactly")
    model_dir.mkdir(parents=True, exist_ok=False)

    camera_records: dict[int, StagedFrame] = {}
    for image in database_images:
        record = records[image.name]
        previous = camera_records.setdefault(image.camera_id, record)
        if not _same_camera_calibration(previous.frame, record.frame):
            raise ValueError(f"camera_id {image.camera_id} has inconsistent calibration")

    camera_lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(camera_records)}",
    ]
    for camera_id in sorted(camera_records):
        frame = camera_records[camera_id].frame
        height, width = frame.image_size
        fx, fy, cx, cy = _validate_pinhole_intrinsics(frame)
        camera_lines.append(
            f"{camera_id} PINHOLE {width} {height} "
            f"{fx:.17g} {fy:.17g} {cx:.17g} {cy:.17g}"
        )
    (model_dir / "cameras.txt").write_text(
        "\n".join(camera_lines) + "\n", encoding="utf-8"
    )

    image_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(database_images)}, mean observations per image: 0",
    ]
    for image in sorted(database_images, key=lambda item: item.image_id):
        frame = records[image.name].frame
        world_to_camera = torch.linalg.inv(
            frame.camera_to_world.detach().to(device="cpu", dtype=torch.float64)
        )
        qw, qx, qy, qz = _rotation_matrix_to_qvec(world_to_camera[:3, :3])
        tx, ty, tz = (float(value) for value in world_to_camera[:3, 3])
        image_lines.append(
            f"{image.image_id} {qw:.17g} {qx:.17g} {qy:.17g} {qz:.17g} "
            f"{tx:.17g} {ty:.17g} {tz:.17g} {image.camera_id} {image.name}"
        )
        image_lines.append("")
    # COLMAP requires the observations line (including the final one) to end
    # with a newline, even though every image initially has zero observations.
    (model_dir / "images.txt").write_text(
        "\n".join(image_lines) + "\n", encoding="utf-8"
    )
    (model_dir / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"
        "# Number of points: 0, mean track length: 0\n",
        encoding="utf-8",
    )


def build_colmap_commands(output_dir: Path, colmap_binary: str) -> dict[str, list[str]]:
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
        ],
        "exhaustive_matcher": [
            colmap_binary,
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
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
            # point_triangulator keeps registered image poses fixed. These
            # options additionally lock every representable intrinsic value.
            "--Mapper.ba_refine_focal_length",
            "0",
            "--Mapper.ba_refine_principal_point",
            "0",
            "--Mapper.ba_refine_extra_params",
            "0",
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
    if not command or any(not isinstance(argument, str) for argument in command):
        raise ValueError("COLMAP command must be a non-empty string argument array")
    subprocess.run(list(command), check=True)


def _frame_mapping(record: StagedFrame) -> dict[str, Any]:
    frame = record.frame
    height, width = frame.image_size
    return {
        "source_index": record.source_index,
        "frame_index": frame.frame_index,
        "camera_id": frame.camera_id,
        "camera_channel": record.channel,
        "observation_timestamp_ns": int(frame.timestamp.item()),
        "capture_timestamp_ns": int(frame.capture_timestamp.item()),
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
    manifest: CanonicalDatasetManifest,
    split: Any,
    staged_frames: Sequence[StagedFrame],
    commands: dict[str, list[str]],
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": _MAPPING_SCHEMA_VERSION,
        "status": status,
        "dataset": "nuscenes",
        "scene": normalize_nuscenes_scene_name(args.scene),
        "version": args.version,
        "nuscenes_root": str(args.nuscenes_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "camera_channels": list(args.cameras),
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
            "static_value": 255,
            "dynamic_value": 0,
            "box_scale": args.actor_box_scale,
            "track_count": len(manifest.actor_tracks),
        },
        "known_pose_contract": {
            "camera_model": "PINHOLE",
            "camera_convention": "opencv",
            "world_frame": "nuscenes_global",
            "pose_refinement": False,
            "intrinsics_refinement": False,
        },
        "commands": commands,
        "frames": [_frame_mapping(record) for record in staged_frames],
        "summary": {
            "source_frame_count": len(manifest),
            "train_frame_count": len(staged_frames),
            "eval_frame_count": len(split.eval_manifest),
            "train_capture_count": len(
                {record.frame.frame_index for record in staged_frames}
            ),
            "dynamic_mask_pixel_count": sum(
                record.dynamic_pixel_count for record in staged_frames
            ),
        },
    }


def _write_mapping(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _point_count(points3d_path: Path) -> int:
    if not points3d_path.is_file():
        raise FileNotFoundError(
            f"COLMAP TXT conversion did not create points3D.txt: {points3d_path}"
        )
    with points3d_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip() and not line.startswith("#"))


def _require_nonempty_points3d(points3d_path: Path) -> int:
    count = _point_count(points3d_path)
    if count == 0:
        raise RuntimeError(
            "COLMAP triangulation produced zero points; refusing to mark the "
            "initialization complete"
        )
    return count


def prepare_nuscenes_colmap(args: argparse.Namespace) -> dict[str, Any]:
    if args.split_offset >= args.split_every:
        raise ValueError("split_offset must satisfy 0 <= offset < split_every")
    manifest = load_nuscenes_manifest(
        args.nuscenes_root,
        scene=args.scene,
        version=args.version,
        camera_channels=tuple(args.cameras),
        require_lidar=False,
        include_stationary_actors=False,
    )
    split, selected = select_training_frames(
        manifest,
        every=args.split_every,
        offset=args.split_offset,
        start_position=args.split_start_position,
    )
    output_dir = args.output_dir.resolve()
    commands = build_colmap_commands(output_dir, args.colmap_binary)
    payload = _mapping_payload(
        args=args,
        manifest=manifest,
        split=split,
        staged_frames=selected,
        commands=commands,
        status="dry_run" if args.dry_run else "planned",
    )
    if args.dry_run:
        payload["output_directory_nonempty"] = (
            output_dir.is_dir() and any(output_dir.iterdir())
        )
        return payload

    _initialize_output_directory(output_dir)
    staged = stage_training_frames(
        selected,
        manifest.actor_tracks,
        images_dir=output_dir / "images",
        masks_dir=output_dir / "masks",
        actor_box_scale=args.actor_box_scale,
    )
    payload = _mapping_payload(
        args=args,
        manifest=manifest,
        split=split,
        staged_frames=staged,
        commands=commands,
        status="staged",
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

    db_by_name = {image.name: image for image in database_images}
    for frame_payload in payload["frames"]:
        image = db_by_name[str(frame_payload["image_name"])]
        frame_payload["database_image_id"] = image.image_id
        frame_payload["database_camera_id"] = image.camera_id
    points_path = output_dir / "triangulated_text" / "points3D.txt"
    point_count = _require_nonempty_points3d(points_path)
    payload["status"] = "complete"
    payload["final_points3D_path"] = str(points_path.resolve())
    payload["summary"]["sfm_point_count"] = point_count
    _write_mapping(mapping_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = prepare_nuscenes_colmap(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
