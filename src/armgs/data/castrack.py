"""StreetGS-compatible CAStrack conversion for Waymo scenes.

The official StreetGS Waymo converter stores one JSON record per source frame
with detector boxes in the Waymo vehicle frame.  This module deliberately
does not depend on the Waymo SDK: callers pass the already-loaded capture
timestamps, vehicle poses, and canonical FRONT camera frames, and receive the
same :class:`~armgs.data.schema.ActorTrack` records used by the rest of ArmGS.

Both of the common JSON layouts are accepted:

* the original, multi-scene CAStrack result, whose top-level keys are
  ``segment-<context>_with_camera_labels``; and
* an extracted scene file containing either that single keyed entry or the
  source-frame mapping directly.

Use :func:`extract_castrack_scene_json` once when starting from the large
validation result.  Subsequent scene loads then avoid parsing the full file.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch import Tensor

from .schema import ActorTrack, ActorTrackSample, CanonicalFrame


CASTRACK_ACTOR_SOURCE = "castrack"

_CLASS_NAMES: Mapping[str, str] = {
    "Vehicle": "vehicle",
    "Pedestrian": "pedestrian",
    "Cyclist": "cyclist",
}
_MICROSECONDS_TO_NANOSECONDS = 1_000
_LIFECYCLE_PADDING_NS = 100_000_000
_CORNER_SIGNS = torch.tensor(
    (
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
    ),
    dtype=torch.float64,
)


@dataclass(frozen=True)
class _Detection:
    raw_object_id: int
    class_name: str
    source_frame_index: int
    timestamp_micros: int
    frame_index: int
    dimensions_lwh: Tensor
    world_from_actor: Tensor
    first_seen_order: int


def _scene_key(sequence: str) -> str:
    if not isinstance(sequence, str) or not sequence.strip():
        raise ValueError("sequence must be a non-empty string")
    value = sequence.strip()
    if value.startswith("segment-"):
        return (
            value
            if value.endswith("_with_camera_labels")
            else f"{value}_with_camera_labels"
        )
    return f"segment-{value}_with_camera_labels"


def _read_json_object(path: str | Path) -> tuple[Path, Mapping[str, Any]]:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"CAStrack path is not a file: {resolved}")
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid CAStrack JSON: {resolved}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("CAStrack JSON root must be an object")
    return resolved, payload


def _looks_like_scene_frames(payload: Mapping[str, Any]) -> bool:
    if not payload:
        return False
    for frame_id, record in payload.items():
        try:
            source_frame_index = int(frame_id)
        except (TypeError, ValueError):
            return False
        if source_frame_index < 0 or not isinstance(record, Mapping):
            return False
        if not {"obj_ids", "name", "boxes_lidar"}.issubset(record):
            return False
    return True


def _scene_frames(
    payload: Mapping[str, Any], *, sequence: str
) -> Mapping[str, Any]:
    key = _scene_key(sequence)
    if key in payload:
        scene = payload[key]
        if not isinstance(scene, Mapping) or not _looks_like_scene_frames(scene):
            raise ValueError(f"CAStrack scene {key!r} is malformed or empty")
        return scene
    if _looks_like_scene_frames(payload):
        return payload
    raise KeyError(f"CAStrack JSON does not contain scene {key!r}")


def extract_castrack_scene_json(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    sequence: str,
    overwrite: bool = False,
) -> Path:
    """Extract one scene from a full CAStrack JSON into a compact keyed file.

    This intentionally performs the expensive full-file parse only once.  The
    result retains the canonical scene key, so loading it with a mismatched
    ``sequence`` fails instead of silently using another scene.
    """

    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")
    _, payload = _read_json_object(source_path)
    key = _scene_key(sequence)
    if key not in payload:
        raise KeyError(f"CAStrack JSON does not contain scene {key!r}")
    scene = payload[key]
    if not isinstance(scene, Mapping) or not _looks_like_scene_frames(scene):
        raise ValueError(f"CAStrack scene {key!r} is malformed or empty")

    destination = Path(destination_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"CAStrack extraction already exists: {destination}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump({key: scene}, handle, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists() and not overwrite:
            raise FileExistsError(f"CAStrack extraction already exists: {destination}")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def _non_negative_finite_float(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _validate_world_from_vehicle(value: Tensor, *, timestamp_micros: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(
            f"vehicle transform at {timestamp_micros} must be a torch.Tensor"
        )
    transform = value.detach().to(device="cpu", dtype=torch.float64)
    if transform.shape != (4, 4) or not torch.isfinite(transform).all():
        raise ValueError(
            f"vehicle transform at {timestamp_micros} must be finite [4,4]"
        )
    expected_last_row = transform.new_tensor((0.0, 0.0, 0.0, 1.0))
    if not torch.allclose(transform[3], expected_last_row, atol=1.0e-5, rtol=1.0e-5):
        raise ValueError(
            f"vehicle transform at {timestamp_micros} is not homogeneous"
        )
    rotation = transform[:3, :3]
    identity = torch.eye(3, dtype=transform.dtype)
    if not torch.allclose(rotation.T @ rotation, identity, atol=1.0e-4, rtol=1.0e-4):
        raise ValueError(
            f"vehicle transform at {timestamp_micros} rotation is not orthonormal"
        )
    if not torch.allclose(torch.linalg.det(rotation), transform.new_tensor(1.0), atol=1.0e-4):
        raise ValueError(
            f"vehicle transform at {timestamp_micros} rotation must have determinant +1"
        )
    return transform


def _rotation_matrix_to_quaternion_wxyz(rotation: Tensor) -> Tensor:
    m00, m01, m02 = rotation[0]
    m10, m11, m12 = rotation[1]
    m20, m21, m22 = rotation[2]
    one = rotation.new_tensor(1.0)
    quaternion = torch.stack(
        (
            0.5 * torch.sqrt(torch.clamp(one + m00 + m11 + m22, min=0.0)),
            torch.copysign(
                0.5 * torch.sqrt(torch.clamp(one + m00 - m11 - m22, min=0.0)),
                m21 - m12,
            ),
            torch.copysign(
                0.5 * torch.sqrt(torch.clamp(one - m00 + m11 - m22, min=0.0)),
                m02 - m20,
            ),
            torch.copysign(
                0.5 * torch.sqrt(torch.clamp(one - m00 - m11 + m22, min=0.0)),
                m10 - m01,
            ),
        )
    )
    return quaternion / torch.linalg.vector_norm(quaternion).clamp_min(1.0e-12)


def _box_to_world(
    box: Sequence[Any],
    *,
    world_from_vehicle: Tensor,
    source_frame_index: int,
    raw_object_id: int,
) -> tuple[Tensor, Tensor]:
    if isinstance(box, (str, bytes)) or not isinstance(box, Sequence) or len(box) != 7:
        raise ValueError(
            "CAStrack boxes_lidar rows must contain "
            "[cx,cy,cz,length,width,height,heading]"
        )
    try:
        values = torch.tensor([float(item) for item in box], dtype=torch.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"non-numeric CAStrack box for object {raw_object_id} at frame "
            f"{source_frame_index}"
        ) from error
    if not torch.isfinite(values).all() or torch.any(values[3:6] <= 0.0):
        raise ValueError(
            f"invalid CAStrack box for object {raw_object_id} at frame "
            f"{source_frame_index}"
        )
    center = values[:3]
    dimensions_lwh = values[3:6]
    heading = float(values[6].item())
    cosine = math.cos(heading)
    sine = math.sin(heading)
    vehicle_from_actor = torch.eye(4, dtype=torch.float64)
    vehicle_from_actor[:3, :3] = vehicle_from_actor.new_tensor(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    )
    vehicle_from_actor[:3, 3] = center
    return dimensions_lwh, world_from_vehicle @ vehicle_from_actor


def _has_front_corner_visibility(
    frame: CanonicalFrame,
    *,
    dimensions_lwh: Tensor,
    world_from_actor: Tensor,
) -> bool:
    dimensions = dimensions_lwh.to(dtype=torch.float64, device="cpu")
    transform = world_from_actor.to(dtype=torch.float64, device="cpu")
    local_corners = _CORNER_SIGNS * (dimensions / 2.0)
    world_corners = local_corners @ transform[:3, :3].T + transform[:3, 3]

    camera_to_world = frame.camera_to_world.detach().to(
        device="cpu", dtype=torch.float64
    )
    camera_corners = (
        world_corners - camera_to_world[:3, 3]
    ) @ camera_to_world[:3, :3]
    if frame.camera_convention == "opengl":
        camera_corners[:, 1:] = -camera_corners[:, 1:]
    elif frame.camera_convention != "opencv":  # CanonicalFrame validates this.
        raise ValueError("camera_convention must be 'opencv' or 'opengl'")

    intrinsics = frame.intrinsics.detach().to(device="cpu", dtype=torch.float64)
    homogeneous = camera_corners @ intrinsics.T
    depth = homogeneous[:, 2]
    finite = torch.isfinite(homogeneous).all(dim=-1)
    positive_depth = depth > torch.finfo(torch.float64).eps
    projectable = finite & positive_depth
    if not projectable.any():
        return False
    pixels = homogeneous[projectable, :2] / depth[projectable, None]
    x, y = pixels.unbind(dim=-1)
    height, width = frame.image_size
    inside = (
        torch.isfinite(pixels).all(dim=-1)
        & (x >= 0.0)
        & (x < width)
        & (y >= 0.0)
        & (y < height)
    )
    return bool(inside.any())


def _aligned_inputs(
    *,
    source_frame_indices: Sequence[int],
    selected_timestamps_micros: Sequence[int],
    relative_indices: Mapping[int, int],
    vehicle_transforms: Mapping[int, Tensor],
    front_frames: Sequence[CanonicalFrame],
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    dict[int, int],
    dict[int, Tensor],
    dict[int, CanonicalFrame],
]:
    if isinstance(source_frame_indices, (str, bytes)) or not isinstance(
        source_frame_indices, Sequence
    ):
        raise TypeError("source_frame_indices must be a sequence of integers")
    if isinstance(selected_timestamps_micros, (str, bytes)) or not isinstance(
        selected_timestamps_micros, Sequence
    ):
        raise TypeError("selected_timestamps_micros must be a sequence of integers")
    sources = tuple(source_frame_indices)
    timestamps = tuple(selected_timestamps_micros)
    if not sources or len(sources) != len(timestamps):
        raise ValueError(
            "source_frame_indices and selected_timestamps_micros must be "
            "non-empty and have equal length"
        )
    for value, name in (
        *((value, "source frame index") for value in sources),
        *((value, "Waymo timestamp") for value in timestamps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} values must be non-negative integers")
    if any(right <= left for left, right in zip(sources, sources[1:])):
        raise ValueError("source_frame_indices must be strictly increasing")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("selected_timestamps_micros must be strictly increasing")

    timestamp_to_relative: dict[int, int] = {}
    transforms: dict[int, Tensor] = {}
    frames_by_index: dict[int, CanonicalFrame] = {}
    for timestamp in timestamps:
        if timestamp not in relative_indices:
            raise ValueError(f"relative_indices is missing timestamp {timestamp}")
        relative_index = relative_indices[timestamp]
        if (
            isinstance(relative_index, bool)
            or not isinstance(relative_index, int)
            or relative_index < 0
        ):
            raise ValueError("relative frame indices must be non-negative integers")
        if relative_index in timestamp_to_relative.values():
            raise ValueError("relative frame indices must be unique")
        timestamp_to_relative[timestamp] = relative_index
        if timestamp not in vehicle_transforms:
            raise ValueError(f"vehicle_transforms is missing timestamp {timestamp}")
        transforms[timestamp] = _validate_world_from_vehicle(
            vehicle_transforms[timestamp], timestamp_micros=timestamp
        )

    if isinstance(front_frames, (str, bytes)) or not isinstance(front_frames, Sequence):
        raise TypeError("front_frames must be a sequence of CanonicalFrame objects")
    for frame in front_frames:
        if not isinstance(frame, CanonicalFrame):
            raise TypeError("front_frames must contain only CanonicalFrame objects")
        if frame.camera_id != 0:
            raise ValueError("front_frames must contain only Waymo FRONT camera_id 0")
        if frame.frame_index in frames_by_index:
            raise ValueError(f"duplicate FRONT frame_index {frame.frame_index}")
        frames_by_index[frame.frame_index] = frame

    expected_relative_indices = set(timestamp_to_relative.values())
    if set(frames_by_index) != expected_relative_indices:
        missing = sorted(expected_relative_indices - set(frames_by_index))
        extra = sorted(set(frames_by_index) - expected_relative_indices)
        raise ValueError(
            f"front_frames do not align with selected captures; missing={missing}, "
            f"extra={extra}"
        )
    for timestamp, relative_index in timestamp_to_relative.items():
        capture = frames_by_index[relative_index].capture_timestamp
        assert capture is not None  # CanonicalFrame normalizes this field.
        expected_timestamp_ns = timestamp * _MICROSECONDS_TO_NANOSECONDS
        if int(capture.item()) != expected_timestamp_ns:
            raise ValueError(
                f"FRONT frame_index {relative_index} capture timestamp does not "
                f"match Waymo timestamp {timestamp}"
            )
    return sources, timestamps, timestamp_to_relative, transforms, frames_by_index


def load_castrack_actor_tracks(
    path: str | Path,
    *,
    sequence: str,
    source_frame_indices: Sequence[int],
    selected_timestamps_micros: Sequence[int],
    relative_indices: Mapping[int, int],
    vehicle_transforms: Mapping[int, Tensor],
    front_frames: Sequence[CanonicalFrame],
    filter_static_actors: bool = True,
    static_std_threshold: float = 0.5,
    static_displacement_threshold: float = 2.0,
    lifecycle_padding_ns: int = _LIFECYCLE_PADDING_NS,
) -> tuple[ActorTrack, ...]:
    """Convert one selected Waymo CAStrack range into canonical actor tracks.

    ``source_frame_indices`` and ``selected_timestamps_micros`` are parallel,
    ordered sequences.  CAStrack boxes are interpreted in the vehicle frame;
    each supplied ``vehicle_transforms[timestamp]`` is used as
    ``world_from_vehicle`` without changing its origin, so pre-centered Waymo
    transforms remain aligned with centered canonical camera poses.

    StreetGS semantics retained here are:

    * only Vehicle, Pedestrian, and Cyclist detections are actors;
    * a detection is retained only if at least one of its eight cuboid corners
      projects inside the selected FRONT image at positive depth;
    * dimensions are the per-axis maximum over retained track samples; and
    * a track is dynamic when any world-position standard deviation exceeds
      ``static_std_threshold`` or endpoint displacement exceeds
      ``static_displacement_threshold`` (strict inequalities).

    Actor samples use nominal Waymo capture timestamps, in nanoseconds.  Actor
    IDs are contiguous and deterministic in first-visible-detection order;
    raw CAStrack IDs remain the grouping identity but are intentionally not
    exposed through the canonical schema.
    """

    if not isinstance(filter_static_actors, bool):
        raise TypeError("filter_static_actors must be a boolean")
    std_threshold = _non_negative_finite_float(
        static_std_threshold, name="static_std_threshold"
    )
    displacement_threshold = _non_negative_finite_float(
        static_displacement_threshold, name="static_displacement_threshold"
    )
    if (
        isinstance(lifecycle_padding_ns, bool)
        or not isinstance(lifecycle_padding_ns, int)
        or lifecycle_padding_ns < 0
    ):
        raise ValueError("lifecycle_padding_ns must be a non-negative integer")

    (
        sources,
        timestamps,
        timestamp_to_relative,
        transforms,
        frames_by_index,
    ) = _aligned_inputs(
        source_frame_indices=source_frame_indices,
        selected_timestamps_micros=selected_timestamps_micros,
        relative_indices=relative_indices,
        vehicle_transforms=vehicle_transforms,
        front_frames=front_frames,
    )
    _, payload = _read_json_object(path)
    scene = _scene_frames(payload, sequence=sequence)

    grouped: dict[int, list[_Detection]] = defaultdict(list)
    first_seen_order_by_object: dict[int, int] = {}
    next_seen_order = 0
    for source_frame_index, timestamp_micros in zip(sources, timestamps):
        frame_payload = scene.get(str(source_frame_index))
        if not isinstance(frame_payload, Mapping):
            raise ValueError(
                f"CAStrack scene is missing source frame {source_frame_index}"
            )
        object_ids = frame_payload.get("obj_ids")
        class_names = frame_payload.get("name")
        boxes = frame_payload.get("boxes_lidar")
        for values, name in (
            (object_ids, "obj_ids"),
            (class_names, "name"),
            (boxes, "boxes_lidar"),
        ):
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise ValueError(
                    f"CAStrack {name} at frame {source_frame_index} must be an array"
                )
        assert isinstance(object_ids, Sequence)
        assert isinstance(class_names, Sequence)
        assert isinstance(boxes, Sequence)
        if not (len(object_ids) == len(class_names) == len(boxes)):
            raise ValueError(
                f"CAStrack arrays have different lengths at frame {source_frame_index}"
            )

        seen_in_frame: set[int] = set()
        relative_index = timestamp_to_relative[timestamp_micros]
        front_frame = frames_by_index[relative_index]
        for raw_object_id, raw_class_name, box in zip(
            object_ids, class_names, boxes
        ):
            if (
                isinstance(raw_object_id, bool)
                or not isinstance(raw_object_id, int)
                or raw_object_id < 0
            ):
                raise ValueError(
                    f"CAStrack object IDs at frame {source_frame_index} must be "
                    "non-negative integers"
                )
            if raw_object_id in seen_in_frame:
                raise ValueError(
                    f"duplicate CAStrack object {raw_object_id} at frame "
                    f"{source_frame_index}"
                )
            seen_in_frame.add(raw_object_id)
            if raw_object_id not in first_seen_order_by_object:
                first_seen_order_by_object[raw_object_id] = next_seen_order
                next_seen_order += 1
            if raw_class_name not in _CLASS_NAMES:
                continue

            dimensions_lwh, world_from_actor = _box_to_world(
                box,
                world_from_vehicle=transforms[timestamp_micros],
                source_frame_index=source_frame_index,
                raw_object_id=raw_object_id,
            )
            if not _has_front_corner_visibility(
                front_frame,
                dimensions_lwh=dimensions_lwh,
                world_from_actor=world_from_actor,
            ):
                continue
            canonical_class_name = _CLASS_NAMES[str(raw_class_name)]
            records = grouped[raw_object_id]
            if records and records[0].class_name != canonical_class_name:
                raise ValueError(
                    f"CAStrack object {raw_object_id} changes class from "
                    f"{records[0].class_name} to {canonical_class_name}"
                )
            records.append(
                _Detection(
                    raw_object_id=raw_object_id,
                    class_name=canonical_class_name,
                    source_frame_index=source_frame_index,
                    timestamp_micros=timestamp_micros,
                    frame_index=relative_index,
                    dimensions_lwh=dimensions_lwh,
                    world_from_actor=world_from_actor,
                    first_seen_order=first_seen_order_by_object[raw_object_id],
                )
            )

    candidates: list[tuple[int, list[_Detection]]] = []
    for raw_object_id, records in grouped.items():
        records.sort(key=lambda record: record.timestamp_micros)
        if filter_static_actors:
            positions = torch.stack(
                [record.world_from_actor[:3, 3] for record in records]
            )
            coordinate_std = torch.std(positions, dim=0, correction=0)
            endpoint_displacement = torch.linalg.vector_norm(
                positions[-1] - positions[0]
            )
            dynamic = bool(torch.any(coordinate_std > std_threshold)) or (
                float(endpoint_displacement.item()) > displacement_threshold
            )
            if not dynamic:
                continue
        candidates.append((raw_object_id, records))
    candidates.sort(key=lambda item: item[1][0].first_seen_order)

    scene_times_ns = [timestamp * _MICROSECONDS_TO_NANOSECONDS for timestamp in timestamps]
    scene_times_ns.extend(int(frame.timestamp.item()) for frame in front_frames)
    lifecycle_lower = min(scene_times_ns)
    lifecycle_upper = max(scene_times_ns)

    tracks: list[ActorTrack] = []
    for actor_id, (_raw_object_id, records) in enumerate(candidates):
        dimensions_lwh = torch.stack(
            [record.dimensions_lwh for record in records]
        ).amax(dim=0)
        samples = tuple(
            ActorTrackSample(
                timestamp=torch.tensor(
                    record.timestamp_micros * _MICROSECONDS_TO_NANOSECONDS,
                    dtype=torch.int64,
                ),
                translation=record.world_from_actor[:3, 3].clone(),
                quaternion_wxyz=_rotation_matrix_to_quaternion_wxyz(
                    record.world_from_actor[:3, :3]
                ),
                frame_index=record.frame_index,
            )
            for record in records
        )
        tracks.append(
            ActorTrack(
                actor_id=actor_id,
                class_name=records[0].class_name,
                dimensions_lwh=dimensions_lwh,
                samples=samples,
                lifecycle_start_timestamp=torch.tensor(
                    max(
                        lifecycle_lower,
                        int(samples[0].timestamp.item()) - lifecycle_padding_ns,
                    ),
                    dtype=torch.int64,
                ),
                lifecycle_end_timestamp=torch.tensor(
                    min(
                        lifecycle_upper,
                        int(samples[-1].timestamp.item()) + lifecycle_padding_ns,
                    ),
                    dtype=torch.int64,
                ),
            )
        )
    return tuple(tracks)


__all__ = [
    "CASTRACK_ACTOR_SOURCE",
    "extract_castrack_scene_json",
    "load_castrack_actor_tracks",
]
