"""Memory-bounded adapter for the official nuScenes metadata tables.

Only key samples from one scene are materialized.  The large ``sample_data``,
``ego_pose``, and ``sample_annotation`` tables are streamed and filtered before
JSON decoding, which avoids the multi-gigabyte dictionaries created by the
official development kit on the full trainval release.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import replace
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Literal

import torch
from torch import Tensor

from .schema import (
    ActorTrack,
    ActorTrackSample,
    CanonicalDatasetManifest,
    CanonicalFrame,
    LidarFrame,
    LidarProjection,
)


NUSCENES_CAMERA_CHANNELS: tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

_LIDAR_CHANNEL = "LIDAR_TOP"
_SMALL_JSON_LIMIT = 64 * 1024 * 1024
_TOKEN_PATTERN = re.compile(r'"token"\s*:\s*"([^"]+)"')
_SAMPLE_TOKEN_PATTERN = re.compile(r'"sample_token"\s*:\s*"([^"]+)"')
_SAMPLE_DATA_TOKEN_PATTERN = re.compile(r"[0-9a-fA-F]{32}")
_INT64_MAX = 2**63 - 1
_ACTOR_LIFECYCLE_PADDING_NS = 100_000_000


def normalize_nuscenes_scene_name(scene: int | str) -> str:
    """Normalize ``61``, ``0061``, or ``scene-0061`` to ``scene-0061``."""

    if isinstance(scene, bool):
        raise TypeError("nuScenes scene selector must be an integer or string")
    if isinstance(scene, int):
        number = scene
    elif isinstance(scene, str):
        value = scene.strip().lower()
        if value.startswith("scene-"):
            value = value[6:]
        if not value or not value.isdecimal():
            raise ValueError(f"invalid nuScenes scene selector: {scene!r}")
        number = int(value)
    else:
        raise TypeError("nuScenes scene selector must be an integer or string")
    if number < 0 or number > 9999:
        raise ValueError("nuScenes scene number must be in [0, 9999]")
    return f"scene-{number:04d}"


def _normalize_sky_mask_reject_tokens(
    tokens: Collection[str] | None,
) -> frozenset[str]:
    normalized: set[str] = set()
    for token in tokens or ():
        if (
            not isinstance(token, str)
            or _SAMPLE_DATA_TOKEN_PATTERN.fullmatch(token) is None
        ):
            raise ValueError(
                "sky mask reject tokens must be 32-character hexadecimal strings"
            )
        canonical = token.lower()
        if canonical in normalized:
            raise ValueError(f"duplicate sky mask reject token: {token}")
        normalized.add(canonical)
    return frozenset(normalized)


def parse_nuscenes_sky_mask_reject_list(path: str | Path) -> frozenset[str]:
    """Parse a UTF-8 sample-data-token reject list.

    Blank lines, full-line comments, and trailing ``#`` comments are ignored.
    Tokens are normalized to lowercase so case-only duplicates cannot silently
    select the same nuScenes record twice.
    """

    reject_path = Path(path)
    if not reject_path.is_file():
        raise FileNotFoundError(
            f"nuScenes sky mask reject list does not exist: {reject_path}"
        )
    try:
        lines = reject_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(
            f"nuScenes sky mask reject list is not valid UTF-8: {reject_path}"
        ) from error

    tokens: set[str] = set()
    for line_number, raw_line in enumerate(lines, start=1):
        token = raw_line.partition("#")[0].strip()
        if not token:
            continue
        if _SAMPLE_DATA_TOKEN_PATTERN.fullmatch(token) is None:
            raise ValueError(
                f"malformed sky mask reject token on line {line_number}: {token!r}"
            )
        canonical = token.lower()
        if canonical in tokens:
            raise ValueError(
                f"duplicate sky mask reject token on line {line_number}: {token}"
            )
        tokens.add(canonical)
    return frozenset(tokens)


def _sky_mask_for_sample_data(
    root: Path | None,
    channel: str,
    record: Mapping[str, Any],
) -> Path | None:
    """Resolve ``<root>/<channel>/<sample_data_token>.png`` strictly."""

    if root is None:
        return None
    token = record.get("token")
    if not isinstance(token, str) or not token or Path(token).name != token:
        raise ValueError(f"{channel} sample_data has an invalid token")
    path = root / channel / f"{token}.png"
    if not path.is_file():
        raise FileNotFoundError(
            "missing nuScenes sky mask for "
            f"{channel} sample_data token {token}: {path}"
        )
    return path


def _metadata_roots(root: str | Path, version: str) -> tuple[Path, Path]:
    data_root = Path(root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"nuScenes data root does not exist: {data_root}")
    if not version or Path(version).name != version:
        raise ValueError("version must be a single non-empty directory name")
    nested = data_root / version
    if (nested / "scene.json").is_file():
        return data_root, nested
    if data_root.name == version and (data_root / "scene.json").is_file():
        return data_root.parent, data_root
    raise FileNotFoundError(
        f"nuScenes metadata not found at {nested} (or directly in {data_root})"
    )


def _table_path(metadata_root: Path, name: str) -> Path:
    path = metadata_root / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing nuScenes metadata table: {path}")
    return path


def _load_table(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid nuScenes JSON table: {path}") from error
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"nuScenes table must be an array of objects: {path}")
    return value


def _iter_official_raw_records(path: Path) -> Iterator[str]:
    """Yield objects from the pretty-printed official JSON array.

    nuScenes tables place each top-level object between a line containing only
    ``{`` and one containing ``}``/``},``.  This fast path deliberately scans
    by line rather than parsing every row in the multi-gigabyte trainval tables.
    Small/custom tables use :func:`_load_table` instead.
    """

    current: list[str] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if current is None:
                if stripped == "{":
                    current = [line]
                continue
            current.append(line)
            if stripped in {"}", "},"}:
                raw = "".join(current)
                if stripped == "},":
                    raw = raw.rstrip()[:-1]
                yield raw
                current = None
    if current is not None:
        raise ValueError(f"unterminated object in nuScenes JSON table: {path}")


def _select_records(
    path: Path,
    *,
    field: Literal["token", "sample_token"],
    wanted: set[str],
    keyframes_only: bool = False,
) -> list[dict[str, Any]]:
    if not wanted:
        return []
    if path.stat().st_size <= _SMALL_JSON_LIMIT:
        records = _load_table(path)
        return [
            record
            for record in records
            if record.get(field) in wanted
            and (not keyframes_only or record.get("is_key_frame") is True)
        ]

    pattern = _TOKEN_PATTERN if field == "token" else _SAMPLE_TOKEN_PATTERN
    selected: list[dict[str, Any]] = []
    for raw in _iter_official_raw_records(path):
        match = pattern.search(raw)
        if match is None or match.group(1) not in wanted:
            continue
        if keyframes_only and not re.search(
            r'"is_key_frame"\s*:\s*true\b', raw
        ):
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid object in nuScenes JSON table: {path}") from error
        if not isinstance(record, dict):
            raise ValueError(f"nuScenes table contains a non-object record: {path}")
        selected.append(record)
    return selected


def _index_by_token(records: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        token = record.get("token")
        if not isinstance(token, str) or not token:
            raise ValueError(f"{label} record is missing a token")
        if token in result:
            raise ValueError(f"duplicate {label} token: {token}")
        result[token] = record
    return result


def _finite_vector(record: Mapping[str, Any], key: str, length: int, label: str) -> Tensor:
    value = record.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} {key} must contain {length} values")
    try:
        result = torch.tensor([float(item) for item in value], dtype=torch.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} {key} must be numeric") from error
    if not torch.isfinite(result).all():
        raise ValueError(f"{label} {key} must be finite")
    return result


def _quaternion_rotation_wxyz(quaternion: Tensor, label: str) -> Tensor:
    if quaternion.shape != (4,) or not torch.isfinite(quaternion).all():
        raise ValueError(f"{label} quaternion must be finite [w,x,y,z]")
    norm = torch.linalg.vector_norm(quaternion)
    if norm <= 1.0e-12:
        raise ValueError(f"{label} quaternion cannot have zero norm")
    w, x, y, z = quaternion / norm
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w))),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w))),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))),
        )
    )


def _transform_from_record(record: Mapping[str, Any], label: str) -> Tensor:
    translation = _finite_vector(record, "translation", 3, label)
    quaternion = _finite_vector(record, "rotation", 4, label)
    transform = torch.eye(4, dtype=torch.float64)
    transform[:3, :3] = _quaternion_rotation_wxyz(quaternion, label)
    transform[:3, 3] = translation
    return transform


def _timestamp_ns(record: Mapping[str, Any], label: str) -> Tensor:
    value = record.get("timestamp")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} timestamp must be integer microseconds")
    nanoseconds = value * 1000
    if nanoseconds < 0 or nanoseconds > _INT64_MAX:
        raise OverflowError(f"{label} timestamp exceeds signed nanoseconds")
    return torch.tensor(nanoseconds, dtype=torch.int64)


def read_nuscenes_lidar_bin(path: str | Path) -> tuple[Tensor, Tensor]:
    """Read nuScenes ``[x,y,z,intensity,ring]`` and normalize intensity."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"nuScenes LiDAR scan does not exist: {source}")
    byte_count = source.stat().st_size
    if byte_count == 0 or byte_count % 20 != 0:
        raise ValueError("nuScenes LiDAR scan size must be a non-zero multiple of 20 bytes")
    if sys.byteorder != "little":
        raise RuntimeError("nuScenes LiDAR reader currently requires a little-endian host")
    values = torch.from_file(
        str(source), shared=False, size=byte_count // 4, dtype=torch.float32
    ).clone()
    scan = values.reshape(-1, 5)
    if not torch.isfinite(scan).all():
        raise ValueError("nuScenes LiDAR scan must contain only finite values")
    reflectance = scan[:, 3].contiguous() / 255.0
    return scan[:, :3].contiguous(), reflectance


def project_world_lidar_to_image(
    lidar: LidarFrame,
    camera_to_world: Tensor,
    intrinsics: Tensor,
    camera_id: int,
    image_size: tuple[int, int],
) -> LidarProjection:
    """Project one world-aligned LiDAR scan into an OpenCV camera."""

    if camera_to_world.shape != (4, 4) or intrinsics.shape != (3, 3):
        raise ValueError("camera_to_world and intrinsics must have shapes [4,4] and [3,3]")
    if not torch.isfinite(camera_to_world).all() or not torch.isfinite(intrinsics).all():
        raise ValueError("camera transform and intrinsics must be finite")
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("image_size must contain positive dimensions")

    points = lidar.world_points
    camera_from_world = torch.linalg.inv(camera_to_world).to(points)
    camera_points = torch.einsum(
        "ij,nj->ni", camera_from_world[:3, :3], points
    ) + camera_from_world[:3, 3]
    projected = torch.einsum("ij,nj->ni", intrinsics.to(points), camera_points)
    depth = camera_points[:, 2]
    finite = torch.isfinite(projected).all(dim=-1) & torch.isfinite(depth)
    positive = depth > 0
    safe_depth = torch.where(positive, depth, torch.ones_like(depth))
    coordinates = projected[:, :2] / safe_depth[:, None]
    x, y = coordinates.unbind(dim=-1)
    inside = finite & positive & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    source_indices = torch.nonzero(inside, as_tuple=False).squeeze(1)
    selected = coordinates.index_select(0, source_indices)
    return LidarProjection(
        camera_id=camera_id,
        source_point_indices=source_indices.to(dtype=torch.long),
        image_coordinates=selected,
        pixel_indices=torch.floor(selected).to(dtype=torch.long),
        depths=depth.index_select(0, source_indices),
        image_size=image_size,
    )


def _sample_chain(
    scene_record: Mapping[str, Any], samples: Mapping[str, Mapping[str, Any]]
) -> tuple[Mapping[str, Any], ...]:
    token = scene_record.get("first_sample_token")
    if not isinstance(token, str) or not token:
        raise ValueError("nuScenes scene is missing first_sample_token")
    result: list[Mapping[str, Any]] = []
    visited: set[str] = set()
    while token:
        if token in visited:
            raise ValueError("nuScenes sample chain contains a cycle")
        visited.add(token)
        try:
            record = samples[token]
        except KeyError as error:
            raise ValueError(f"nuScenes scene refers to missing sample: {token}") from error
        if record.get("scene_token") != scene_record.get("token"):
            raise ValueError("nuScenes sample chain crosses scene boundaries")
        result.append(record)
        next_token = record.get("next")
        if not isinstance(next_token, str):
            raise ValueError("nuScenes sample next token must be a string")
        token = next_token
    expected = scene_record.get("nbr_samples")
    if isinstance(expected, int) and expected != len(result):
        raise ValueError(
            f"nuScenes scene declares {expected} samples but chain contains {len(result)}"
        )
    if result[-1].get("token") != scene_record.get("last_sample_token"):
        raise ValueError("nuScenes sample chain does not end at last_sample_token")
    return tuple(result)


def _sensor_channels(
    sensors: Mapping[str, Mapping[str, Any]],
    calibrations: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for calibration_token, calibration in calibrations.items():
        sensor_token = calibration.get("sensor_token")
        if not isinstance(sensor_token, str) or sensor_token not in sensors:
            raise ValueError("calibrated_sensor refers to a missing sensor")
        channel = sensors[sensor_token].get("channel")
        if not isinstance(channel, str) or not channel:
            raise ValueError("nuScenes sensor is missing channel")
        result[calibration_token] = channel
    return result


def _camera_intrinsics(calibration: Mapping[str, Any], channel: str) -> Tensor:
    value = calibration.get("camera_intrinsic")
    if not isinstance(value, list) or len(value) != 3 or any(
        not isinstance(row, list) or len(row) != 3 for row in value
    ):
        raise ValueError(f"{channel} camera_intrinsic must have shape [3,3]")
    try:
        result = torch.tensor(value, dtype=torch.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{channel} camera_intrinsic must be numeric") from error
    if not torch.isfinite(result).all() or torch.abs(torch.linalg.det(result)) < 1.0e-12:
        raise ValueError(f"{channel} camera_intrinsic must be finite and invertible")
    return result


def _dynamic_actor_tracks(
    annotations: Sequence[Mapping[str, Any]],
    instances: Mapping[str, Mapping[str, Any]],
    categories: Mapping[str, Mapping[str, Any]],
    sample_frame_indices: Mapping[str, int],
    sample_timestamps: Mapping[str, Tensor],
    scene_timestamp_bounds: tuple[int, int],
    *,
    include_stationary: bool,
    rigid_motion_threshold: float,
    deformable_motion_threshold: float,
) -> tuple[ActorTrack, ...]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        instance_token = annotation.get("instance_token")
        sample_token = annotation.get("sample_token")
        if not isinstance(instance_token, str) or not isinstance(sample_token, str):
            raise ValueError("nuScenes annotation is missing instance/sample token")
        if sample_token in sample_frame_indices:
            grouped[instance_token].append(annotation)

    allowed_rigid = (
        "vehicle.car",
        "vehicle.bicycle",
        "vehicle.motorcycle",
        "vehicle.bus",
        "vehicle.truck",
        "vehicle.trailer",
        "movable_object.pushable_pullable",
    )
    candidates: list[tuple[str, str, str, list[Mapping[str, Any]]]] = []
    for instance_token, records in grouped.items():
        instance = instances.get(instance_token)
        if instance is None:
            raise ValueError(f"annotation refers to missing instance: {instance_token}")
        category_token = instance.get("category_token")
        category = categories.get(category_token) if isinstance(category_token, str) else None
        class_name = category.get("name") if category is not None else None
        if not isinstance(class_name, str):
            raise ValueError(f"instance refers to missing category: {instance_token}")
        kind: Literal["rigid", "deformable"] | None
        if class_name.startswith("human.pedestrian"):
            kind = "deformable"
        elif any(class_name == name or class_name.startswith(name + ".") for name in allowed_rigid):
            kind = "rigid"
        else:
            kind = None
        if kind is None:
            continue
        records.sort(key=lambda item: sample_frame_indices[str(item["sample_token"])])
        if not include_stationary:
            if len(records) < 2:
                continue
            translations = torch.stack(
                [_finite_vector(record, "translation", 3, "annotation") for record in records]
            )
            displacement = torch.pdist(translations[:, :2]).max().item()
            threshold = (
                deformable_motion_threshold if kind == "deformable" else rigid_motion_threshold
            )
            if displacement <= threshold:
                continue
        candidates.append((instance_token, class_name, kind, records))

    tracks: list[ActorTrack] = []
    for actor_id, (instance_token, class_name, _kind, records) in enumerate(
        sorted(candidates, key=lambda item: item[0])
    ):
        sizes = torch.stack(
            [_finite_vector(record, "size", 3, "annotation") for record in records]
        )
        # nuScenes stores size as [width, length, height].
        dimensions_lwh = torch.stack(
            (sizes[:, 1].max(), sizes[:, 0].max(), sizes[:, 2].max())
        )
        samples: list[ActorTrackSample] = []
        for record in records:
            sample_token = str(record["sample_token"])
            quaternion = _finite_vector(record, "rotation", 4, "annotation")
            quaternion = quaternion / torch.linalg.vector_norm(quaternion).clamp_min(1.0e-12)
            samples.append(
                ActorTrackSample(
                    timestamp=sample_timestamps[sample_token].clone(),
                    translation=_finite_vector(record, "translation", 3, "annotation"),
                    quaternion_wxyz=quaternion,
                    frame_index=sample_frame_indices[sample_token],
                )
            )
        tracks.append(
            ActorTrack(
                actor_id=actor_id,
                class_name=class_name,
                dimensions_lwh=dimensions_lwh,
                samples=tuple(samples),
                lifecycle_start_timestamp=torch.tensor(
                    max(
                        scene_timestamp_bounds[0],
                        int(samples[0].timestamp.item())
                        - _ACTOR_LIFECYCLE_PADDING_NS,
                    ),
                    dtype=torch.int64,
                ),
                lifecycle_end_timestamp=torch.tensor(
                    min(
                        scene_timestamp_bounds[1],
                        int(samples[-1].timestamp.item())
                        + _ACTOR_LIFECYCLE_PADDING_NS,
                    ),
                    dtype=torch.int64,
                ),
            )
        )
    return tuple(tracks)


def load_nuscenes_manifest(
    root: str | Path,
    *,
    scene: int | str = "0061",
    version: str = "v1.0-trainval",
    camera_channels: Sequence[str] = NUSCENES_CAMERA_CHANNELS,
    sky_mask_root: str | Path | None = None,
    sky_mask_reject_tokens: Collection[str] | None = None,
    require_lidar: bool = True,
    retain_unprojected_lidar: bool = False,
    include_stationary_actors: bool = False,
    rigid_motion_threshold: float = 0.5,
    deformable_motion_threshold: float = 0.25,
) -> CanonicalDatasetManifest:
    """Load one official nuScenes scene into ArmGS canonical records.

    When ``sky_mask_root`` is supplied, every selected keyframe must have a
    binary mask at ``<root>/<camera_channel>/<sample_data_token>.png``. Missing
    roots or masks fail eagerly. ``sky_mask_reject_tokens`` keeps those raw
    targets attached to their frames but marks their sky supervision invalid.
    """

    data_root, metadata_root = _metadata_roots(root, version)
    mask_root = Path(sky_mask_root) if sky_mask_root is not None else None
    reject_tokens = _normalize_sky_mask_reject_tokens(sky_mask_reject_tokens)
    if mask_root is not None and not mask_root.is_dir():
        raise FileNotFoundError(
            f"nuScenes sky mask root does not exist: {mask_root}"
        )
    if reject_tokens and mask_root is None:
        raise ValueError("sky mask reject tokens require sky_mask_root")
    scene_name = normalize_nuscenes_scene_name(scene)
    channels = tuple(camera_channels)
    if not channels or len(channels) != len(set(channels)):
        raise ValueError("camera_channels must be non-empty and unique")
    unknown = set(channels) - set(NUSCENES_CAMERA_CHANNELS)
    if unknown:
        raise ValueError(f"unknown nuScenes camera channels: {sorted(unknown)}")
    if not math.isfinite(rigid_motion_threshold) or rigid_motion_threshold < 0:
        raise ValueError("rigid_motion_threshold must be finite and non-negative")
    if not math.isfinite(deformable_motion_threshold) or deformable_motion_threshold < 0:
        raise ValueError("deformable_motion_threshold must be finite and non-negative")

    scenes = _load_table(_table_path(metadata_root, "scene"))
    matching_scenes = [record for record in scenes if record.get("name") == scene_name]
    if len(matching_scenes) != 1:
        raise ValueError(f"nuScenes scene not found or ambiguous: {scene_name}")
    scene_record = matching_scenes[0]
    all_samples = _index_by_token(_load_table(_table_path(metadata_root, "sample")), "sample")
    samples = _sample_chain(scene_record, all_samples)
    sample_tokens = {str(record["token"]) for record in samples}
    frame_indices = {str(record["token"]): index for index, record in enumerate(samples)}
    sample_timestamps = {
        str(record["token"]): _timestamp_ns(record, "sample") for record in samples
    }

    sensors = _index_by_token(_load_table(_table_path(metadata_root, "sensor")), "sensor")
    calibrations = _index_by_token(
        _load_table(_table_path(metadata_root, "calibrated_sensor")),
        "calibrated_sensor",
    )
    calibration_channels = _sensor_channels(sensors, calibrations)
    selected_data = _select_records(
        _table_path(metadata_root, "sample_data"),
        field="sample_token",
        wanted=sample_tokens,
        keyframes_only=True,
    )
    data_by_capture_channel: dict[tuple[str, str], Mapping[str, Any]] = {}
    desired_channels = set(channels) | {_LIDAR_CHANNEL}
    for record in selected_data:
        calibration_token = record.get("calibrated_sensor_token")
        if not isinstance(calibration_token, str) or calibration_token not in calibration_channels:
            raise ValueError("sample_data refers to missing calibrated_sensor")
        channel = calibration_channels[calibration_token]
        if channel not in desired_channels:
            continue
        sample_token = record.get("sample_token")
        if not isinstance(sample_token, str):
            raise ValueError("sample_data is missing sample_token")
        key = (sample_token, channel)
        if key in data_by_capture_channel:
            raise ValueError(f"duplicate key sample_data for {sample_token}/{channel}")
        data_by_capture_channel[key] = record

    required_channels = set(channels) | ({_LIDAR_CHANNEL} if require_lidar else set())
    for sample_token in sample_tokens:
        missing = [
            channel
            for channel in required_channels
            if (sample_token, channel) not in data_by_capture_channel
        ]
        if missing:
            raise ValueError(f"sample {sample_token} is missing key data: {sorted(missing)}")

    selected_camera_tokens = {
        str(record.get("token", "")).lower()
        for (_sample_token, channel), record in data_by_capture_channel.items()
        if channel in channels
    }
    unknown_reject_tokens = reject_tokens - selected_camera_tokens
    if unknown_reject_tokens:
        raise ValueError(
            "sky mask reject list contains tokens absent from the selected "
            "scene/cameras: " + ", ".join(sorted(unknown_reject_tokens))
        )

    pose_tokens = {
        str(record["ego_pose_token"])
        for record in data_by_capture_channel.values()
        if record.get("ego_pose_token") is not None
    }
    ego_poses = _index_by_token(
        _select_records(
            _table_path(metadata_root, "ego_pose"), field="token", wanted=pose_tokens
        ),
        "ego_pose",
    )
    if set(ego_poses) != pose_tokens:
        raise ValueError("one or more selected sample_data ego poses are missing")

    channel_ids = {channel: index for index, channel in enumerate(NUSCENES_CAMERA_CHANNELS)}
    frames: list[CanonicalFrame] = []
    for sample in samples:
        sample_token = str(sample["token"])
        capture_timestamp = sample_timestamps[sample_token]
        lidar_record = data_by_capture_channel.get((sample_token, _LIDAR_CHANNEL))
        lidar: LidarFrame | None = None
        if lidar_record is not None:
            lidar_path = data_root / str(lidar_record.get("filename", ""))
            if lidar_path.is_file():
                points, reflectance = read_nuscenes_lidar_bin(lidar_path)
                lidar_calibration_token = str(lidar_record["calibrated_sensor_token"])
                lidar_pose_token = str(lidar_record["ego_pose_token"])
                sensor_to_world = _transform_from_record(
                    ego_poses[lidar_pose_token], "LiDAR ego pose"
                ) @ _transform_from_record(
                    calibrations[lidar_calibration_token], "LiDAR calibration"
                )
                lidar = LidarFrame(
                    points=points,
                    reflectance=reflectance,
                    sensor_to_world=sensor_to_world,
                    source_path=lidar_path,
                )
            elif require_lidar:
                raise FileNotFoundError(f"missing nuScenes LiDAR scan: {lidar_path}")

        capture_frames: list[CanonicalFrame] = []
        for channel in channels:
            record = data_by_capture_channel[(sample_token, channel)]
            observation_timestamp = _timestamp_ns(record, f"{channel} sample_data")
            image_path = data_root / str(record.get("filename", ""))
            if not image_path.is_file():
                raise FileNotFoundError(f"missing nuScenes image: {image_path}")
            height, width = record.get("height"), record.get("width")
            if (
                isinstance(height, bool)
                or isinstance(width, bool)
                or not isinstance(height, int)
                or not isinstance(width, int)
                or height <= 0
                or width <= 0
            ):
                raise ValueError(f"{channel} sample_data has invalid image dimensions")
            calibration_token = str(record["calibrated_sensor_token"])
            pose_token = str(record["ego_pose_token"])
            camera_to_world = _transform_from_record(
                ego_poses[pose_token], f"{channel} ego pose"
            ) @ _transform_from_record(
                calibrations[calibration_token], f"{channel} calibration"
            )
            camera_id = channel_ids[channel]
            intrinsics = _camera_intrinsics(calibrations[calibration_token], channel)
            projection = (
                project_world_lidar_to_image(
                    lidar, camera_to_world, intrinsics, camera_id, (height, width)
                )
                if lidar is not None
                else None
            )
            capture_frames.append(
                CanonicalFrame(
                    timestamp=observation_timestamp,
                    camera_id=camera_id,
                    camera_convention="opencv",
                    camera_to_world=camera_to_world,
                    intrinsics=intrinsics,
                    image_path=image_path,
                    image_size=(height, width),
                    frame_index=frame_indices[sample_token],
                    capture_timestamp=capture_timestamp.clone(),
                    lidar=lidar,
                    lidar_projection=projection,
                    sky_mask_path=_sky_mask_for_sample_data(
                        mask_root, channel, record
                    ),
                    sky_supervision_valid=(
                        str(record["token"]).lower() not in reject_tokens
                    ),
                )
            )

        if lidar is not None and not retain_unprojected_lidar:
            projections = [frame.lidar_projection for frame in capture_frames]
            if any(projection is None for projection in projections):
                raise RuntimeError("internal nuScenes frame is missing LiDAR projection")
            retained = torch.unique(
                torch.cat(
                    [projection.source_point_indices for projection in projections if projection is not None]
                ),
                sorted=True,
            )
            compact_lidar = LidarFrame(
                points=lidar.points.index_select(0, retained),
                reflectance=lidar.reflectance.index_select(0, retained),
                sensor_to_world=lidar.sensor_to_world,
                source_path=lidar.source_path,
            )
            source_to_compact = torch.full(
                (lidar.points.shape[0],), -1, dtype=torch.long, device=lidar.points.device
            )
            source_to_compact[retained] = torch.arange(
                retained.numel(), dtype=torch.long, device=lidar.points.device
            )
            capture_frames = [
                replace(
                    frame,
                    lidar=compact_lidar,
                    lidar_projection=replace(
                        projection,
                        source_point_indices=source_to_compact.index_select(
                            0, projection.source_point_indices
                        ),
                    ),
                )
                for frame, projection in zip(
                    capture_frames,
                    (projection for projection in projections if projection is not None),
                )
            ]
        frames.extend(capture_frames)

    annotations = _select_records(
        _table_path(metadata_root, "sample_annotation"),
        field="sample_token",
        wanted=sample_tokens,
    )
    instance_tokens = {
        str(record["instance_token"])
        for record in annotations
        if record.get("instance_token") is not None
    }
    instances = _index_by_token(
        _select_records(
            _table_path(metadata_root, "instance"), field="token", wanted=instance_tokens
        ),
        "instance",
    )
    categories = _index_by_token(_load_table(_table_path(metadata_root, "category")), "category")
    actor_tracks = _dynamic_actor_tracks(
        annotations,
        instances,
        categories,
        frame_indices,
        sample_timestamps,
        (
            min(
                *(int(timestamp.item()) for timestamp in sample_timestamps.values()),
                *(int(frame.timestamp.item()) for frame in frames),
            ),
            max(
                *(int(timestamp.item()) for timestamp in sample_timestamps.values()),
                *(int(frame.timestamp.item()) for frame in frames),
            ),
        ),
        include_stationary=include_stationary_actors,
        rigid_motion_threshold=rigid_motion_threshold,
        deformable_motion_threshold=deformable_motion_threshold,
    )
    return CanonicalDatasetManifest(frames=tuple(frames), actor_tracks=actor_tracks)


__all__ = [
    "NUSCENES_CAMERA_CHANNELS",
    "load_nuscenes_manifest",
    "normalize_nuscenes_scene_name",
    "parse_nuscenes_sky_mask_reject_list",
    "project_world_lidar_to_image",
    "read_nuscenes_lidar_bin",
]
