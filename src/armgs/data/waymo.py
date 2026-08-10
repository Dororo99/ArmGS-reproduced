"""Memory-bounded Waymo Open Dataset v2 adapter for ArmGS.

The adapter consumes one v2 Parquet context and materializes the embedded
camera JPEGs into a deterministic PNG cache.  Camera observations, all LiDAR
sensors and both returns, vehicle poses, and native Waymo LiDAR boxes are
canonicalized into :mod:`armgs.data.schema` records.

Raw Waymo actor tracks remain the default ``waymo_gt`` fallback.  Supplying a
CAStrack JSON switches actor construction to StreetGaussians' detector/tracker
records while retaining the same canonical camera and world-coordinate frame.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from decimal import Decimal, ROUND_HALF_EVEN
from io import BytesIO
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Literal

import torch
from torch import Tensor

from .castrack import load_castrack_actor_tracks
from .nuscenes import project_world_lidar_to_image
from .schema import (
    ActorTrack,
    ActorTrackSample,
    CanonicalDatasetManifest,
    CanonicalFrame,
    LidarFrame,
    LidarProjection,
)


WAYMO_CAMERA_IDS: Mapping[str, int] = {
    "FRONT": 1,
    "FRONT_LEFT": 2,
    "FRONT_RIGHT": 3,
    "SIDE_LEFT": 4,
    "SIDE_RIGHT": 5,
}
WAYMO_CAMERA_CHANNELS: tuple[str, ...] = tuple(WAYMO_CAMERA_IDS)
WAYMO_ACTOR_SOURCE = "waymo_gt"

# Waymo camera extrinsics map the native camera sensor frame (+x forward,
# +y left, +z up) into the vehicle frame.  ArmGS/gsplat cameras use OpenCV
# coordinates (+x right, +y down, +z forward), hence this right-side change of
# basis when constructing camera-to-world.
WAYMO_OPENCV_TO_NATIVE = torch.tensor(
    (
        (0.0, 0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    dtype=torch.float64,
)

_TOP_LIDAR_ID = 1
_MICROSECONDS_TO_NANOSECONDS = 1_000
_NANOSECONDS_PER_SECOND = Decimal(1_000_000_000)
_ACTOR_LIFECYCLE_PADDING_NS = 100_000_000
_ACTOR_CLASS_NAMES: Mapping[int, str] = {
    1: "vehicle",
    2: "pedestrian",
    4: "cyclist",
}

_BASE_COMPONENTS = (
    "camera_image",
    "camera_calibration",
    "vehicle_pose",
    "lidar_box",
)
_LIDAR_COMPONENTS = ("lidar", "lidar_pose", "lidar_calibration")


def _configure_tensorflow_cpu_only() -> None:
    """Prevent Waymo TensorFlow utilities from claiming training GPUs."""

    try:
        import tensorflow as tf
    except ImportError:
        return
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as error:
        # Repeated calls after CPU-only initialization are harmless. A visible
        # GPU means another TensorFlow op initialized the runtime too early.
        if tf.config.get_visible_devices("GPU"):
            raise RuntimeError(
                "TensorFlow initialized Waymo GPU devices before ArmGS could "
                "disable them; start a fresh process and load Waymo through "
                "armgs.data.waymo before any TensorFlow GPU operation"
            ) from error
        return
    if tf.config.get_visible_devices("GPU"):
        raise RuntimeError("failed to make Waymo TensorFlow utilities CPU-only")


def _waymo_v2() -> Any:
    _configure_tensorflow_cpu_only()
    try:
        from waymo_open_dataset import v2
    except ImportError as error:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "Waymo loading requires the 'waymo-open-dataset-tf' v2 package"
        ) from error
    return v2


def _pyarrow_dataset() -> Any:
    try:
        import pyarrow.dataset as dataset
    except ImportError as error:  # pragma: no cover - optional dependency guard
        raise ImportError("Waymo loading requires optional dependency 'pyarrow'") from error
    return dataset


def _normalize_camera_channels(
    camera_channels: Sequence[str],
    camera_ids: Sequence[int] | None,
) -> tuple[str, ...]:
    if camera_ids is not None:
        if not camera_ids or len(set(camera_ids)) != len(camera_ids):
            raise ValueError("camera_ids must be non-empty and unique")
        names_by_id = {value: name for name, value in WAYMO_CAMERA_IDS.items()}
        unknown_ids = set(camera_ids) - set(names_by_id)
        if unknown_ids:
            raise ValueError(
                "unknown Waymo camera id(s): "
                + ", ".join(str(value) for value in sorted(unknown_ids))
            )
        requested = {names_by_id[value] for value in camera_ids}
    else:
        if not camera_channels:
            raise ValueError("camera_channels must be non-empty")
        normalized = tuple(str(value).strip().upper() for value in camera_channels)
        if any(not value for value in normalized) or len(set(normalized)) != len(
            normalized
        ):
            raise ValueError("camera_channels must be non-empty and unique")
        unknown = set(normalized) - set(WAYMO_CAMERA_CHANNELS)
        if unknown:
            raise ValueError(
                "unknown Waymo camera channel(s): " + ", ".join(sorted(unknown))
            )
        requested = set(normalized)
    # Stable Waymo priority keeps FRONT first even if a caller supplies a set
    # or an arbitrary ordering.
    return tuple(channel for channel in WAYMO_CAMERA_CHANNELS if channel in requested)


def _component_paths(
    root: Path,
    *,
    parquet_dir: str,
    sequence: str,
    require_lidar: bool,
) -> dict[str, Path]:
    if not parquet_dir or Path(parquet_dir).name != parquet_dir:
        raise ValueError("parquet_dir must be one non-empty directory name")
    if not sequence or Path(sequence).name != sequence:
        raise ValueError("sequence must be one non-empty context name")
    components = _BASE_COMPONENTS + (_LIDAR_COMPONENTS if require_lidar else ())
    paths = {
        component: root / parquet_dir / component / f"{sequence}.parquet"
        for component in components
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Waymo v2 component(s): " + ", ".join(missing))
    return paths


def _iter_parquet_rows(
    path: Path,
    *,
    equals_any: Mapping[str, Sequence[int]] | None = None,
    batch_size: int = 16,
) -> Iterator[dict[str, Any]]:
    """Yield filtered rows without expanding a whole range-image file."""

    dataset_module = _pyarrow_dataset()
    parquet_dataset = dataset_module.dataset(path, format="parquet")
    expression = None
    for column, values in (equals_any or {}).items():
        current = dataset_module.field(column).isin(list(values))
        expression = current if expression is None else expression & current
    try:
        batches = parquet_dataset.to_batches(
            filter=expression,
            batch_size=batch_size,
        )
        for batch in batches:
            yield from batch.to_pylist()
    except Exception as error:
        raise ValueError(f"failed to read Waymo parquet {path}: {error}") from error


def _matrix(values: Sequence[float], label: str) -> Tensor:
    try:
        result = torch.tensor(values, dtype=torch.float64).reshape(4, 4)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{label} must contain 16 numeric values") from error
    if not torch.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    expected = result.new_tensor((0.0, 0.0, 0.0, 1.0))
    if not torch.allclose(result[3], expected, atol=1.0e-6, rtol=1.0e-6):
        raise ValueError(f"{label} must be homogeneous")
    rotation = result[:3, :3]
    identity = torch.eye(3, dtype=result.dtype)
    if not torch.allclose(rotation.T @ rotation, identity, atol=1.0e-4, rtol=1.0e-4):
        raise ValueError(f"{label} rotation must be orthonormal")
    if not torch.allclose(torch.linalg.det(rotation), result.new_tensor(1.0), atol=1.0e-4):
        raise ValueError(f"{label} rotation must have determinant +1")
    return result


def _capture_timestamp_ns(timestamp_micros: int) -> Tensor:
    value = int(timestamp_micros) * _MICROSECONDS_TO_NANOSECONDS
    if value < -(2**63) or value > 2**63 - 1:
        raise ValueError("Waymo timestamp overflows signed nanoseconds")
    return torch.tensor(value, dtype=torch.int64)


def _observation_timestamp_ns(timestamp_seconds: float) -> Tensor:
    if not math.isfinite(timestamp_seconds):
        raise ValueError("Waymo camera pose timestamp must be finite")
    value = int(
        (Decimal(str(timestamp_seconds)) * _NANOSECONDS_PER_SECOND).to_integral_value(
            rounding=ROUND_HALF_EVEN
        )
    )
    if value < -(2**63) or value > 2**63 - 1:
        raise ValueError("Waymo camera pose timestamp overflows signed nanoseconds")
    return torch.tensor(value, dtype=torch.int64)


def _vehicle_poses(path: Path, sequence: str) -> tuple[list[int], dict[int, Any], dict[int, Tensor]]:
    v2 = _waymo_v2()
    components: dict[int, Any] = {}
    transforms: dict[int, Tensor] = {}
    for row in _iter_parquet_rows(path, batch_size=256):
        component = v2.VehiclePoseComponent.from_dict(row)
        if component.key.segment_context_name != sequence:
            raise ValueError("vehicle_pose contains a different context")
        timestamp = int(component.key.frame_timestamp_micros)
        if timestamp in components:
            raise ValueError(f"duplicate Waymo vehicle pose timestamp: {timestamp}")
        components[timestamp] = component
        transforms[timestamp] = _matrix(
            component.world_from_vehicle.transform,
            "world_from_vehicle",
        )
    timestamps = sorted(components)
    if not timestamps:
        raise ValueError("Waymo vehicle_pose contains no captures")
    return timestamps, components, transforms


def _world_center_from_vehicle_transforms(
    vehicle_transforms: Mapping[int, Tensor],
) -> Tensor:
    """Return the mean vehicle translation over a complete Waymo context."""

    if not vehicle_transforms:
        raise ValueError("Waymo vehicle transforms must be non-empty")
    translations = torch.stack(
        [transform[:3, 3] for transform in vehicle_transforms.values()],
        dim=0,
    )
    return translations.mean(dim=0)


def _center_vehicle_transforms(
    vehicle_transforms: Mapping[int, Tensor],
    world_center: Tensor,
) -> dict[int, Tensor]:
    centered: dict[int, Tensor] = {}
    for timestamp, transform in vehicle_transforms.items():
        result = transform.clone()
        result[:3, 3] -= world_center.to(result)
        centered[timestamp] = result
    return centered


def load_waymo_world_center(
    root: str | Path,
    *,
    sequence: str,
    parquet_dir: str = "training",
) -> Tensor:
    """Load the full-context mean vehicle translation used for centering.

    The center is intentionally computed before any selected frame range is
    applied, matching StreetGaussians' Waymo coordinate normalization.
    """

    data_root = Path(root).resolve(strict=True)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Waymo root is not a directory: {data_root}")
    if not parquet_dir or Path(parquet_dir).name != parquet_dir:
        raise ValueError("parquet_dir must be one non-empty directory name")
    if not sequence or Path(sequence).name != sequence:
        raise ValueError("sequence must be one non-empty context name")
    path = data_root / parquet_dir / "vehicle_pose" / f"{sequence}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing Waymo v2 vehicle_pose component: {path}")
    _timestamps, _components, transforms = _vehicle_poses(path, sequence)
    return _world_center_from_vehicle_transforms(transforms)


def _camera_calibrations(
    path: Path,
    *,
    sequence: str,
    raw_camera_ids: Sequence[int],
    target_size: tuple[int, int],
) -> dict[int, tuple[Any, Tensor]]:
    v2 = _waymo_v2()
    _target_height, target_width = target_size
    result: dict[int, tuple[Any, Tensor]] = {}
    for row in _iter_parquet_rows(
        path,
        equals_any={"key.camera_name": raw_camera_ids},
        batch_size=8,
    ):
        component = v2.CameraCalibrationComponent.from_dict(row)
        if component.key.segment_context_name != sequence:
            raise ValueError("camera_calibration contains a different context")
        camera_id = int(component.key.camera_name)
        if camera_id in result:
            raise ValueError(f"duplicate Waymo camera calibration: {camera_id}")
        source_width = int(component.width)
        source_height = int(component.height)
        if source_width <= 0 or source_height <= 0:
            raise ValueError(f"invalid source image size for Waymo camera {camera_id}")
        # StreetGaussians derives one non-upsampling scale from source width,
        # applies it to both rows of K, and floors the resized dimensions.
        # FRONT 1920x1280 therefore uses 5/6 for both axes even though the
        # materialized integer image is exactly 1600x1066.
        scale = min(1.0, target_width / source_width)
        derived_size = (int(source_height * scale), int(source_width * scale))
        if derived_size != target_size:
            raise ValueError(
                f"target_size {target_size} is incompatible with uniform Waymo "
                f"camera scaling; camera {camera_id} derives {derived_size}"
            )
        intrinsic = component.intrinsic
        intrinsics = torch.tensor(
            (
                (float(intrinsic.f_u) * scale, 0.0, float(intrinsic.c_u) * scale),
                (0.0, float(intrinsic.f_v) * scale, float(intrinsic.c_v) * scale),
                (0.0, 0.0, 1.0),
            ),
            dtype=torch.float64,
        )
        if not torch.isfinite(intrinsics).all() or torch.linalg.det(intrinsics) <= 0:
            raise ValueError(f"invalid intrinsics for Waymo camera {camera_id}")
        result[camera_id] = (component, intrinsics)
    missing = set(raw_camera_ids) - set(result)
    if missing:
        raise ValueError(
            "missing Waymo camera calibration(s): "
            + ", ".join(str(value) for value in sorted(missing))
        )
    return result


def _atomic_cache_image(
    encoded: bytes,
    path: Path,
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> Path:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - optional dependency guard
        raise ImportError("Waymo loading requires optional dependency 'Pillow'") from error

    target_height, target_width = target_size
    if path.is_file():
        try:
            with Image.open(path) as cached:
                if cached.size == (target_width, target_height):
                    return path
        except Exception:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    source_height, source_width = source_size
    temporary: Path | None = None
    try:
        with Image.open(BytesIO(encoded)) as image:
            image = image.convert("RGB")
            if image.size != (source_width, source_height):
                raise ValueError(
                    f"Waymo image has size {image.size}, expected "
                    f"{(source_width, source_height)}"
                )
            if image.size != (target_width, target_height):
                image = image.resize(
                    (target_width, target_height),
                    Image.Resampling.BILINEAR,
                )
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                image.save(handle, format="PNG", compress_level=3)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return path


def _camera_rows(
    path: Path,
    *,
    sequence: str,
    selected_timestamps: Sequence[int],
    source_indices: Mapping[int, int],
    channels: Sequence[str],
    calibrations: Mapping[int, tuple[Any, Tensor]],
    target_size: tuple[int, int],
    cache_root: Path,
    world_center: Tensor | None = None,
) -> dict[tuple[int, int], tuple[Tensor, Tensor, Tensor, Path]]:
    v2 = _waymo_v2()
    raw_ids = tuple(WAYMO_CAMERA_IDS[channel] for channel in channels)
    rows: dict[tuple[int, int], tuple[Tensor, Tensor, Tensor, Path]] = {}
    for row in _iter_parquet_rows(
        path,
        equals_any={
            "key.frame_timestamp_micros": selected_timestamps,
            "key.camera_name": raw_ids,
        },
        batch_size=8,
    ):
        component = v2.CameraImageComponent.from_dict(row)
        if component.key.segment_context_name != sequence:
            raise ValueError("camera_image contains a different context")
        timestamp = int(component.key.frame_timestamp_micros)
        raw_camera_id = int(component.key.camera_name)
        key = (timestamp, raw_camera_id)
        if key in rows:
            raise ValueError(f"duplicate Waymo camera image row: {key}")
        calibration, intrinsics = calibrations[raw_camera_id]
        image_pose = _matrix(component.pose.transform, "camera image pose")
        vehicle_from_camera_native = _matrix(
            calibration.extrinsic.transform,
            "camera extrinsic",
        )
        camera_to_world = (
            image_pose
            @ vehicle_from_camera_native
            @ WAYMO_OPENCV_TO_NATIVE.to(image_pose)
        )
        if world_center is not None:
            camera_to_world[:3, 3] -= world_center.to(camera_to_world)
        channel = next(name for name, value in WAYMO_CAMERA_IDS.items() if value == raw_camera_id)
        source_index = source_indices[timestamp]
        image_path = cache_root / sequence / "images" / channel / f"{source_index:08d}.png"
        if not isinstance(component.image, (bytes, bytearray)):
            raise ValueError("Waymo camera image payload must be bytes")
        image_path = _atomic_cache_image(
            bytes(component.image),
            image_path,
            source_size=(int(calibration.height), int(calibration.width)),
            target_size=target_size,
        )
        rows[key] = (
            _observation_timestamp_ns(float(component.pose_timestamp)),
            camera_to_world,
            intrinsics.clone(),
            image_path,
        )
    expected = {
        (timestamp, raw_camera_id)
        for timestamp in selected_timestamps
        for raw_camera_id in raw_ids
    }
    missing = expected - set(rows)
    extra = set(rows) - expected
    if missing or extra:
        raise ValueError(
            f"Waymo camera rows do not match selected captures; missing={len(missing)}, "
            f"extra={len(extra)}"
        )
    return rows


def _lidar_calibrations(path: Path, sequence: str) -> dict[int, Any]:
    v2 = _waymo_v2()
    result: dict[int, Any] = {}
    for row in _iter_parquet_rows(path, batch_size=8):
        component = v2.LiDARCalibrationComponent.from_dict(row)
        if component.key.segment_context_name != sequence:
            raise ValueError("lidar_calibration contains a different context")
        laser_id = int(component.key.laser_name)
        if laser_id in result:
            raise ValueError(f"duplicate Waymo LiDAR calibration: {laser_id}")
        _matrix(component.extrinsic.transform, "LiDAR extrinsic")
        result[laser_id] = component
    if not result:
        raise ValueError("Waymo lidar_calibration contains no sensors")
    return result


def _selected_lidar_pose_rows(
    path: Path,
    *,
    sequence: str,
    selected_timestamps: Sequence[int],
) -> Iterator[Any]:
    v2 = _waymo_v2()
    previous: int | None = None
    for row in _iter_parquet_rows(
        path,
        equals_any={"key.frame_timestamp_micros": selected_timestamps},
        batch_size=1,
    ):
        component = v2.LiDARPoseComponent.from_dict(row)
        if component.key.segment_context_name != sequence:
            raise ValueError("lidar_pose contains a different context")
        if int(component.key.laser_name) != _TOP_LIDAR_ID:
            raise ValueError("Waymo lidar_pose must contain only TOP LiDAR")
        timestamp = int(component.key.frame_timestamp_micros)
        if previous is not None and timestamp <= previous:
            raise ValueError("Waymo lidar_pose timestamps must be strictly increasing")
        previous = timestamp
        yield component


def _decode_lidar_frames(
    paths: Mapping[str, Path],
    *,
    sequence: str,
    selected_timestamps: Sequence[int],
    vehicle_components: Mapping[int, Any],
    vehicle_transforms: Mapping[int, Tensor],
    lidar_returns: Literal["first", "both"],
) -> dict[int, LidarFrame]:
    v2 = _waymo_v2()
    try:
        from waymo_open_dataset.v2.perception.utils import lidar_utils
    except ImportError as error:  # pragma: no cover - optional dependency guard
        raise ImportError("Waymo loading requires official v2 lidar_utils") from error

    calibrations = _lidar_calibrations(paths["lidar_calibration"], sequence)
    selected_set = set(selected_timestamps)
    pose_iterator = iter(
        _selected_lidar_pose_rows(
            paths["lidar_pose"],
            sequence=sequence,
            selected_timestamps=selected_timestamps,
        )
    )
    next_pose = next(pose_iterator, None)

    def pose_for_timestamp(timestamp: int) -> Any:
        nonlocal next_pose
        while next_pose is not None and int(next_pose.key.frame_timestamp_micros) < timestamp:
            next_pose = next(pose_iterator, None)
        if next_pose is None or int(next_pose.key.frame_timestamp_micros) != timestamp:
            raise ValueError(f"missing TOP LiDAR pixel pose at timestamp {timestamp}")
        return next_pose

    result: dict[int, LidarFrame] = {}
    current_timestamp: int | None = None
    seen_sensors: set[int] = set()
    point_parts: list[Tensor] = []
    reflectance_parts: list[Tensor] = []

    def flush() -> None:
        nonlocal current_timestamp, seen_sensors, point_parts, reflectance_parts
        if current_timestamp is None:
            return
        if seen_sensors != set(calibrations):
            raise ValueError(
                f"Waymo capture {current_timestamp} LiDAR sensors do not match calibration"
            )
        points = (
            torch.cat(point_parts, dim=0)
            if point_parts
            else torch.empty((0, 3), dtype=torch.float32)
        )
        reflectance = (
            torch.cat(reflectance_parts, dim=0)
            if reflectance_parts
            else torch.empty((0,), dtype=torch.float32)
        )
        result[current_timestamp] = LidarFrame(
            points=points,
            reflectance=reflectance,
            sensor_to_world=vehicle_transforms[current_timestamp].clone(),
            source_path=paths["lidar"],
        )
        current_timestamp = None
        seen_sensors = set()
        point_parts = []
        reflectance_parts = []

    previous_row_timestamp: int | None = None
    for row in _iter_parquet_rows(
        paths["lidar"],
        equals_any={"key.frame_timestamp_micros": selected_timestamps},
        batch_size=1,
    ):
        component = v2.LiDARComponent.from_dict(row)
        if component.key.segment_context_name != sequence:
            raise ValueError("lidar contains a different context")
        timestamp = int(component.key.frame_timestamp_micros)
        if timestamp not in selected_set:
            continue
        if previous_row_timestamp is not None and timestamp < previous_row_timestamp:
            raise ValueError("Waymo LiDAR rows must be ordered by timestamp")
        previous_row_timestamp = timestamp
        if current_timestamp is None:
            current_timestamp = timestamp
        elif timestamp != current_timestamp:
            flush()
            current_timestamp = timestamp
        laser_id = int(component.key.laser_name)
        if laser_id in seen_sensors:
            raise ValueError(f"duplicate Waymo LiDAR sensor {laser_id} at {timestamp}")
        if laser_id not in calibrations:
            raise ValueError(f"missing calibration for Waymo LiDAR sensor {laser_id}")
        seen_sensors.add(laser_id)
        pixel_pose = pose_for_timestamp(timestamp) if laser_id == _TOP_LIDAR_ID else None
        range_image_returns = component.range_image_returns
        if lidar_returns == "first":
            range_image_returns = range_image_returns[:1]
        for range_image in range_image_returns:
            if range_image is None:
                continue
            converted = lidar_utils.convert_range_image_to_point_cloud(
                range_image=range_image,
                calibration=calibrations[laser_id],
                pixel_pose=(pixel_pose.range_image_return1 if pixel_pose is not None else None),
                frame_pose=(vehicle_components[timestamp] if pixel_pose is not None else None),
                keep_polar_features=True,
            )
            values = torch.as_tensor(converted.numpy(), dtype=torch.float32)
            if values.ndim != 2 or values.shape[1] != 6:
                raise ValueError("official Waymo LiDAR conversion must return [N,6]")
            if not torch.isfinite(values).all():
                raise ValueError("Waymo LiDAR conversion returned non-finite values")
            reflectance_parts.append(values[:, 1].clone())
            point_parts.append(values[:, 3:6].clone())
    flush()
    missing = set(selected_timestamps) - set(result)
    if missing:
        raise ValueError(f"missing decoded Waymo LiDAR capture(s): {len(missing)}")
    return result


def _rotation_matrix_to_quaternion_wxyz(rotation: Tensor) -> Tensor:
    m00, m01, m02 = rotation[0]
    m10, m11, m12 = rotation[1]
    m20, m21, m22 = rotation[2]
    one = rotation.new_tensor(1.0)
    w = 0.5 * torch.sqrt(torch.clamp(one + m00 + m11 + m22, min=0.0))
    x = 0.5 * torch.sqrt(torch.clamp(one + m00 - m11 - m22, min=0.0))
    y = 0.5 * torch.sqrt(torch.clamp(one - m00 + m11 - m22, min=0.0))
    z = 0.5 * torch.sqrt(torch.clamp(one - m00 - m11 + m22, min=0.0))
    x = torch.copysign(x, m21 - m12)
    y = torch.copysign(y, m02 - m20)
    z = torch.copysign(z, m10 - m01)
    quaternion = torch.stack((w, x, y, z))
    return quaternion / torch.linalg.vector_norm(quaternion).clamp_min(1.0e-12)


def _actor_tracks(
    path: Path,
    *,
    sequence: str,
    selected_timestamps: Sequence[int],
    relative_indices: Mapping[int, int],
    vehicle_transforms: Mapping[int, Tensor],
    lifecycle_bounds: tuple[int, int],
    filter_static_actors: bool,
    static_std_threshold: float,
    static_displacement_threshold: float,
) -> tuple[ActorTrack, ...]:
    v2 = _waymo_v2()
    grouped: dict[str, list[tuple[Any, Tensor]]] = defaultdict(list)
    seen_keys: set[tuple[int, str]] = set()
    for row in _iter_parquet_rows(
        path,
        equals_any={"key.frame_timestamp_micros": selected_timestamps},
        batch_size=256,
    ):
        component = v2.LiDARBoxComponent.from_dict(row)
        if component.key.segment_context_name != sequence:
            raise ValueError("lidar_box contains a different context")
        class_type = int(component.type)
        # StreetGS excludes signs and misc/unknown categories from actors.
        if class_type not in _ACTOR_CLASS_NAMES:
            continue
        timestamp = int(component.key.frame_timestamp_micros)
        object_id = str(component.key.laser_object_id)
        key = (timestamp, object_id)
        if key in seen_keys:
            raise ValueError(f"duplicate Waymo LiDAR box: {key}")
        seen_keys.add(key)
        box = component.box
        center = torch.tensor(
            (float(box.center.x), float(box.center.y), float(box.center.z)),
            dtype=torch.float64,
        )
        size = torch.tensor(
            (float(box.size.x), float(box.size.y), float(box.size.z)),
            dtype=torch.float64,
        )
        heading = float(box.heading)
        if (
            not torch.isfinite(center).all()
            or not torch.isfinite(size).all()
            or torch.any(size <= 0)
            or not math.isfinite(heading)
        ):
            raise ValueError(f"invalid Waymo LiDAR box for actor {object_id}")
        cosine = math.cos(heading)
        sine = math.sin(heading)
        vehicle_from_actor = torch.eye(4, dtype=torch.float64)
        vehicle_from_actor[:3, :3] = torch.tensor(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
            dtype=torch.float64,
        )
        vehicle_from_actor[:3, 3] = center
        world_from_actor = vehicle_transforms[timestamp] @ vehicle_from_actor
        grouped[object_id].append((component, world_from_actor))

    candidates: list[tuple[str, list[tuple[Any, Tensor]]]] = []
    for object_id, records in grouped.items():
        records.sort(key=lambda item: int(item[0].key.frame_timestamp_micros))
        if filter_static_actors:
            if len(records) < 2:
                continue
            positions = torch.stack([world_from_actor[:3, 3] for _, world_from_actor in records])
            coordinate_std = torch.std(positions, dim=0, correction=0)
            endpoint_displacement = torch.linalg.vector_norm(positions[-1] - positions[0])
            if not (
                bool(torch.any(coordinate_std > static_std_threshold))
                or float(endpoint_displacement.item()) > static_displacement_threshold
            ):
                continue
        candidates.append((object_id, records))

    tracks: list[ActorTrack] = []
    for actor_id, (_object_id, records) in enumerate(sorted(candidates, key=lambda item: item[0])):
        sizes = torch.stack(
            [
                torch.tensor(
                    (
                        float(component.box.size.x),
                        float(component.box.size.y),
                        float(component.box.size.z),
                    ),
                    dtype=torch.float64,
                )
                for component, _ in records
            ]
        )
        dimensions_lwh = sizes.amax(dim=0)
        samples: list[ActorTrackSample] = []
        for component, world_from_actor in records:
            timestamp_micros = int(component.key.frame_timestamp_micros)
            samples.append(
                ActorTrackSample(
                    timestamp=_capture_timestamp_ns(timestamp_micros),
                    translation=world_from_actor[:3, 3].clone(),
                    quaternion_wxyz=_rotation_matrix_to_quaternion_wxyz(
                        world_from_actor[:3, :3]
                    ),
                    frame_index=relative_indices[timestamp_micros],
                )
            )
        class_name = _ACTOR_CLASS_NAMES[int(records[0][0].type)]
        tracks.append(
            ActorTrack(
                actor_id=actor_id,
                class_name=class_name,
                dimensions_lwh=dimensions_lwh,
                samples=tuple(samples),
                lifecycle_start_timestamp=torch.tensor(
                    max(
                        lifecycle_bounds[0],
                        int(samples[0].timestamp.item()) - _ACTOR_LIFECYCLE_PADDING_NS,
                    ),
                    dtype=torch.int64,
                ),
                lifecycle_end_timestamp=torch.tensor(
                    min(
                        lifecycle_bounds[1],
                        int(samples[-1].timestamp.item()) + _ACTOR_LIFECYCLE_PADDING_NS,
                    ),
                    dtype=torch.int64,
                ),
            )
        )
    return tuple(tracks)


def _sky_mask_path(
    root: Path | None,
    *,
    sequence: str,
    channel: str,
    source_frame_index: int,
) -> Path | None:
    """Resolve ``<root>/<sequence>/<channel>/<source-index:08d>.png``."""

    if root is None:
        return None
    path = root / sequence / channel / f"{source_frame_index:08d}.png"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing Waymo sky mask for {channel} frame {source_frame_index}: {path}"
        )
    return path


def _compact_lidar(
    lidar: LidarFrame,
    projections: Sequence[LidarProjection],
) -> tuple[LidarFrame, tuple[LidarProjection, ...]]:
    if not projections:
        return lidar, ()
    retained = torch.unique(
        torch.cat([projection.source_point_indices for projection in projections]),
        sorted=True,
    )
    compact = LidarFrame(
        points=lidar.points.index_select(0, retained),
        reflectance=lidar.reflectance.index_select(0, retained),
        sensor_to_world=lidar.sensor_to_world,
        source_path=lidar.source_path,
    )
    source_to_compact = torch.full(
        (lidar.points.shape[0],),
        -1,
        dtype=torch.long,
        device=lidar.points.device,
    )
    source_to_compact[retained] = torch.arange(
        retained.numel(),
        dtype=torch.long,
        device=lidar.points.device,
    )
    remapped = tuple(
        replace(
            projection,
            source_point_indices=source_to_compact.index_select(
                0, projection.source_point_indices
            ),
        )
        for projection in projections
    )
    return compact, remapped


def load_waymo_v2_manifest(
    root: str | Path,
    *,
    sequence: str,
    parquet_dir: str = "training",
    camera_channels: Sequence[str] = ("FRONT",),
    camera_ids: Sequence[int] | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    target_size: tuple[int, int] = (1066, 1600),
    cache_dir: str | Path | None = None,
    sky_mask_root: str | Path | None = None,
    require_lidar: bool = True,
    lidar_returns: Literal["first", "both"] = "both",
    retain_unprojected_lidar: bool = False,
    filter_static_actors: bool = True,
    static_std_threshold: float = 0.5,
    static_displacement_threshold: float = 2.0,
    center_world: bool = False,
    castrack_path: str | Path | None = None,
) -> CanonicalDatasetManifest:
    """Load one Waymo v2 context into the ArmGS canonical representation.

    ``start_frame`` and ``end_frame`` index the timestamp-sorted source
    captures and the end is inclusive, matching StreetGaussians' selected-frame
    convention.  Embedded JPEGs are decoded to
    ``<cache_dir>/<sequence>/images/<channel>/<source-index:08d>.png``.
    Optional sky masks use
    ``<sky_mask_root>/<sequence>/<channel>/<source-index:08d>.png``.

    Raw Waymo camera IDs may be supplied through ``camera_ids`` as a backwards
    compatible alias.  Canonical frame camera IDs are always zero-based
    (Waymo ID minus one).  Set ``require_lidar=False`` for camera-only jobs such
    as sky-mask generation; no range-image Parquet is opened in that mode.
    ``lidar_returns="first"`` matches StreetGaussians initialization, while
    the backwards-compatible default ``"both"`` decodes both range-image
    returns from every LiDAR sensor.  With ``center_world=True``, the mean
    vehicle translation over the complete context (not only the selected
    range) is subtracted consistently from camera, LiDAR, and actor poses.
    When ``castrack_path`` is provided, actor tracks come from that full or
    scene-extracted CAStrack JSON instead of ``lidar_box``.  CAStrack visibility
    follows StreetGaussians' FRONT-camera rule, so FRONT must be requested.
    """

    data_root = Path(root).resolve(strict=True)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Waymo root is not a directory: {data_root}")
    channels = _normalize_camera_channels(camera_channels, camera_ids)
    if castrack_path is not None and "FRONT" not in channels:
        raise ValueError("CAStrack actor loading requires the FRONT camera")
    if isinstance(start_frame, bool) or start_frame < 0:
        raise ValueError("start_frame must be a non-negative integer")
    if end_frame is not None and (isinstance(end_frame, bool) or end_frame < 0):
        raise ValueError("end_frame must be a non-negative integer or None")
    if (
        len(target_size) != 2
        or isinstance(target_size[0], bool)
        or isinstance(target_size[1], bool)
        or target_size[0] <= 0
        or target_size[1] <= 0
    ):
        raise ValueError("target_size must contain positive (height, width)")
    if not isinstance(filter_static_actors, bool):
        raise TypeError("filter_static_actors must be a boolean")
    if not isinstance(center_world, bool):
        raise TypeError("center_world must be a boolean")
    if lidar_returns not in ("first", "both"):
        raise ValueError("lidar_returns must be 'first' or 'both'")
    for value, label in (
        (static_std_threshold, "static_std_threshold"),
        (static_displacement_threshold, "static_displacement_threshold"),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{label} must be finite and non-negative")

    paths = _component_paths(
        data_root,
        parquet_dir=parquet_dir,
        sequence=sequence,
        require_lidar=require_lidar,
    )
    all_timestamps, vehicle_components, vehicle_transforms = _vehicle_poses(
        paths["vehicle_pose"], sequence
    )
    world_center = (
        _world_center_from_vehicle_transforms(vehicle_transforms)
        if center_world
        else None
    )
    if world_center is not None:
        vehicle_transforms = _center_vehicle_transforms(
            vehicle_transforms,
            world_center,
        )
    effective_end = len(all_timestamps) - 1 if end_frame is None else end_frame
    if start_frame >= len(all_timestamps):
        raise ValueError("start_frame is outside the Waymo context")
    if effective_end < start_frame or effective_end >= len(all_timestamps):
        raise ValueError("end_frame is outside or before the selected Waymo range")
    selected_timestamps = all_timestamps[start_frame : effective_end + 1]
    source_indices = {timestamp: index for index, timestamp in enumerate(all_timestamps)}
    relative_indices = {
        timestamp: index for index, timestamp in enumerate(selected_timestamps)
    }
    raw_camera_ids = tuple(WAYMO_CAMERA_IDS[channel] for channel in channels)
    calibrations = _camera_calibrations(
        paths["camera_calibration"],
        sequence=sequence,
        raw_camera_ids=raw_camera_ids,
        target_size=target_size,
    )
    cache_root = (
        Path(cache_dir)
        if cache_dir is not None
        else data_root / "armgs_cache"
    )
    camera_rows = _camera_rows(
        paths["camera_image"],
        sequence=sequence,
        selected_timestamps=selected_timestamps,
        source_indices=source_indices,
        channels=channels,
        calibrations=calibrations,
        target_size=target_size,
        cache_root=cache_root,
        world_center=world_center,
    )
    lidar_by_timestamp = (
        _decode_lidar_frames(
            paths,
            sequence=sequence,
            selected_timestamps=selected_timestamps,
            vehicle_components=vehicle_components,
            vehicle_transforms=vehicle_transforms,
            lidar_returns=lidar_returns,
        )
        if require_lidar
        else {}
    )
    sky_root = Path(sky_mask_root) if sky_mask_root is not None else None

    frames: list[CanonicalFrame] = []
    for relative_index, timestamp in enumerate(selected_timestamps):
        capture_timestamp = _capture_timestamp_ns(timestamp)
        lidar = lidar_by_timestamp.get(timestamp)
        capture_frames: list[CanonicalFrame] = []
        projections: list[LidarProjection] = []
        for channel in channels:
            raw_camera_id = WAYMO_CAMERA_IDS[channel]
            observation_timestamp, camera_to_world, intrinsics, image_path = camera_rows[
                (timestamp, raw_camera_id)
            ]
            canonical_camera_id = raw_camera_id - 1
            projection = (
                project_world_lidar_to_image(
                    lidar,
                    camera_to_world,
                    intrinsics,
                    canonical_camera_id,
                    target_size,
                )
                if lidar is not None
                else None
            )
            if projection is not None:
                projections.append(projection)
            capture_frames.append(
                CanonicalFrame(
                    timestamp=observation_timestamp,
                    camera_id=canonical_camera_id,
                    camera_convention="opencv",
                    camera_to_world=camera_to_world,
                    intrinsics=intrinsics,
                    image_path=image_path,
                    image_size=target_size,
                    frame_index=relative_index,
                    capture_timestamp=capture_timestamp.clone(),
                    lidar=lidar,
                    lidar_projection=projection,
                    sky_mask_path=_sky_mask_path(
                        sky_root,
                        sequence=sequence,
                        channel=channel,
                        source_frame_index=source_indices[timestamp],
                    ),
                )
            )
        if lidar is not None and not retain_unprojected_lidar:
            compact_lidar, compact_projections = _compact_lidar(lidar, projections)
            capture_frames = [
                replace(
                    frame,
                    lidar=compact_lidar,
                    lidar_projection=compact_projection,
                )
                for frame, compact_projection in zip(capture_frames, compact_projections)
            ]
        frames.extend(capture_frames)

    if castrack_path is not None:
        front_camera_id = WAYMO_CAMERA_IDS["FRONT"] - 1
        actor_tracks = load_castrack_actor_tracks(
            castrack_path,
            sequence=sequence,
            source_frame_indices=tuple(
                source_indices[timestamp] for timestamp in selected_timestamps
            ),
            selected_timestamps_micros=selected_timestamps,
            relative_indices=relative_indices,
            vehicle_transforms=vehicle_transforms,
            front_frames=tuple(
                frame for frame in frames if frame.camera_id == front_camera_id
            ),
            filter_static_actors=filter_static_actors,
            static_std_threshold=static_std_threshold,
            static_displacement_threshold=static_displacement_threshold,
        )
    else:
        scene_timestamps = [int(frame.timestamp.item()) for frame in frames]
        scene_timestamps.extend(
            int(_capture_timestamp_ns(timestamp).item())
            for timestamp in selected_timestamps
        )
        timestamp_bounds = (min(scene_timestamps), max(scene_timestamps))
        actor_tracks = _actor_tracks(
            paths["lidar_box"],
            sequence=sequence,
            selected_timestamps=selected_timestamps,
            relative_indices=relative_indices,
            vehicle_transforms=vehicle_transforms,
            lifecycle_bounds=timestamp_bounds,
            filter_static_actors=filter_static_actors,
            static_std_threshold=static_std_threshold,
            static_displacement_threshold=static_displacement_threshold,
        )
    return CanonicalDatasetManifest(frames=tuple(frames), actor_tracks=actor_tracks)


__all__ = [
    "WAYMO_ACTOR_SOURCE",
    "WAYMO_CAMERA_CHANNELS",
    "WAYMO_CAMERA_IDS",
    "WAYMO_OPENCV_TO_NATIVE",
    "load_waymo_world_center",
    "load_waymo_v2_manifest",
]
