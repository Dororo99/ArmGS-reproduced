"""Safe, torch-only readers for a useful KITTI sequence slice.

The loader targets the common odometry-style layout (``calib.txt``,
``poses.txt``, ``times.txt``, ``image_2/``, ``velodyne/``) and also accepts the
raw-data ``image_02/data`` directory spelling. KITTI poses are interpreted as
rectified camera-0 to world transforms. Canonical timestamps are integer
nanoseconds, never float32 seconds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import math
from pathlib import Path
import struct
import sys
from typing import Literal
import xml.etree.ElementTree as ET

import torch
from torch import Tensor

from .schema import (
    ActorTrack,
    ActorTrackSample,
    CameraConvention,
    CanonicalDatasetManifest,
    CanonicalFrame,
    LidarFrame,
    LidarProjection,
)


_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def _existing_file(path: str | Path, label: str) -> Path:
    result = Path(path)
    if not result.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {result}")
    return result


def _finite_tensor(values: Sequence[float], shape: tuple[int, ...], label: str) -> Tensor:
    tensor = torch.tensor(values, dtype=torch.float64).reshape(shape)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{label} must contain only finite values")
    return tensor


def _homogeneous(matrix: Tensor) -> Tensor:
    if matrix.shape != (3, 4):
        raise ValueError("matrix must have shape [3,4]")
    result = torch.eye(4, dtype=matrix.dtype, device=matrix.device)
    result[:3] = matrix
    return result


def _validate_rigid_transform(transform: Tensor, label: str) -> None:
    if transform.shape != (4, 4) or not torch.isfinite(transform).all():
        raise ValueError(f"{label} must be a finite [4,4] transform")
    expected = transform.new_tensor([0.0, 0.0, 0.0, 1.0])
    if not torch.allclose(transform[3], expected, atol=1.0e-6, rtol=1.0e-6):
        raise ValueError(f"{label} must have homogeneous bottom row [0,0,0,1]")
    rotation = transform[:3, :3]
    identity = torch.eye(3, dtype=transform.dtype, device=transform.device)
    if not torch.allclose(rotation.T @ rotation, identity, atol=1.0e-4, rtol=1.0e-4):
        raise ValueError(f"{label} rotation must be orthonormal")
    if not torch.allclose(torch.linalg.det(rotation), transform.new_tensor(1.0), atol=1.0e-4):
        raise ValueError(f"{label} rotation must have determinant +1")


@dataclass(frozen=True)
class KittiCalibration:
    """Rectified KITTI camera projections and Velodyne extrinsics."""

    projections: tuple[Tensor, Tensor, Tensor, Tensor]
    rectification: Tensor
    lidar_to_camera0: Tensor

    def __post_init__(self) -> None:
        if len(self.projections) != 4:
            raise ValueError("KITTI calibration requires P0 through P3")
        for camera_id, projection in enumerate(self.projections):
            if projection.shape != (3, 4):
                raise ValueError(f"P{camera_id} must have shape [3,4]")
            if not projection.is_floating_point() or not torch.isfinite(projection).all():
                raise ValueError(f"P{camera_id} must be finite floating point")
            intrinsic = projection[:, :3]
            if torch.abs(torch.linalg.det(intrinsic)) < 1.0e-12:
                raise ValueError(f"P{camera_id} left 3x3 block must be invertible")
        if self.rectification.shape != (3, 3):
            raise ValueError("R0_rect must have shape [3,3]")
        if not torch.isfinite(self.rectification).all():
            raise ValueError("R0_rect must be finite")
        identity = torch.eye(
            3, dtype=self.rectification.dtype, device=self.rectification.device
        )
        if not torch.allclose(
            self.rectification.T @ self.rectification,
            identity,
            atol=1.0e-4,
            rtol=1.0e-4,
        ):
            raise ValueError("R0_rect must be orthonormal")
        _validate_rigid_transform(self.lidar_to_camera0, "Tr_velo_to_cam")

    def projection(self, camera_id: int) -> Tensor:
        if camera_id < 0 or camera_id >= len(self.projections):
            raise ValueError("camera_id must be in [0,3]")
        return self.projections[camera_id]

    def intrinsics(self, camera_id: int) -> Tensor:
        return self.projection(camera_id)[:, :3]

    @property
    def lidar_to_rectified_camera0(self) -> Tensor:
        rectification = torch.eye(4, dtype=self.rectification.dtype)
        rectification[:3, :3] = self.rectification
        return rectification @ self.lidar_to_camera0

    def camera_from_rectified_camera0(self, camera_id: int) -> Tensor:
        """Return the rectified camera-i <- rectified camera-0 transform."""

        projection = self.projection(camera_id)
        translation = torch.linalg.solve(projection[:, :3], projection[:, 3])
        result = torch.eye(4, dtype=projection.dtype, device=projection.device)
        result[:3, 3] = translation
        return result

    def camera_to_world(self, base_camera0_to_world: Tensor, camera_id: int) -> Tensor:
        _validate_rigid_transform(base_camera0_to_world, "camera0_to_world")
        camera_i_from_camera0 = self.camera_from_rectified_camera0(camera_id).to(
            base_camera0_to_world
        )
        result = base_camera0_to_world @ torch.linalg.inv(camera_i_from_camera0)
        _validate_rigid_transform(result, "camera_to_world")
        return result


def parse_kitti_calibration(path: str | Path) -> KittiCalibration:
    """Parse P0-P3, R0_rect, and Tr_velo_to_cam from KITTI text."""

    source = _existing_file(path, "KITTI calibration")
    entries: dict[str, list[float]] = {}
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"invalid calibration line {line_number}: missing ':'")
        key, raw_values = line.split(":", 1)
        key = key.strip()
        if key in entries:
            raise ValueError(f"duplicate calibration key: {key}")
        try:
            values = [float(item) for item in raw_values.split()]
        except ValueError as error:
            raise ValueError(f"invalid numeric value for calibration key {key}") from error
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"calibration key {key} must be finite")
        entries[key] = values

    def values_for(*keys: str) -> list[float]:
        present = [key for key in keys if key in entries]
        if not present:
            raise ValueError(f"missing calibration key {keys[0]}")
        if len(present) > 1:
            raise ValueError(f"ambiguous aliases for calibration key {keys[0]}")
        return entries[present[0]]

    projections = tuple(
        _finite_tensor(
            values_for(f"P{camera_id}", f"P_rect_0{camera_id}"),
            (3, 4),
            f"P{camera_id}",
        )
        for camera_id in range(4)
    )
    rectification = _finite_tensor(
        values_for("R0_rect", "R_rect_00"), (3, 3), "R0_rect"
    )
    lidar_raw = _finite_tensor(
        values_for("Tr_velo_to_cam", "Tr"), (3, 4), "Tr_velo_to_cam"
    )
    return KittiCalibration(
        projections=projections,  # type: ignore[arg-type]
        rectification=rectification,
        lidar_to_camera0=_homogeneous(lidar_raw),
    )


def parse_kitti_poses(path: str | Path) -> tuple[Tensor, ...]:
    """Read odometry ``poses.txt`` as rectified camera-0 to world matrices."""

    source = _existing_file(path, "KITTI poses")
    poses: list[Tensor] = []
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            values = [float(item) for item in line.split()]
        except ValueError as error:
            raise ValueError(f"invalid pose value on line {line_number}") from error
        if len(values) != 12 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"pose line {line_number} must contain 12 finite values")
        pose = _homogeneous(_finite_tensor(values, (3, 4), f"pose line {line_number}"))
        _validate_rigid_transform(pose, f"pose line {line_number}")
        poses.append(pose)
    if not poses:
        raise ValueError("poses file must contain at least one pose")
    return tuple(poses)


def parse_kitti_timestamps(
    path: str | Path, *, units_per_second: int = 1_000_000_000
) -> Tensor:
    """Convert decimal seconds to strictly increasing signed int64 timestamps."""

    source = _existing_file(path, "KITTI times")
    if units_per_second <= 0:
        raise ValueError("units_per_second must be positive")
    result: list[int] = []
    scale = Decimal(units_per_second)
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        value = raw_line.strip()
        if not value:
            continue
        try:
            decimal_value = Decimal(value)
        except InvalidOperation as error:
            raise ValueError(f"invalid timestamp on line {line_number}") from error
        if not decimal_value.is_finite():
            raise ValueError(f"timestamp on line {line_number} must be finite")
        timestamp = int((decimal_value * scale).to_integral_value(rounding=ROUND_HALF_EVEN))
        if timestamp < _INT64_MIN or timestamp > _INT64_MAX:
            raise OverflowError(f"timestamp on line {line_number} exceeds int64")
        if result and timestamp <= result[-1]:
            raise ValueError("timestamps must be strictly increasing after conversion")
        result.append(timestamp)
    if not result:
        raise ValueError("times file must contain at least one timestamp")
    return torch.tensor(result, dtype=torch.int64)


def read_velodyne_bin(path: str | Path) -> tuple[Tensor, Tensor]:
    """Read a little-endian KITTI ``[x,y,z,reflectance]`` float32 scan."""

    source = _existing_file(path, "Velodyne scan")
    byte_count = source.stat().st_size
    if byte_count == 0 or byte_count % 16 != 0:
        raise ValueError("Velodyne scan size must be a non-zero multiple of 16 bytes")
    if sys.byteorder != "little":
        raise RuntimeError("KITTI Velodyne reader currently requires a little-endian host")
    values = torch.from_file(
        str(source), shared=False, size=byte_count // 4, dtype=torch.float32
    ).clone()
    scan = values.reshape(-1, 4)
    if not torch.isfinite(scan).all():
        raise ValueError("Velodyne scan must contain only finite values")
    return scan[:, :3].contiguous(), scan[:, 3].contiguous()


def project_velodyne_to_image(
    points: Tensor,
    calibration: KittiCalibration,
    camera_id: int,
    image_size: tuple[int, int],
) -> LidarProjection:
    """Project valid Velodyne points, filtering z<=0 and out-of-frame samples."""

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N,3]")
    if not points.is_floating_point() or not torch.isfinite(points).all():
        raise ValueError("points must be finite floating point")
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("image_size must contain positive dimensions")
    transform = calibration.lidar_to_rectified_camera0.to(points)
    projection = calibration.projection(camera_id).to(points)
    homogeneous = torch.cat((points, torch.ones_like(points[:, :1])), dim=-1)
    rectified = torch.einsum("ij,nj->ni", transform, homogeneous)
    projected = torch.einsum("ij,nj->ni", projection, rectified)
    depth = projected[:, 2]
    finite = torch.isfinite(projected).all(dim=-1)
    positive = depth > 0
    safe_depth = torch.where(positive, depth, torch.ones_like(depth))
    image_coordinates = projected[:, :2] / safe_depth[:, None]
    x, y = image_coordinates.unbind(dim=-1)
    inside = finite & positive & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    source_indices = torch.nonzero(inside, as_tuple=False).squeeze(1)
    selected_coordinates = image_coordinates[source_indices]
    pixel_indices = torch.floor(selected_coordinates).to(dtype=torch.long)
    return LidarProjection(
        camera_id=camera_id,
        source_point_indices=source_indices.to(dtype=torch.long),
        image_coordinates=selected_coordinates,
        pixel_indices=pixel_indices,
        depths=depth[source_indices],
        image_size=image_size,
    )


@dataclass(frozen=True)
class KittiTrackletPose:
    frame_index: int
    translation_lidar: Tensor
    rotation_rpy: Tensor
    occlusion: int | None = None
    truncation: int | None = None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("tracklet frame_index must be non-negative")
        if self.translation_lidar.shape != (3,) or self.rotation_rpy.shape != (3,):
            raise ValueError("tracklet translation and rotation must have shape [3]")
        if not torch.isfinite(self.translation_lidar).all() or not torch.isfinite(
            self.rotation_rpy
        ).all():
            raise ValueError("tracklet pose must be finite")


@dataclass(frozen=True)
class KittiTracklet:
    object_type: str
    dimensions_hwl: Tensor
    first_frame: int
    poses: tuple[KittiTrackletPose, ...]

    def __post_init__(self) -> None:
        if not self.object_type.strip():
            raise ValueError("tracklet object type cannot be empty")
        if self.first_frame < 0:
            raise ValueError("tracklet first_frame must be non-negative")
        if self.dimensions_hwl.shape != (3,) or not torch.isfinite(
            self.dimensions_hwl
        ).all():
            raise ValueError("tracklet dimensions must be finite [h,w,l]")
        if torch.any(self.dimensions_hwl <= 0):
            raise ValueError("tracklet dimensions must be positive")
        if not self.poses:
            raise ValueError("tracklet must contain at least one pose")


def _child_text(parent: ET.Element, tag: str, *, required: bool = True) -> str | None:
    child = parent.find(tag)
    text = child.text.strip() if child is not None and child.text is not None else None
    if required and not text:
        raise ValueError(f"tracklet XML is missing <{tag}>")
    return text


def _xml_float(parent: ET.Element, tag: str) -> float:
    raw = _child_text(parent, tag)
    try:
        value = float(raw)  # type: ignore[arg-type]
    except ValueError as error:
        raise ValueError(f"tracklet <{tag}> must be numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"tracklet <{tag}> must be finite")
    return value


def _xml_optional_int(parent: ET.Element, tag: str) -> int | None:
    raw = _child_text(parent, tag, required=False)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"tracklet <{tag}> must be an integer") from error


def parse_kitti_tracklets(path: str | Path) -> tuple[KittiTracklet, ...]:
    """Parse the geometry/lifecycle subset of a KITTI tracklet XML file."""

    source = _existing_file(path, "KITTI tracklets")
    try:
        root = ET.parse(source).getroot()
    except ET.ParseError as error:
        raise ValueError(f"invalid KITTI tracklet XML: {source}") from error
    tracklets_root = root if root.tag == "tracklets" else root.find("tracklets")
    if tracklets_root is None:
        raise ValueError("tracklet XML must contain a <tracklets> element")
    items = tracklets_root.findall("item")
    declared_count = _xml_optional_int(tracklets_root, "count")
    if declared_count is not None and declared_count != len(items):
        raise ValueError("tracklet count does not match the number of items")

    result: list[KittiTracklet] = []
    for item in items:
        object_type = _child_text(item, "objectType")
        first_frame_raw = _child_text(item, "first_frame")
        try:
            first_frame = int(first_frame_raw)  # type: ignore[arg-type]
        except ValueError as error:
            raise ValueError("tracklet <first_frame> must be an integer") from error
        dimensions = torch.tensor(
            [_xml_float(item, "h"), _xml_float(item, "w"), _xml_float(item, "l")],
            dtype=torch.float64,
        )
        poses_root = item.find("poses")
        if poses_root is None:
            raise ValueError("tracklet item must contain <poses>")
        pose_items = poses_root.findall("item")
        pose_count = _xml_optional_int(poses_root, "count")
        if pose_count is not None and pose_count != len(pose_items):
            raise ValueError("tracklet pose count does not match the number of items")
        poses: list[KittiTrackletPose] = []
        for offset, pose_item in enumerate(pose_items):
            translation = torch.tensor(
                [
                    _xml_float(pose_item, "tx"),
                    _xml_float(pose_item, "ty"),
                    _xml_float(pose_item, "tz"),
                ],
                dtype=torch.float64,
            )
            rotation = torch.tensor(
                [
                    _xml_float(pose_item, "rx"),
                    _xml_float(pose_item, "ry"),
                    _xml_float(pose_item, "rz"),
                ],
                dtype=torch.float64,
            )
            poses.append(
                KittiTrackletPose(
                    frame_index=first_frame + offset,
                    translation_lidar=translation,
                    rotation_rpy=rotation,
                    occlusion=_xml_optional_int(pose_item, "occlusion"),
                    truncation=_xml_optional_int(pose_item, "truncation"),
                )
            )
        result.append(
            KittiTracklet(
                object_type=object_type or "",
                dimensions_hwl=dimensions,
                first_frame=first_frame,
                poses=tuple(poses),
            )
        )
    return tuple(result)


def _rpy_rotation_matrix(rotation_rpy: Tensor) -> Tensor:
    roll, pitch, yaw = rotation_rpy.unbind()
    one = roll.new_tensor(1.0)
    zero = roll.new_tensor(0.0)
    rx = torch.stack(
        (
            one,
            zero,
            zero,
            zero,
            torch.cos(roll),
            -torch.sin(roll),
            zero,
            torch.sin(roll),
            torch.cos(roll),
        )
    ).reshape(3, 3)
    ry = torch.stack(
        (
            torch.cos(pitch),
            zero,
            torch.sin(pitch),
            zero,
            one,
            zero,
            -torch.sin(pitch),
            zero,
            torch.cos(pitch),
        )
    ).reshape(3, 3)
    rz = torch.stack(
        (
            torch.cos(yaw),
            -torch.sin(yaw),
            zero,
            torch.sin(yaw),
            torch.cos(yaw),
            zero,
            zero,
            zero,
            one,
        )
    ).reshape(3, 3)
    return rz @ ry @ rx


def _rotation_matrix_to_quaternion_wxyz(rotation: Tensor) -> Tensor:
    """Numerically stable matrix-to-quaternion conversion for one rotation."""

    # All branches use positive square roots and copysign, avoiding division by
    # the trace near 180-degree rotations.
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


def canonicalize_kitti_tracklets(
    tracklets: Sequence[KittiTracklet],
    timestamps: Tensor,
    camera0_to_world: Sequence[Tensor],
    calibration: KittiCalibration,
) -> tuple[ActorTrack, ...]:
    """Transform Velodyne-frame KITTI tracks into world-frame actor tracks."""

    if timestamps.ndim != 1 or timestamps.dtype != torch.int64:
        raise ValueError("timestamps must be a one-dimensional int64 tensor")
    if len(camera0_to_world) != timestamps.numel():
        raise ValueError("pose and timestamp counts must match")
    camera0_from_lidar = calibration.lidar_to_rectified_camera0
    tracks: list[ActorTrack] = []
    for actor_id, tracklet in enumerate(tracklets):
        samples: list[ActorTrackSample] = []
        for pose in tracklet.poses:
            if pose.frame_index >= timestamps.numel():
                raise ValueError("tracklet frame is outside pose/timestamp sequence")
            world_from_camera0 = camera0_to_world[pose.frame_index]
            _validate_rigid_transform(world_from_camera0, "camera0_to_world")
            world_from_lidar = world_from_camera0 @ camera0_from_lidar.to(
                world_from_camera0
            )
            lidar_from_actor = torch.eye(4, dtype=world_from_lidar.dtype)
            lidar_from_actor[:3, :3] = _rpy_rotation_matrix(
                pose.rotation_rpy.to(world_from_lidar)
            )
            lidar_from_actor[:3, 3] = pose.translation_lidar.to(world_from_lidar)
            world_from_actor = world_from_lidar @ lidar_from_actor
            samples.append(
                ActorTrackSample(
                    timestamp=timestamps[pose.frame_index].clone(),
                    translation=world_from_actor[:3, 3],
                    quaternion_wxyz=_rotation_matrix_to_quaternion_wxyz(
                        world_from_actor[:3, :3]
                    ),
                    frame_index=pose.frame_index,
                    occlusion=pose.occlusion,
                    truncation=pose.truncation,
                )
            )
        h, w, length = tracklet.dimensions_hwl
        tracks.append(
            ActorTrack(
                actor_id=actor_id,
                class_name=tracklet.object_type,
                dimensions_lwh=torch.stack((length, w, h)),
                samples=tuple(samples),
            )
        )
    return tuple(tracks)


def read_png_size(path: str | Path) -> tuple[int, int]:
    """Read ``(height,width)`` from a PNG IHDR without an image dependency."""

    source = _existing_file(path, "PNG image")
    with source.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"not a valid PNG header: {source}")
    width, height = struct.unpack(">II", header[16:24])
    if width == 0 or height == 0:
        raise ValueError("PNG dimensions must be positive")
    return height, width


def _resolve_required(root: Path, override: str | Path | None, name: str) -> Path:
    return _existing_file(Path(override) if override is not None else root / name, name)


def _image_directory(root: Path, camera_id: int) -> Path:
    candidates = (root / f"image_{camera_id}", root / f"image_{camera_id:02d}" / "data")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"missing image directory for camera {camera_id}; tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _indexed_images(directory: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        try:
            frame_index = int(path.stem)
        except ValueError:
            continue
        if frame_index in result:
            raise ValueError(f"duplicate image frame index {frame_index} in {directory}")
        result[frame_index] = path
    if not result:
        raise ValueError(f"image directory contains no numerically named images: {directory}")
    return result


def _mask_for_frame(directory: Path | None, image_path: Path, label: str) -> Path | None:
    if directory is None:
        return None
    if not directory.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {directory}")
    candidates = (directory / image_path.name, directory / f"{image_path.stem}.png")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing {label} for frame {image_path.stem} in {directory}")


def load_kitti_manifest(
    root: str | Path,
    *,
    camera_ids: Sequence[int] = (2,),
    image_size: tuple[int, int] | Mapping[int, tuple[int, int]] | None = None,
    calibration_path: str | Path | None = None,
    poses_path: str | Path | None = None,
    times_path: str | Path | None = None,
    velodyne_dir: str | Path | None = None,
    tracklet_path: str | Path | None = None,
    sky_mask_dirs: Mapping[int, str | Path] | None = None,
    actor_mask_dirs: Mapping[int, str | Path] | None = None,
    require_lidar: bool = True,
    camera_convention: CameraConvention = "opencv",
) -> CanonicalDatasetManifest:
    """Load a validated, indexable canonical manifest from a KITTI sequence."""

    sequence_root = Path(root)
    if not sequence_root.is_dir():
        raise FileNotFoundError(f"KITTI sequence root does not exist: {sequence_root}")
    if not camera_ids or len(set(camera_ids)) != len(camera_ids):
        raise ValueError("camera_ids must be non-empty and unique")
    if camera_convention != "opencv":
        raise ValueError("KITTI loader emits OpenCV camera coordinates")
    calibration = parse_kitti_calibration(
        _resolve_required(sequence_root, calibration_path, "calib.txt")
    )
    poses = parse_kitti_poses(_resolve_required(sequence_root, poses_path, "poses.txt"))
    timestamps = parse_kitti_timestamps(
        _resolve_required(sequence_root, times_path, "times.txt")
    )
    if len(poses) != timestamps.numel():
        raise ValueError("KITTI pose and timestamp counts must match")

    image_maps = {
        camera_id: _indexed_images(_image_directory(sequence_root, camera_id))
        for camera_id in camera_ids
    }
    frame_indices = set(next(iter(image_maps.values())))
    for camera_id, mapping in image_maps.items():
        if set(mapping) != frame_indices:
            raise ValueError(f"camera {camera_id} image frame indices do not match")
    if max(frame_indices) >= len(poses):
        raise ValueError("image frame index is outside pose/timestamp sequence")

    lidar_directory = Path(velodyne_dir) if velodyne_dir is not None else sequence_root / "velodyne"
    if require_lidar and not lidar_directory.is_dir():
        raise FileNotFoundError(f"Velodyne directory does not exist: {lidar_directory}")
    sky_directories = {
        camera_id: Path(path) for camera_id, path in (sky_mask_dirs or {}).items()
    }
    actor_directories = {
        camera_id: Path(path) for camera_id, path in (actor_mask_dirs or {}).items()
    }
    unknown_mask_cameras = (set(sky_directories) | set(actor_directories)) - set(camera_ids)
    if unknown_mask_cameras:
        raise ValueError("mask directories reference cameras not requested by camera_ids")

    frames: list[CanonicalFrame] = []
    for frame_index in sorted(frame_indices):
        timestamp = timestamps[frame_index]
        base_pose = poses[frame_index]
        source_stem = next(iter(image_maps.values()))[frame_index].stem
        scan_path = lidar_directory / f"{source_stem}.bin"
        lidar: LidarFrame | None = None
        if scan_path.is_file():
            points, reflectance = read_velodyne_bin(scan_path)
            world_from_lidar = base_pose.to(points) @ calibration.lidar_to_rectified_camera0.to(
                points
            )
            lidar = LidarFrame(
                points=points,
                reflectance=reflectance,
                sensor_to_world=world_from_lidar,
                source_path=scan_path,
            )
        elif require_lidar:
            raise FileNotFoundError(f"missing Velodyne scan: {scan_path}")

        for camera_id in camera_ids:
            image_path = image_maps[camera_id][frame_index]
            if isinstance(image_size, Mapping):
                if camera_id not in image_size:
                    raise ValueError(f"image_size is missing camera {camera_id}")
                current_size = image_size[camera_id]
            elif image_size is not None:
                current_size = image_size
            else:
                if image_path.suffix.lower() != ".png":
                    raise ValueError("image_size is required for non-PNG images")
                current_size = read_png_size(image_path)
            projection = (
                project_velodyne_to_image(lidar.points, calibration, camera_id, current_size)
                if lidar is not None
                else None
            )
            frames.append(
                CanonicalFrame(
                    timestamp=timestamp.clone(),
                    camera_id=camera_id,
                    camera_convention=camera_convention,
                    camera_to_world=calibration.camera_to_world(base_pose, camera_id),
                    intrinsics=calibration.intrinsics(camera_id).clone(),
                    image_path=image_path,
                    image_size=current_size,
                    frame_index=frame_index,
                    lidar=lidar,
                    lidar_projection=projection,
                    sky_mask_path=_mask_for_frame(
                        sky_directories.get(camera_id), image_path, "sky mask"
                    ),
                    actor_mask_path=_mask_for_frame(
                        actor_directories.get(camera_id), image_path, "actor mask"
                    ),
                )
            )

    actor_tracks: tuple[ActorTrack, ...] = ()
    if tracklet_path is not None:
        actor_tracks = canonicalize_kitti_tracklets(
            parse_kitti_tracklets(tracklet_path), timestamps, poses, calibration
        )
    return CanonicalDatasetManifest(frames=tuple(frames), actor_tracks=actor_tracks)


__all__ = [
    "KittiCalibration",
    "KittiTracklet",
    "KittiTrackletPose",
    "canonicalize_kitti_tracklets",
    "load_kitti_manifest",
    "parse_kitti_calibration",
    "parse_kitti_poses",
    "parse_kitti_timestamps",
    "parse_kitti_tracklets",
    "project_velodyne_to_image",
    "read_png_size",
    "read_velodyne_bin",
]
