"""Canonical, dependency-light dataset records for ArmGS training."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

import torch
from torch import Tensor
from torch.utils.data import Dataset


CameraConvention = Literal["opencv", "opengl"]


def _as_existing_file(value: str | Path, name: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist or is not a file: {path}")
    return path


def _require_finite(value: Tensor, name: str) -> None:
    if not value.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _validate_transform(value: Tensor, name: str) -> None:
    if value.shape != (4, 4):
        raise ValueError(f"{name} must have shape [4,4]")
    _require_finite(value, name)
    expected = value.new_tensor([0.0, 0.0, 0.0, 1.0])
    if not torch.allclose(value[3], expected, atol=1.0e-5, rtol=1.0e-5):
        raise ValueError(f"{name} must be a homogeneous rigid transform")
    rotation = value[:3, :3]
    identity = torch.eye(3, dtype=value.dtype, device=value.device)
    if not torch.allclose(rotation.T @ rotation, identity, atol=1.0e-4, rtol=1.0e-4):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not torch.allclose(torch.linalg.det(rotation), value.new_tensor(1.0), atol=1.0e-4):
        raise ValueError(f"{name} rotation must have determinant +1")


def _validate_timestamp(timestamp: Tensor, name: str = "timestamp") -> None:
    if timestamp.shape != () or timestamp.dtype != torch.int64:
        raise ValueError(f"{name} must be a scalar torch.int64 tensor")


@dataclass(frozen=True)
class LidarFrame:
    """A Velodyne scan in sensor coordinates plus its world transform."""

    points: Tensor
    reflectance: Tensor
    sensor_to_world: Tensor
    source_path: Path

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("lidar points must have shape [N,3]")
        if self.reflectance.shape != (self.points.shape[0],):
            raise ValueError("lidar reflectance must have shape [N]")
        _require_finite(self.points, "lidar points")
        _require_finite(self.reflectance, "lidar reflectance")
        if self.points.device != self.reflectance.device:
            raise ValueError("lidar points and reflectance must share a device")
        if self.points.dtype != self.reflectance.dtype:
            raise ValueError("lidar points and reflectance must share a dtype")
        _validate_transform(self.sensor_to_world, "sensor_to_world")
        object.__setattr__(
            self, "source_path", _as_existing_file(self.source_path, "lidar source")
        )

    @property
    def world_points(self) -> Tensor:
        """Transform scan points to the canonical world frame."""

        transform = self.sensor_to_world.to(self.points)
        return torch.einsum(
            "ij,nj->ni", transform[:3, :3], self.points
        ) + transform[:3, 3]


@dataclass(frozen=True)
class LidarProjection:
    """Positive-depth LiDAR samples that land inside one image."""

    camera_id: int
    source_point_indices: Tensor
    image_coordinates: Tensor
    pixel_indices: Tensor
    depths: Tensor
    image_size: tuple[int, int]

    def __post_init__(self) -> None:
        count = self.source_point_indices.numel()
        height, width = self.image_size
        if self.camera_id < 0:
            raise ValueError("camera_id must be non-negative")
        if height <= 0 or width <= 0:
            raise ValueError("image_size must contain positive dimensions")
        if self.source_point_indices.shape != (count,):
            raise ValueError("source_point_indices must have shape [M]")
        if self.source_point_indices.dtype != torch.long:
            raise ValueError("source_point_indices must have dtype torch.long")
        if torch.any(self.source_point_indices < 0):
            raise ValueError("source_point_indices cannot be negative")
        if self.image_coordinates.shape != (count, 2):
            raise ValueError("image_coordinates must have shape [M,2]")
        if self.pixel_indices.shape != (count, 2) or self.pixel_indices.dtype != torch.long:
            raise ValueError("pixel_indices must have shape [M,2] and dtype torch.long")
        if self.depths.shape != (count,):
            raise ValueError("depths must have shape [M]")
        _require_finite(self.image_coordinates, "image_coordinates")
        _require_finite(self.depths, "depths")
        devices = {
            self.source_point_indices.device,
            self.image_coordinates.device,
            self.pixel_indices.device,
            self.depths.device,
        }
        if len(devices) != 1:
            raise ValueError("lidar projection tensors must share a device")
        if torch.any(self.depths <= 0):
            raise ValueError("projected lidar depths must be positive")
        if count:
            x, y = self.image_coordinates.unbind(dim=-1)
            if torch.any((x < 0) | (x >= width) | (y < 0) | (y >= height)):
                raise ValueError("image_coordinates must lie inside the image")
            pixel_x, pixel_y = self.pixel_indices.unbind(dim=-1)
            if torch.any(
                (pixel_x < 0)
                | (pixel_x >= width)
                | (pixel_y < 0)
                | (pixel_y >= height)
            ):
                raise ValueError("pixel_indices must lie inside the image")


@dataclass(frozen=True)
class ActorTrackSample:
    """One canonical actor-to-world pose at an exact dataset timestamp."""

    timestamp: Tensor
    translation: Tensor
    quaternion_wxyz: Tensor
    frame_index: int
    occlusion: int | None = None
    truncation: int | None = None

    def __post_init__(self) -> None:
        _validate_timestamp(self.timestamp)
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self.translation.shape != (3,):
            raise ValueError("actor translation must have shape [3]")
        if self.quaternion_wxyz.shape != (4,):
            raise ValueError("actor quaternion must have shape [4]")
        _require_finite(self.translation, "actor translation")
        _require_finite(self.quaternion_wxyz, "actor quaternion")
        norm = torch.linalg.vector_norm(self.quaternion_wxyz)
        if not torch.allclose(norm, norm.new_tensor(1.0), atol=1.0e-5, rtol=1.0e-5):
            raise ValueError("actor quaternion must be normalized")


@dataclass(frozen=True)
class ActorTrack:
    """Object identity, size, and its finite lifecycle in the sequence."""

    actor_id: int
    class_name: str
    dimensions_lwh: Tensor
    samples: tuple[ActorTrackSample, ...]

    def __post_init__(self) -> None:
        if self.actor_id < 0:
            raise ValueError("actor_id must be non-negative")
        if not self.class_name.strip():
            raise ValueError("class_name cannot be empty")
        if self.dimensions_lwh.shape != (3,):
            raise ValueError("dimensions_lwh must have shape [3]")
        _require_finite(self.dimensions_lwh, "actor dimensions")
        if torch.any(self.dimensions_lwh <= 0):
            raise ValueError("actor dimensions must be positive")
        if not self.samples:
            raise ValueError("actor track must contain at least one sample")
        timestamps = [int(sample.timestamp.item()) for sample in self.samples]
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("actor sample timestamps must be strictly increasing")
        frame_indices = [sample.frame_index for sample in self.samples]
        if any(right <= left for left, right in zip(frame_indices, frame_indices[1:])):
            raise ValueError("actor frame indices must be strictly increasing")

    @property
    def lifecycle_timestamps(self) -> tuple[Tensor, Tensor]:
        return self.samples[0].timestamp, self.samples[-1].timestamp


@dataclass(frozen=True)
class CanonicalFrame:
    """One camera observation and all optional supervision aligned to it."""

    timestamp: Tensor
    camera_id: int
    camera_convention: CameraConvention
    camera_to_world: Tensor
    intrinsics: Tensor
    image_path: Path
    image_size: tuple[int, int]
    frame_index: int
    lidar: LidarFrame | None = None
    lidar_projection: LidarProjection | None = None
    sky_mask_path: Path | None = None
    actor_mask_path: Path | None = None

    def __post_init__(self) -> None:
        _validate_timestamp(self.timestamp)
        if self.camera_id < 0 or self.frame_index < 0:
            raise ValueError("camera_id and frame_index must be non-negative")
        if self.camera_convention not in ("opencv", "opengl"):
            raise ValueError("camera_convention must be 'opencv' or 'opengl'")
        _validate_transform(self.camera_to_world, "camera_to_world")
        if self.intrinsics.shape != (3, 3):
            raise ValueError("intrinsics must have shape [3,3]")
        _require_finite(self.intrinsics, "intrinsics")
        if torch.abs(torch.linalg.det(self.intrinsics)) < 1.0e-12:
            raise ValueError("intrinsics must be invertible")
        height, width = self.image_size
        if height <= 0 or width <= 0:
            raise ValueError("image_size must contain positive dimensions")
        object.__setattr__(self, "image_path", _as_existing_file(self.image_path, "image"))
        if self.sky_mask_path is not None:
            object.__setattr__(
                self,
                "sky_mask_path",
                _as_existing_file(self.sky_mask_path, "sky mask"),
            )
        if self.actor_mask_path is not None:
            object.__setattr__(
                self,
                "actor_mask_path",
                _as_existing_file(self.actor_mask_path, "actor mask"),
            )
        if self.lidar_projection is not None:
            if self.lidar is None:
                raise ValueError("lidar_projection requires lidar data")
            if self.lidar_projection.camera_id != self.camera_id:
                raise ValueError("lidar projection camera_id does not match the frame")
            if self.lidar_projection.image_size != self.image_size:
                raise ValueError("lidar projection image_size does not match the frame")
            if self.lidar_projection.source_point_indices.numel() and torch.any(
                self.lidar_projection.source_point_indices >= self.lidar.points.shape[0]
            ):
                raise ValueError("lidar projection refers to a missing source point")


@dataclass(frozen=True)
class CanonicalDatasetManifest(Sequence[CanonicalFrame]):
    """Indexable sequence manifest; timestamps are always signed nanoseconds."""

    frames: tuple[CanonicalFrame, ...]
    actor_tracks: tuple[ActorTrack, ...] = ()
    timestamp_unit: Literal["nanoseconds"] = "nanoseconds"

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("dataset manifest must contain at least one frame")
        actor_ids = [track.actor_id for track in self.actor_tracks]
        if len(actor_ids) != len(set(actor_ids)):
            raise ValueError("actor track ids must be unique")

    @overload
    def __getitem__(self, index: int) -> CanonicalFrame: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[CanonicalFrame, ...]: ...

    def __getitem__(self, index: int | slice) -> CanonicalFrame | tuple[CanonicalFrame, ...]:
        return self.frames[index]

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[CanonicalFrame]:
        return iter(self.frames)


class CanonicalFrameDataset(Dataset[CanonicalFrame]):
    """Minimal PyTorch Dataset view over a canonical manifest."""

    def __init__(self, manifest: CanonicalDatasetManifest) -> None:
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> CanonicalFrame:
        return self.manifest[index]


__all__ = [
    "ActorTrack",
    "ActorTrackSample",
    "CameraConvention",
    "CanonicalDatasetManifest",
    "CanonicalFrame",
    "CanonicalFrameDataset",
    "LidarFrame",
    "LidarProjection",
]
