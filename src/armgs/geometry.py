"""Quaternion geometry and timestamped actor pose interpolation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .structures import GaussianSet


def normalize_quaternion(quaternion: Tensor, eps: float = 1.0e-8) -> Tensor:
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must have final dimension four")
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if torch.any(norm.detach() < eps):
        raise ValueError("cannot normalize a near-zero quaternion")
    return quaternion / norm


def quaternion_multiply(left: Tensor, right: Tensor) -> Tensor:
    """Hamilton product for quaternions stored as ``[w,x,y,z]``."""

    if left.shape[-1] != 4 or right.shape[-1] != 4:
        raise ValueError("quaternions must have final dimension four")
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quaternion_to_rotation_matrix(quaternion: Tensor) -> Tensor:
    quaternion = normalize_quaternion(quaternion)
    w, x, y, z = quaternion.unbind(dim=-1)
    two = 2.0
    return torch.stack(
        (
            1.0 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1.0 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1.0 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def rotate_points(quaternion: Tensor, points: Tensor) -> Tensor:
    if points.shape[-1] != 3:
        raise ValueError("points must have final dimension three")
    rotation = quaternion_to_rotation_matrix(quaternion)
    return torch.matmul(rotation, points.unsqueeze(-1)).squeeze(-1)


def quaternion_slerp(left: Tensor, right: Tensor, weight: Tensor) -> Tensor:
    """Shortest-path spherical interpolation with stable near-linear fallback."""

    left = normalize_quaternion(left)
    right = normalize_quaternion(right)
    while weight.ndim < left.ndim:
        weight = weight.unsqueeze(-1)
    dot = (left * right).sum(dim=-1, keepdim=True)
    right = torch.where(dot < 0.0, -right, right)
    dot = dot.abs().clamp(max=1.0)

    theta = torch.acos(dot.clamp(max=1.0 - 1.0e-7))
    sin_theta = torch.sin(theta).clamp_min(1.0e-8)
    spherical = (
        torch.sin((1.0 - weight) * theta) / sin_theta * left
        + torch.sin(weight * theta) / sin_theta * right
    )
    linear = (1.0 - weight) * left + weight * right
    result = torch.where(dot > 0.9995, linear, spherical)
    return normalize_quaternion(result)


@dataclass(frozen=True)
class InterpolatedPose:
    quaternions: Tensor
    translations: Tensor


class PoseTrajectory(nn.Module):
    """Learnable actor poses at tracked timestamps with interpolation."""

    def __init__(
        self, timestamps: Tensor, quaternions: Tensor, translations: Tensor
    ) -> None:
        super().__init__()
        if timestamps.ndim != 1 or timestamps.numel() == 0:
            raise ValueError("timestamps must be a non-empty one-dimensional tensor")
        count = timestamps.numel()
        if quaternions.shape != (count, 4):
            raise ValueError("quaternions must have shape [T,4]")
        if translations.shape != (count, 3):
            raise ValueError("translations must have shape [T,3]")
        if not quaternions.is_floating_point() or not translations.is_floating_point():
            raise ValueError("quaternions and translations must be floating point")
        if quaternions.device != translations.device:
            raise ValueError("quaternions and translations must share a device")
        if quaternions.dtype != translations.dtype:
            raise ValueError("quaternions and translations must share a dtype")
        # Keep absolute time in float64. Casting large microsecond/nanosecond
        # values to float32 before subtraction can collapse adjacent frames.
        timestamps = timestamps.to(
            device=translations.device, dtype=torch.float64
        )
        if not torch.isfinite(timestamps).all():
            raise ValueError("timestamps must be finite")
        if count > 1 and torch.any(timestamps[1:] <= timestamps[:-1]):
            raise ValueError("timestamps must be strictly increasing")
        self.register_buffer("timestamps", timestamps.detach().clone())
        self.quaternions = nn.Parameter(normalize_quaternion(quaternions.detach().clone()))
        self.translations = nn.Parameter(translations.detach().clone())

    def _apply(self, fn):  # type: ignore[no-untyped-def]
        # Preserve float64 absolute timestamps across model.float()/half().
        timestamps = self.timestamps
        result = super()._apply(fn)
        self._buffers["timestamps"] = timestamps.to(
            device=self.timestamps.device, dtype=torch.float64
        )
        return result

    def interpolate(
        self, query_timestamps: Tensor, *, extrapolate: bool = False
    ) -> InterpolatedPose:
        """Interpolate poses, optionally extrapolating at trajectory boundaries."""

        if not isinstance(extrapolate, bool):
            raise TypeError("extrapolate must be a boolean")
        queries = query_timestamps.reshape(-1).to(self.timestamps)
        if not torch.isfinite(queries).all():
            raise ValueError("query timestamps must be finite")
        if self.timestamps.numel() == 1:
            return InterpolatedPose(
                quaternions=normalize_quaternion(self.quaternions).expand(
                    queries.numel(), -1
                ),
                translations=self.translations.expand(queries.numel(), -1),
            )

        upper = torch.searchsorted(self.timestamps, queries, right=False)
        upper = upper.clamp(1, self.timestamps.numel() - 1)
        lower = upper - 1
        lower_time = self.timestamps[lower]
        upper_time = self.timestamps[upper]
        weight = (queries - lower_time) / (upper_time - lower_time)
        if not extrapolate:
            weight = weight.clamp(0.0, 1.0)
        weight = weight.to(self.translations)

        translations = torch.lerp(
            self.translations[lower], self.translations[upper], weight[:, None]
        )
        quaternions = quaternion_slerp(
            self.quaternions[lower], self.quaternions[upper], weight
        )
        return InterpolatedPose(quaternions=quaternions, translations=translations)


def transform_actor_gaussians(
    gaussians: GaussianSet, pose_quaternion: Tensor, pose_translation: Tensor
) -> GaussianSet:
    """Move one actor's canonical Gaussian set into world coordinates."""

    if pose_quaternion.shape != (4,) or pose_translation.shape != (3,):
        raise ValueError("one actor pose must have shapes [4] and [3]")
    pose_quaternion = normalize_quaternion(pose_quaternion)
    expanded_pose = pose_quaternion.unsqueeze(0).expand(gaussians.count, -1)
    world_means = rotate_points(expanded_pose, gaussians.means) + pose_translation
    world_quaternions = normalize_quaternion(
        quaternion_multiply(expanded_pose, gaussians.quaternions)
    )
    return gaussians.with_updates(
        means=world_means,
        quaternions=world_quaternions,
    )
