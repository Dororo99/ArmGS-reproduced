"""Validated tensor containers and renderer-facing contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isqrt
from typing import Any, Protocol

import torch
from torch import Tensor


@dataclass(frozen=True)
class GaussianSet:
    """A collection of 3-D Gaussian attributes.

    Quaternions use the ``[w, x, y, z]`` convention throughout this package.
    SH coefficients use shape ``[N, (degree + 1)^2, 3]``.
    """

    means: Tensor
    quaternions: Tensor
    scales: Tensor
    opacities: Tensor
    sh_coefficients: Tensor
    group_ids: Tensor | None = None

    def __post_init__(self) -> None:
        count = self.means.shape[0]
        expected = {
            "means": (count, 3),
            "quaternions": (count, 4),
            "scales": (count, 3),
            "opacities": (count, 1),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if self.sh_coefficients.ndim != 3:
            raise ValueError("sh_coefficients must have shape [N, K, 3]")
        if self.sh_coefficients.shape[0] != count or self.sh_coefficients.shape[2] != 3:
            raise ValueError("sh_coefficients must have shape [N, K, 3]")
        coefficient_count = self.sh_coefficients.shape[1]
        degree_plus_one = isqrt(coefficient_count)
        if degree_plus_one * degree_plus_one != coefficient_count:
            raise ValueError("SH coefficient count must be a perfect square")
        if self.group_ids is not None and self.group_ids.shape != (count,):
            raise ValueError("group_ids must have shape [N]")

        tensors = [
            self.means,
            self.quaternions,
            self.scales,
            self.opacities,
            self.sh_coefficients,
        ]
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("all Gaussian tensors must be on the same device")
        if len({tensor.dtype for tensor in tensors}) != 1:
            raise ValueError("all Gaussian floating tensors must share a dtype")

    @property
    def count(self) -> int:
        return self.means.shape[0]

    @property
    def sh_degree(self) -> int:
        return isqrt(self.sh_coefficients.shape[1]) - 1

    def with_updates(self, **changes: Tensor | None) -> "GaussianSet":
        return replace(self, **changes)

    @classmethod
    def concatenate(cls, sets: list["GaussianSet"]) -> "GaussianSet":
        if not sets:
            raise ValueError("at least one GaussianSet is required")
        has_groups = [item.group_ids is not None for item in sets]
        if any(has_groups) and not all(has_groups):
            raise ValueError("group_ids must be present on every set or none")
        return cls(
            means=torch.cat([item.means for item in sets], dim=0),
            quaternions=torch.cat([item.quaternions for item in sets], dim=0),
            scales=torch.cat([item.scales for item in sets], dim=0),
            opacities=torch.cat([item.opacities for item in sets], dim=0),
            sh_coefficients=torch.cat(
                [item.sh_coefficients for item in sets], dim=0
            ),
            group_ids=(
                torch.cat([item.group_ids for item in sets], dim=0)  # type: ignore[arg-type]
                if all(has_groups)
                else None
            ),
        )


@dataclass(frozen=True)
class RasterizationInput:
    """Activated Gaussian attributes and camera-specific precomputed RGB.

    ``colors`` is ``[N,3]`` for one camera or ``[B,N,3]`` for a camera batch.
    Scales and opacities are expected after their positivity/sigmoid activation.
    ``camera_to_world`` accepts ``[B,3,4]`` or homogeneous ``[B,4,4]``.
    ``camera_convention`` explicitly selects OpenCV (+Z forward, +Y down) or
    OpenGL (-Z forward, +Y up); the backend converts to its native convention.
    """
    means: Tensor
    quaternions: Tensor
    scales: Tensor
    opacities: Tensor
    colors: Tensor
    camera_to_world: Tensor
    intrinsics: Tensor
    image_size: tuple[int, int]
    group_ids: Tensor | None = None
    camera_convention: str = "opencv"
    velocities: Tensor | None = None
    camera_linear_velocity: Tensor | None = None
    camera_angular_velocity: Tensor | None = None
    rolling_shutter_time: Tensor | None = None
    rolling_shutter_direction: int = 1


@dataclass(frozen=True)
class RasterizationOutput:
    rgb: Tensor
    depth: Tensor
    accumulated_alpha: Tensor
    actor_alpha: Tensor | None = None
    group_alpha: Tensor | None = None
    group_labels: Tensor | None = None
    metadata: dict[str, Any] | None = None


class GaussianRasterizer(Protocol):
    """Backend contract; implementations may use gsplat or another CUDA kernel."""

    def __call__(self, inputs: RasterizationInput) -> RasterizationOutput: ...

