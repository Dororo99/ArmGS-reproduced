"""Point-cloud initialization utilities for ArmGS Gaussian scenes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .structures import GaussianSet

_SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class GaussianInitializationConfig:
    """Explicit assumptions not specified by the ArmGS paper."""

    sh_degree: int = 3
    initial_opacity: float = 0.1
    initial_scale: float = 0.05
    voxel_size: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.sh_degree <= 3:
            raise ValueError("sh_degree must lie in [0, 3]")
        if not 0.0 < self.initial_opacity < 1.0:
            raise ValueError("initial_opacity must lie in (0, 1)")
        if not torch.isfinite(torch.tensor(self.initial_scale)):
            raise ValueError("initial_scale must be finite")
        if self.initial_scale <= 0:
            raise ValueError("initial_scale must be positive")
        if self.voxel_size is not None:
            if not torch.isfinite(torch.tensor(self.voxel_size)):
                raise ValueError("voxel_size must be finite")
            if self.voxel_size <= 0:
                raise ValueError("voxel_size must be positive")


def _validated_points_and_colors(
    points: Tensor, colors: Tensor | None
) -> tuple[Tensor, Tensor]:
    if points.ndim != 2 or points.shape[-1] != 3 or points.shape[0] == 0:
        raise ValueError("points must have non-empty shape [N,3]")
    if not points.is_floating_point():
        raise ValueError("points must be floating point")
    if not torch.isfinite(points).all():
        raise ValueError("points must be finite")
    if colors is None:
        colors = torch.full_like(points, 0.5)
    if colors.shape != points.shape:
        raise ValueError("colors must have shape [N,3]")
    if not colors.is_floating_point():
        raise ValueError("colors must be floating point")
    colors = colors.to(points)
    if not torch.isfinite(colors).all():
        raise ValueError("colors must be finite")
    if torch.any(colors < 0.0) or torch.any(colors > 1.0):
        raise ValueError("colors must lie in [0, 1]")
    return points, colors


def voxel_downsample(
    points: Tensor,
    colors: Tensor,
    voxel_size: float,
) -> tuple[Tensor, Tensor]:
    """Average points and RGB within deterministic integer voxels."""

    points, colors = _validated_points_and_colors(points, colors)
    if not torch.isfinite(torch.tensor(voxel_size)) or voxel_size <= 0:
        raise ValueError("voxel_size must be finite and positive")
    coordinates = torch.floor(points / voxel_size).to(torch.int64)
    _, inverse = torch.unique(
        coordinates, dim=0, sorted=True, return_inverse=True
    )
    voxel_count = int(inverse.max().item()) + 1
    point_sums = torch.zeros(
        voxel_count, 3, device=points.device, dtype=points.dtype
    )
    color_sums = torch.zeros_like(point_sums)
    counts = torch.zeros(
        voxel_count, 1, device=points.device, dtype=points.dtype
    )
    point_sums.index_add_(0, inverse, points)
    color_sums.index_add_(0, inverse, colors)
    counts.index_add_(
        0,
        inverse,
        torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype),
    )
    return point_sums / counts, color_sums / counts


def initialize_gaussians_from_points(
    points: Tensor,
    colors: Tensor | None = None,
    *,
    config: GaussianInitializationConfig | None = None,
) -> GaussianSet:
    """Create conventional 3DGS parameters from LiDAR or SfM points."""

    config = config or GaussianInitializationConfig()
    points, colors = _validated_points_and_colors(points, colors)
    if config.voxel_size is not None:
        points, colors = voxel_downsample(points, colors, config.voxel_size)

    count = points.shape[0]
    coefficient_count = (config.sh_degree + 1) ** 2
    sh_coefficients = torch.zeros(
        count,
        coefficient_count,
        3,
        device=points.device,
        dtype=points.dtype,
    )
    sh_coefficients[:, 0] = (colors - 0.5) / _SH_C0
    quaternions = torch.zeros(
        count, 4, device=points.device, dtype=points.dtype
    )
    quaternions[:, 0] = 1.0
    return GaussianSet(
        means=points.clone(),
        quaternions=quaternions,
        scales=torch.full(
            (count, 3),
            config.initial_scale,
            device=points.device,
            dtype=points.dtype,
        ),
        opacities=torch.full(
            (count, 1),
            config.initial_opacity,
            device=points.device,
            dtype=points.dtype,
        ),
        sh_coefficients=sh_coefficients,
    )


def world_points_to_actor_local(
    world_points: Tensor,
    actor_to_world: Tensor,
    *,
    box_dimensions: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Transform points into an actor frame and optionally filter its 3-D box."""

    if world_points.ndim != 2 or world_points.shape[-1] != 3:
        raise ValueError("world_points must have shape [N,3]")
    if not world_points.is_floating_point() or not torch.isfinite(world_points).all():
        raise ValueError("world_points must be finite floating point")
    if actor_to_world.shape == (3, 4):
        bottom = actor_to_world.new_tensor([0.0, 0.0, 0.0, 1.0])
        actor_to_world = torch.cat((actor_to_world, bottom[None]), dim=0)
    if actor_to_world.shape != (4, 4):
        raise ValueError("actor_to_world must have shape [3,4] or [4,4]")
    actor_to_world = actor_to_world.to(world_points)
    if not torch.isfinite(actor_to_world).all():
        raise ValueError("actor_to_world must be finite")
    if torch.abs(torch.linalg.det(actor_to_world)).detach() < 1.0e-12:
        raise ValueError("actor_to_world must be invertible")

    homogeneous = torch.cat(
        (world_points, torch.ones_like(world_points[:, :1])), dim=-1
    )
    world_to_actor = torch.linalg.inv(actor_to_world)
    local = torch.einsum("ij,nj->ni", world_to_actor, homogeneous)[:, :3]
    mask = torch.ones(
        world_points.shape[0], dtype=torch.bool, device=world_points.device
    )
    if box_dimensions is not None:
        if box_dimensions.shape != (3,):
            raise ValueError("box_dimensions must have shape [3]")
        box_dimensions = box_dimensions.to(world_points)
        if not torch.isfinite(box_dimensions).all() or torch.any(
            box_dimensions <= 0
        ):
            raise ValueError("box_dimensions must be finite and positive")
        mask = (local.abs() <= box_dimensions[None] / 2.0).all(dim=-1)
    return local[mask], mask


def load_colmap_points3d_text(path: str | Path) -> tuple[Tensor, Tensor]:
    """Load XYZ/RGB from COLMAP points3D.txt without a NumPy dependency."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    positions: list[list[float]] = []
    colors: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 7:
                raise ValueError(
                    f"invalid COLMAP points3D row at line {line_number}"
                )
            try:
                xyz = [float(value) for value in fields[1:4]]
                rgb = [int(value) for value in fields[4:7]]
            except ValueError as error:
                raise ValueError(
                    f"invalid COLMAP numeric value at line {line_number}"
                ) from error
            if any(channel < 0 or channel > 255 for channel in rgb):
                raise ValueError(
                    f"COLMAP RGB is out of range at line {line_number}"
                )
            positions.append(xyz)
            colors.append([channel / 255.0 for channel in rgb])
    if not positions:
        raise ValueError("COLMAP points3D file contains no points")
    points = torch.tensor(positions, dtype=torch.float32)
    rgb_tensor = torch.tensor(colors, dtype=torch.float32)
    if not torch.isfinite(points).all():
        raise ValueError("COLMAP points must be finite")
    return points, rgb_tensor


__all__ = [
    "GaussianInitializationConfig",
    "initialize_gaussians_from_points",
    "load_colmap_points3d_text",
    "voxel_downsample",
    "world_points_to_actor_local",
]
