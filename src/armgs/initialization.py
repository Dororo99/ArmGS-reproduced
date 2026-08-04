"""Point-cloud initialization utilities for ArmGS Gaussian scenes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .structures import GaussianSet

_SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class GaussianInitializationConfig:
    """Point-cloud initialization settings.

    ArmGS follows 3DGS for density control. The reference 3DGS initializer
    derives each isotropic scale from the mean squared distance to its three
    nearest neighbours. initial_scale is therefore used only for a degenerate
    one-point cloud, where no neighbour distance exists.
    """

    sh_degree: int = 3
    initial_opacity: float = 0.1
    initial_scale: float = 0.05
    voxel_size: float | None = None
    knn_neighbors: int = 3
    knn_chunk_size: int = 1024
    knn_backend: str = "auto"
    minimum_squared_distance: float = 1.0e-7

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
        if (
            isinstance(self.knn_neighbors, bool)
            or not isinstance(self.knn_neighbors, int)
            or self.knn_neighbors <= 0
        ):
            raise ValueError("knn_neighbors must be a positive integer")
        if (
            isinstance(self.knn_chunk_size, bool)
            or not isinstance(self.knn_chunk_size, int)
            or self.knn_chunk_size <= 0
        ):
            raise ValueError("knn_chunk_size must be a positive integer")
        if self.knn_backend not in {"auto", "scipy", "torch"}:
            raise ValueError(
                "knn_backend must be one of: auto, scipy, torch"
            )
        if not torch.isfinite(torch.tensor(self.minimum_squared_distance)):
            raise ValueError("minimum_squared_distance must be finite")
        if self.minimum_squared_distance <= 0:
            raise ValueError("minimum_squared_distance must be positive")


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


@torch.no_grad()
def estimate_knn_isotropic_scales(
    points: Tensor,
    *,
    neighbor_count: int = 3,
    chunk_size: int = 1024,
    backend: str = "auto",
    minimum_squared_distance: float = 1.0e-7,
    singleton_scale: float | None = None,
) -> Tensor:
    """Estimate reference-3DGS isotropic scales with bounded memory.

    For every point, this computes the square root of the mean squared
    distance to the nearest min(neighbor_count, N - 1) other points. The mean
    squared distance is clamped to minimum_squared_distance exactly like the
    official 3DGS initializer before it is square-rooted.

    The auto backend uses SciPy cKDTree for CPU point clouds when available,
    avoiding quadratic work for production-size inputs. Otherwise, query and
    reference points are processed in two-dimensional PyTorch chunks. That
    fallback remains an exact brute-force search, but its temporary storage is
    O(chunk_size**2) rather than O(N**2).

    The returned tensor contains positive scales with shape [N, 3]. The
    learnable scene module converts these values to log-scale parameters.
    """

    if points.ndim != 2 or points.shape[-1] != 3 or points.shape[0] == 0:
        raise ValueError("points must have non-empty shape [N,3]")
    if not points.is_floating_point() or not torch.isfinite(points).all():
        raise ValueError("points must be finite floating point")
    if (
        isinstance(neighbor_count, bool)
        or not isinstance(neighbor_count, int)
        or neighbor_count <= 0
    ):
        raise ValueError("neighbor_count must be a positive integer")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer")
    if backend not in {"auto", "scipy", "torch"}:
        raise ValueError("backend must be one of: auto, scipy, torch")
    minimum_squared_distance_tensor = torch.as_tensor(
        minimum_squared_distance, device=points.device, dtype=points.dtype
    )
    if (
        not torch.isfinite(minimum_squared_distance_tensor)
        or minimum_squared_distance_tensor <= 0
    ):
        raise ValueError("minimum_squared_distance must be finite and positive")

    point_count = points.shape[0]
    if point_count == 1:
        if singleton_scale is None:
            scale = minimum_squared_distance_tensor.sqrt()
        else:
            scale = torch.as_tensor(
                singleton_scale, device=points.device, dtype=points.dtype
            )
            if not torch.isfinite(scale) or scale <= 0:
                raise ValueError("singleton_scale must be finite and positive")
        return scale.expand(1, 3).clone()

    effective_neighbors = min(neighbor_count, point_count - 1)
    use_scipy = backend == "scipy" or (
        backend == "auto" and points.device.type == "cpu"
    )
    if use_scipy:
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            if backend == "scipy":
                raise ImportError(
                    "the scipy kNN backend requires scipy"
                ) from None
        else:
            numpy_points = (
                points.detach().to(device="cpu", dtype=torch.float64).numpy()
            )
            tree = cKDTree(numpy_points)
            try:
                distances, _ = tree.query(
                    numpy_points,
                    k=effective_neighbors + 1,
                    workers=-1,
                )
            except TypeError:  # pragma: no cover - old SciPy compatibility
                distances, _ = tree.query(
                    numpy_points,
                    k=effective_neighbors + 1,
                )
            mean_squared_distance = torch.from_numpy(
                (distances[:, 1:] ** 2).mean(axis=-1)
            ).to(device=points.device, dtype=points.dtype)
            mean_squared_distance.clamp_min_(minimum_squared_distance)
            scales = mean_squared_distance.sqrt()
            return scales[:, None].expand(-1, 3).clone()

    # Half precision matrix products are unavailable on some CPU builds and
    # lose too much precision for world-coordinate squared distances.
    distance_dtype = (
        torch.float32
        if points.dtype in (torch.float16, torch.bfloat16)
        else points.dtype
    )
    distance_points = points.detach().to(dtype=distance_dtype)
    mean_squared_distances: list[Tensor] = []

    for query_start in range(0, point_count, chunk_size):
        query_stop = min(query_start + chunk_size, point_count)
        query = distance_points[query_start:query_stop]
        best = torch.full(
            (query.shape[0], effective_neighbors),
            torch.inf,
            device=points.device,
            dtype=distance_dtype,
        )

        for reference_start in range(0, point_count, chunk_size):
            reference_stop = min(reference_start + chunk_size, point_count)
            reference = distance_points[reference_start:reference_stop]
            coordinate_differences = (
                query[:, None, :] - reference[None, :, :]
            )
            squared_distances = coordinate_differences.square().sum(dim=-1)

            overlap_start = max(query_start, reference_start)
            overlap_stop = min(query_stop, reference_stop)
            if overlap_start < overlap_stop:
                diagonal = torch.arange(
                    overlap_start,
                    overlap_stop,
                    device=points.device,
                )
                squared_distances[
                    diagonal - query_start, diagonal - reference_start
                ] = torch.inf

            candidates = torch.cat((best, squared_distances), dim=-1)
            best = torch.topk(
                candidates,
                k=effective_neighbors,
                dim=-1,
                largest=False,
                sorted=False,
            ).values

        mean_squared_distances.append(best.mean(dim=-1))

    mean_squared_distance = torch.cat(mean_squared_distances).clamp_min(
        float(minimum_squared_distance)
    )
    scales = mean_squared_distance.sqrt().to(dtype=points.dtype)
    return scales[:, None].expand(-1, 3).clone()


def merge_colored_point_clouds(
    point_clouds: Sequence[tuple[Tensor, Tensor]],
    *,
    voxel_size: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Merge same-frame colored point clouds and optionally voxel-deduplicate.

    This is intended for combining LiDAR and known-pose COLMAP/SfM points.
    Every input must already use the same coordinate system, device, and dtype;
    rejecting mismatches avoids silently merging an unaligned SfM model or
    incurring an unexpected device transfer. When voxel_size is supplied,
    positions and colors from both modalities are averaged per occupied voxel.
    """

    if not point_clouds:
        raise ValueError("at least one point cloud is required")
    for points, colors in point_clouds:
        if points.device != colors.device:
            raise ValueError(
                "points and colors within each cloud must share a device"
            )
        if points.dtype != colors.dtype:
            raise ValueError(
                "points and colors within each cloud must share a dtype"
            )
    validated = [
        _validated_points_and_colors(points, colors)
        for points, colors in point_clouds
    ]
    reference_points = validated[0][0]
    for points, colors in validated[1:]:
        if points.device != reference_points.device:
            raise ValueError("all point clouds must share a device")
        if points.dtype != reference_points.dtype:
            raise ValueError("all point clouds must share a dtype")
        if colors.device != reference_points.device:
            raise ValueError("all point clouds must share a device")
        if colors.dtype != reference_points.dtype:
            raise ValueError("all point clouds must share a dtype")

    merged_points = torch.cat([points for points, _ in validated], dim=0)
    merged_colors = torch.cat([colors for _, colors in validated], dim=0)
    if voxel_size is None:
        return merged_points, merged_colors
    return voxel_downsample(merged_points, merged_colors, voxel_size)


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
        scales=estimate_knn_isotropic_scales(
            points,
            neighbor_count=config.knn_neighbors,
            chunk_size=config.knn_chunk_size,
            backend=config.knn_backend,
            minimum_squared_distance=config.minimum_squared_distance,
            singleton_scale=config.initial_scale,
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
    "estimate_knn_isotropic_scales",
    "initialize_gaussians_from_points",
    "load_colmap_points3d_text",
    "merge_colored_point_clouds",
    "voxel_downsample",
    "world_points_to_actor_local",
]
