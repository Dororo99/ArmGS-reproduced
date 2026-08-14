"""Point-cloud initialization utilities for ArmGS Gaussian scenes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .structures import GaussianSet

_SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class StreetGSBackgroundPreprocessingResult:
    """StreetGS-compatible world-space background initialization data."""

    points: Tensor
    colors: Tensor
    lidar_point_count: int
    sfm_input_point_count: int
    sfm_retained_point_count: int
    lidar_aabb_center: Tensor
    lidar_aabb_half_diagonal: Tensor


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


@torch.no_grad()
def preprocess_streetgs_waymo_background(
    lidar_points: Tensor,
    lidar_colors: Tensor,
    sfm_points: Tensor | None = None,
    sfm_colors: Tensor | None = None,
    *,
    camera_centers: Tensor | None = None,
    voxel_size: float = 0.15,
    radius_outlier_nb_points: int = 10,
    radius_outlier_radius: float = 0.5,
    sfm_extent_multiplier: float = 2.0,
    filter_sfm_near_or_below_cameras: bool = False,
    camera_extent: float = 20.0,
) -> StreetGSBackgroundPreprocessingResult:
    """Prepare the Waymo background in the official StreetGS order.

    The LiDAR cloud is first processed by Open3D's color-aware
    ``voxel_down_sample`` and ``remove_radius_outlier`` operations. Its AABB
    center and half-diagonal define the SfM acceptance sphere. SfM points
    strictly inside ``sfm_extent_multiplier * half_diagonal`` are appended to
    the filtered LiDAR cloud without another voxel operation.

    The optional camera filter reproduces StreetGS's ``filter_colmap`` branch:
    it removes an SfM point when it is within ``camera_extent`` of any camera
    or below any camera along Waymo's world Z axis. Official Waymo validation
    configs leave this branch disabled, hence the ``False`` default.

    Open3D is imported lazily so it remains a Waymo preprocessing dependency,
    rather than a requirement for importing the ArmGS core package.
    """

    def _require_positive(value: float, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be finite and positive")
        try:
            converted = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{name} must be finite and positive"
            ) from None
        if not math.isfinite(converted) or converted <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return converted

    if lidar_points.device != lidar_colors.device:
        raise ValueError("LiDAR points and colors must share a device")
    if lidar_points.dtype != lidar_colors.dtype:
        raise ValueError("LiDAR points and colors must share a dtype")
    lidar_points, lidar_colors = _validated_points_and_colors(
        lidar_points, lidar_colors
    )
    voxel_size = _require_positive(voxel_size, "voxel_size")
    radius_outlier_radius = _require_positive(
        radius_outlier_radius, "radius_outlier_radius"
    )
    sfm_extent_multiplier = _require_positive(
        sfm_extent_multiplier, "sfm_extent_multiplier"
    )
    camera_extent = _require_positive(camera_extent, "camera_extent")
    if (
        isinstance(radius_outlier_nb_points, bool)
        or not isinstance(radius_outlier_nb_points, int)
        or radius_outlier_nb_points <= 0
    ):
        raise ValueError("radius_outlier_nb_points must be a positive integer")
    if not isinstance(filter_sfm_near_or_below_cameras, bool):
        raise ValueError(
            "filter_sfm_near_or_below_cameras must be boolean"
        )
    if (sfm_points is None) != (sfm_colors is None):
        raise ValueError("sfm_points and sfm_colors must be provided together")

    try:
        import numpy as np
        import open3d as o3d
    except ImportError as error:
        raise ImportError(
            "StreetGS Waymo background preprocessing requires Open3D"
        ) from error

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(
        lidar_points.detach().to(device="cpu", dtype=torch.float64).numpy()
    )
    point_cloud.colors = o3d.utility.Vector3dVector(
        lidar_colors.detach().to(device="cpu", dtype=torch.float64).numpy()
    )
    point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    point_cloud, _ = point_cloud.remove_radius_outlier(
        nb_points=radius_outlier_nb_points,
        radius=radius_outlier_radius,
    )
    filtered_lidar_numpy = np.asarray(point_cloud.points)
    filtered_colors_numpy = np.asarray(point_cloud.colors)
    if filtered_lidar_numpy.shape != filtered_colors_numpy.shape:
        raise RuntimeError(
            "Open3D returned mismatched LiDAR point and color arrays"
        )
    if (
        filtered_lidar_numpy.ndim != 2
        or filtered_lidar_numpy.shape[1:] != (3,)
        or filtered_lidar_numpy.shape[0] == 0
    ):
        raise ValueError(
            "StreetGS LiDAR preprocessing removed every background point"
        )

    # Copy the Open3D-owned buffers before its point cloud goes out of scope.
    filtered_lidar = torch.from_numpy(
        np.array(filtered_lidar_numpy, copy=True)
    ).to(device=lidar_points.device, dtype=lidar_points.dtype)
    filtered_colors = torch.from_numpy(
        np.array(filtered_colors_numpy, copy=True)
    ).to(device=lidar_colors.device, dtype=lidar_colors.dtype)
    filtered_lidar, filtered_colors = _validated_points_and_colors(
        filtered_lidar, filtered_colors
    )

    minimum = filtered_lidar.amin(dim=0)
    maximum = filtered_lidar.amax(dim=0)
    lidar_aabb_center = (minimum + maximum) * 0.5
    lidar_aabb_half_diagonal = torch.linalg.vector_norm(
        maximum - minimum
    ) * 0.5

    sfm_input_point_count = 0
    sfm_retained_point_count = 0
    merged_points = filtered_lidar
    merged_colors = filtered_colors
    if sfm_points is not None and sfm_colors is not None:
        sfm_points, sfm_colors = _validated_points_and_colors(
            sfm_points, sfm_colors
        )
        sfm_points = sfm_points.to(filtered_lidar)
        sfm_colors = sfm_colors.to(filtered_colors)
        sfm_input_point_count = sfm_points.shape[0]
        keep = torch.linalg.vector_norm(
            sfm_points - lidar_aabb_center[None], dim=-1
        ) < sfm_extent_multiplier * lidar_aabb_half_diagonal

        if filter_sfm_near_or_below_cameras:
            if camera_centers is None:
                raise ValueError(
                    "camera_centers are required when the camera SfM filter "
                    "is enabled"
                )
            if (
                camera_centers.ndim != 2
                or camera_centers.shape[-1] != 3
                or camera_centers.shape[0] == 0
                or not torch.isfinite(camera_centers).all()
            ):
                raise ValueError(
                    "camera_centers must have finite, non-empty shape [C,3]"
                )
            camera_centers = camera_centers.to(sfm_points)
            distances = torch.linalg.vector_norm(
                sfm_points[:, None] - camera_centers[None], dim=-1
            )
            near_any_camera = (distances < camera_extent).any(dim=-1)
            below_any_camera = (
                sfm_points[:, None, 2] < camera_centers[None, :, 2]
            ).any(dim=-1)
            keep &= ~(near_any_camera | below_any_camera)

        retained_sfm_points = sfm_points[keep]
        retained_sfm_colors = sfm_colors[keep]
        sfm_retained_point_count = retained_sfm_points.shape[0]
        # StreetGS concatenates the modalities here; there is no second voxel.
        merged_points = torch.cat((filtered_lidar, retained_sfm_points), dim=0)
        merged_colors = torch.cat((filtered_colors, retained_sfm_colors), dim=0)

    return StreetGSBackgroundPreprocessingResult(
        points=merged_points,
        colors=merged_colors,
        lidar_point_count=filtered_lidar.shape[0],
        sfm_input_point_count=sfm_input_point_count,
        sfm_retained_point_count=sfm_retained_point_count,
        lidar_aabb_center=lidar_aabb_center,
        lidar_aabb_half_diagonal=lidar_aabb_half_diagonal,
    )


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
    "StreetGSBackgroundPreprocessingResult",
    "estimate_knn_isotropic_scales",
    "initialize_gaussians_from_points",
    "load_colmap_points3d_text",
    "merge_colored_point_clouds",
    "preprocess_streetgs_waymo_background",
    "voxel_downsample",
    "world_points_to_actor_local",
]
