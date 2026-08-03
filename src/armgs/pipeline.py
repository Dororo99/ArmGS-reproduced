"""Paper-ordered ArmGS composite rendering pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .compositing import composite_sky
from .model import ArmGSCore
from .scene import CompositeGaussianScene
from .spherical_harmonics import (
    camera_to_gaussian_directions,
    spherical_harmonics_to_rgb,
)
from .structures import (
    GaussianRasterizer,
    GaussianSet,
    RasterizationInput,
    RasterizationOutput,
)

CameraConvention = Literal["opencv", "opengl"]


@dataclass(frozen=True)
class CameraView:
    """One image/view and its unambiguous geometry/appearance metadata."""

    camera_to_world: Tensor
    intrinsics: Tensor
    image_size: tuple[int, int]
    timestamp: Tensor
    camera_id: Tensor
    training_row: Tensor | None = None
    camera_convention: CameraConvention = "opencv"
    view_direction: Tensor | None = None
    sky_directions: Tensor | None = None
    visible_indices: Tensor | None = None
    gaussian_velocities: Tensor | None = None
    camera_linear_velocity: Tensor | None = None
    camera_angular_velocity: Tensor | None = None
    rolling_shutter_time: Tensor | None = None
    rolling_shutter_direction: int = 1


@dataclass(frozen=True)
class ArmGSRenderOutput:
    """Final image plus the intermediates required by equation (9)."""

    rgb: Tensor
    foreground_rgb: Tensor
    sky_rgb: Tensor | None
    depth: Tensor
    non_sky_accumulated_alpha: Tensor
    actor_alpha: Tensor | None
    rasterization: RasterizationOutput
    composite_gaussians: GaussianSet
    refined_colors: Tensor
    visible_indices: Tensor | None


def _homogeneous_camera_to_world(camera_to_world: Tensor) -> Tensor:
    if camera_to_world.shape == (4, 4):
        return camera_to_world
    if camera_to_world.shape == (3, 4):
        row = camera_to_world.new_tensor([0.0, 0.0, 0.0, 1.0])
        return torch.cat((camera_to_world, row[None]), dim=0)
    raise ValueError("camera_to_world must have shape [3,4] or [4,4]")


def camera_center_and_forward(
    camera_to_world: Tensor, convention: CameraConvention
) -> tuple[Tensor, Tensor]:
    """Return world camera center and optical forward direction."""

    transform = _homogeneous_camera_to_world(camera_to_world)
    if convention == "opencv":
        forward = transform[:3, 2]
    elif convention == "opengl":
        forward = -transform[:3, 2]
    else:
        raise ValueError("camera_convention must be 'opencv' or 'opengl'")
    return transform[:3, 3], F.normalize(forward, dim=-1, eps=1.0e-8)


def conservative_frustum_visible_indices(
    gaussians: GaussianSet,
    camera_to_world: Tensor,
    intrinsics: Tensor,
    image_size: tuple[int, int],
    *,
    convention: CameraConvention,
    near_plane: float = 0.01,
    far_plane: float = 1.0e10,
    sigma_extent: float = 3.0,
    eps2d: float = 0.3,
) -> Tensor:
    """Select Gaussian bounding spheres intersecting the padded view frustum.

    A max-scale 3-D sphere conservatively contains every rotated anisotropic
    Gaussian at the requested sigma extent. Frustum-plane sphere tests avoid
    the false negatives of a center projection approximation. The image bounds
    are additionally padded by sigma_extent * sqrt(eps2d) pixels to match the
    rasterizer's screen-space covariance dilation.
    """

    if near_plane < 0 or far_plane <= near_plane:
        raise ValueError("planes must satisfy 0 <= near < far")
    if sigma_extent <= 0:
        raise ValueError("sigma_extent must be positive")
    if eps2d < 0:
        raise ValueError("eps2d cannot be negative")
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    transform = _homogeneous_camera_to_world(camera_to_world).to(gaussians.means)
    intrinsics = intrinsics.to(gaussians.means)
    if convention == "opengl":
        axis_flip = torch.diag(
            transform.new_tensor([1.0, -1.0, -1.0, 1.0])
        )
        transform = transform @ axis_flip
    elif convention != "opencv":
        raise ValueError("camera_convention must be 'opencv' or 'opengl'")
    world_to_camera = torch.linalg.inv(transform)
    homogeneous_means = torch.cat(
        (gaussians.means, torch.ones_like(gaussians.means[:, :1])), dim=-1
    )
    camera_means = torch.einsum(
        "ij,nj->ni", world_to_camera, homogeneous_means
    )[:, :3]
    radius = sigma_extent * gaussians.scales.max(dim=-1).values
    depth = camera_means[:, 2]

    padding = sigma_extent * math.sqrt(eps2d)
    row_x, row_y, row_z = intrinsics
    planes = torch.stack(
        (
            row_x + padding * row_z,
            (width + padding) * row_z - row_x,
            row_y + padding * row_z,
            (height + padding) * row_z - row_y,
        ),
        dim=0,
    )
    signed_numerators = torch.einsum("pd,nd->np", planes, camera_means)
    sphere_margin = radius[:, None] * torch.linalg.vector_norm(planes, dim=-1)
    intersects_sides = (signed_numerators + sphere_margin >= 0).all(dim=-1)
    visible = (
        (depth + radius > near_plane)
        & (depth - radius < far_plane)
        & intersects_sides
    )
    return torch.nonzero(visible, as_tuple=False).squeeze(1)


def world_ray_directions(
    camera_to_world: Tensor,
    intrinsics: Tensor,
    image_size: tuple[int, int],
    *,
    convention: CameraConvention,
) -> Tensor:
    """Generate normalized world-space pixel-center directions.

    Intrinsics always use the standard image convention (x right, y down).
    OpenCV camera coordinates use +Z forward; OpenGL coordinates use -Z
    forward and +Y up. This conversion is also mirrored by the gsplat adapter.
    """

    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must have shape [3,3]")
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    transform = _homogeneous_camera_to_world(camera_to_world)
    intrinsics = intrinsics.to(transform)
    if not torch.isfinite(intrinsics).all():
        raise ValueError("intrinsics must be finite")
    if torch.abs(torch.linalg.det(intrinsics)).detach() < 1.0e-12:
        raise ValueError("intrinsics must be invertible")

    ys, xs = torch.meshgrid(
        torch.arange(height, device=transform.device, dtype=transform.dtype) + 0.5,
        torch.arange(width, device=transform.device, dtype=transform.dtype) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack((xs, ys, torch.ones_like(xs)), dim=-1)
    camera_directions = torch.einsum(
        "ij,hwj->hwi", torch.linalg.inv(intrinsics), pixels
    )
    if convention == "opengl":
        axis_flip = camera_directions.new_tensor([1.0, -1.0, -1.0])
        camera_directions = camera_directions * axis_flip
    elif convention != "opencv":
        raise ValueError("camera_convention must be 'opencv' or 'opengl'")
    world_directions = torch.einsum(
        "ij,hwj->hwi", transform[:3, :3], camera_directions
    )
    return F.normalize(world_directions, dim=-1, eps=1.0e-8)


class ArmGSCompositeRenderer(nn.Module):
    """Execute actor, local, raster, sky, then global refinement in paper order."""

    def __init__(
        self,
        core: ArmGSCore,
        scene: CompositeGaussianScene,
        rasterizer: GaussianRasterizer,
        *,
        auto_frustum_cull: bool = True,
        cull_near_plane: float | None = None,
        cull_far_plane: float | None = None,
        cull_sigma_extent: float = 3.0,
        cull_eps2d: float | None = None,
    ) -> None:
        super().__init__()
        self.core = core
        self.scene = scene
        self.rasterizer = rasterizer
        self.auto_frustum_cull = bool(auto_frustum_cull)
        backend_config = getattr(rasterizer, "config", None)

        def resolve(value: float | None, name: str, fallback: float) -> float:
            if value is not None:
                return float(value)
            return float(getattr(backend_config, name, fallback))

        self.cull_near_plane = resolve(cull_near_plane, "near_plane", 0.01)
        self.cull_far_plane = resolve(cull_far_plane, "far_plane", 1.0e10)
        self.cull_sigma_extent = float(cull_sigma_extent)
        self.cull_eps2d = resolve(cull_eps2d, "eps2d", 0.3)

    def _frame_embedding(self, view: CameraView) -> Tensor:
        if view.training_row is not None:
            row = torch.as_tensor(view.training_row)
            if row.numel() != 1:
                raise ValueError("training_row must be scalar")
            return self.core.training_frame_embedding(row.reshape(()))
        camera_id = torch.as_tensor(view.camera_id)
        timestamp = torch.as_tensor(view.timestamp)
        if camera_id.numel() != 1 or timestamp.numel() != 1:
            raise ValueError("camera_id and timestamp must be scalar")
        return self.core.novel_view_embedding(
            camera_id.reshape(()), timestamp.reshape(())
        )

    @staticmethod
    def _validate_visible_indices(
        visible_indices: Tensor | None, count: int, device: torch.device
    ) -> Tensor | None:
        if visible_indices is None:
            return None
        indices = visible_indices.to(device=device, dtype=torch.long)
        if indices.ndim != 1:
            raise ValueError("visible_indices must be one-dimensional")
        if indices.numel() and (
            indices.min().detach() < 0 or indices.max().detach() >= count
        ):
            raise ValueError("visible_indices are out of range")
        if indices.unique().numel() != indices.numel():
            raise ValueError("visible_indices must not contain duplicates")
        return indices

    def _refined_colors(
        self,
        gaussians: GaussianSet,
        camera_center: Tensor,
        frame_embedding: Tensor,
        visible_indices: Tensor | None,
    ) -> Tensor:
        directions = camera_to_gaussian_directions(
            gaussians.means, camera_center
        )
        base_colors = spherical_harmonics_to_rgb(
            gaussians.sh_coefficients, directions, gaussians.sh_degree
        )
        indices = self._validate_visible_indices(
            visible_indices, gaussians.count, gaussians.means.device
        )
        if indices is None:
            return self.core.local_refiner(
                gaussians.means, base_colors, frame_embedding
            )
        if indices.numel() == 0:
            return base_colors
        refined_visible = self.core.local_refiner(
            gaussians.means.index_select(0, indices),
            base_colors.index_select(0, indices),
            frame_embedding,
        )
        return base_colors.index_copy(0, indices, refined_visible)

    def forward(self, view: CameraView) -> ArmGSRenderOutput:
        timestamp = torch.as_tensor(view.timestamp)
        if timestamp.numel() != 1:
            raise ValueError("timestamp must be scalar")
        active_actor_count = sum(
            actor.is_active(timestamp) for actor in self.scene.actors
        )
        gaussians = self.scene.gaussians_at(self.core, timestamp.reshape(()))
        camera_to_world = _homogeneous_camera_to_world(
            view.camera_to_world.to(gaussians.means)
        )
        intrinsics = view.intrinsics.to(gaussians.means)
        camera_center, inferred_forward = camera_center_and_forward(
            camera_to_world, view.camera_convention
        )
        if view.view_direction is None:
            view_direction = inferred_forward
        else:
            if view.view_direction.shape != (3,):
                raise ValueError("view_direction must have shape [3]")
            view_direction = F.normalize(
                view.view_direction.to(gaussians.means), dim=-1, eps=1.0e-8
            )
        frame_embedding = self._frame_embedding(view)
        visible_indices = view.visible_indices
        rolling_shutter_active = False
        if view.rolling_shutter_time is not None:
            shutter_values = torch.as_tensor(view.rolling_shutter_time).detach()
            if not torch.isfinite(shutter_values).all():
                raise ValueError("rolling_shutter_time must be finite")
            if torch.any(shutter_values < 0):
                raise ValueError("rolling_shutter_time cannot be negative")
            rolling_shutter_active = bool(torch.any(shutter_values > 0).item())
        if visible_indices is None and self.auto_frustum_cull and not rolling_shutter_active:
            visible_indices = conservative_frustum_visible_indices(
                gaussians,
                camera_to_world,
                intrinsics,
                view.image_size,
                convention=view.camera_convention,
                near_plane=self.cull_near_plane,
                far_plane=self.cull_far_plane,
                sigma_extent=self.cull_sigma_extent,
                eps2d=self.cull_eps2d,
            )
        colors = self._refined_colors(
            gaussians,
            camera_center,
            frame_embedding,
            visible_indices,
        )

        rasterization = self.rasterizer(
            RasterizationInput(
                means=gaussians.means,
                quaternions=gaussians.quaternions,
                scales=gaussians.scales,
                opacities=gaussians.opacities,
                colors=colors,
                camera_to_world=camera_to_world,
                intrinsics=intrinsics,
                image_size=view.image_size,
                group_ids=gaussians.group_ids,
                camera_convention=view.camera_convention,
                velocities=view.gaussian_velocities,
                camera_linear_velocity=view.camera_linear_velocity,
                camera_angular_velocity=view.camera_angular_velocity,
                rolling_shutter_time=view.rolling_shutter_time,
                rolling_shutter_direction=view.rolling_shutter_direction,
            )
        )
        foreground_rgb = rasterization.rgb
        sky_rgb: Tensor | None = None
        composited_rgb = foreground_rgb
        if self.scene.sky is not None:
            if view.sky_directions is None:
                directions = world_ray_directions(
                    camera_to_world,
                    intrinsics,
                    view.image_size,
                    convention=view.camera_convention,
                ).unsqueeze(0)
            else:
                directions = view.sky_directions.to(gaussians.means)
                if directions.shape == (*view.image_size, 3):
                    directions = directions.unsqueeze(0)
                expected = (1, *view.image_size, 3)
                if directions.shape != expected:
                    raise ValueError(
                        f"sky_directions must have shape {expected} or {expected[1:]}"
                    )
                norms = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
                if torch.any(norms.detach() < 1.0e-8):
                    raise ValueError("sky_directions cannot contain zero vectors")
                directions = directions / norms
            sky_rgb = self.scene.sky(directions)
            composited_rgb = composite_sky(
                foreground_rgb,
                rasterization.accumulated_alpha,
                sky_rgb,
            )

        actor_alpha = rasterization.actor_alpha
        if (
            actor_alpha is None
            and len(self.scene.actors) > 0
            and active_actor_count == 0
        ):
            # Frames outside every actor track still need an explicit zero map
            # when the strict foreground objective is enabled.
            actor_alpha = torch.zeros_like(rasterization.accumulated_alpha)

        final_rgb = self.core.refine_image(
            composited_rgb,
            frame_embedding,
            camera_center,
            view_direction,
        )
        return ArmGSRenderOutput(
            rgb=final_rgb,
            foreground_rgb=foreground_rgb,
            sky_rgb=sky_rgb,
            depth=rasterization.depth,
            non_sky_accumulated_alpha=rasterization.accumulated_alpha,
            actor_alpha=actor_alpha,
            rasterization=rasterization,
            composite_gaussians=gaussians,
            refined_colors=colors,
            visible_indices=visible_indices,
        )


__all__ = [
    "ArmGSCompositeRenderer",
    "ArmGSRenderOutput",
    "CameraView",
    "camera_center_and_forward",
    "conservative_frustum_visible_indices",
    "world_ray_directions",
]
