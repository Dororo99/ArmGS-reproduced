"""Adapter for gsplat's differentiable CUDA rasterizer."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch
from torch import Tensor

from ..structures import RasterizationInput, RasterizationOutput


@dataclass(frozen=True)
class GsplatRasterizerConfig:
    near_plane: float = 0.01
    far_plane: float = 1.0e10
    radius_clip: float = 0.0
    eps2d: float = 0.3
    packed: bool = True
    tile_size: int = 16
    sparse_grad: bool = False
    absgrad: bool = False
    rasterize_mode: str = "classic"
    channel_chunk: int = 32
    actor_alpha_mode: str = "aggregate"
    depth_mode: str = "ED"


class GsplatRasterizer:
    """Rasterize precomputed ArmGS RGB and optional actor-group alpha.

    Group ids use negative values for background and non-negative values for
    actors. The default ``aggregate`` mode adds one indicator channel for all
    actors, keeping memory O(HW) rather than O(HWA). ``per_group`` remains an
    explicit diagnostic mode for small actor counts.
    """

    def __init__(self, config: GsplatRasterizerConfig | None = None) -> None:
        self.config = config or GsplatRasterizerConfig()
        if self.config.actor_alpha_mode not in {"aggregate", "per_group", "none"}:
            raise ValueError(
                "actor_alpha_mode must be aggregate, per_group, or none"
            )
        if self.config.depth_mode not in {"D", "ED"}:
            raise ValueError("depth_mode must be D or ED")

    @staticmethod
    def _camera_batch(inputs: RasterizationInput) -> tuple[Tensor, Tensor]:
        camera_to_world = inputs.camera_to_world
        intrinsics = inputs.intrinsics
        if camera_to_world.ndim == 2:
            camera_to_world = camera_to_world.unsqueeze(0)
        if camera_to_world.ndim != 3:
            raise ValueError("camera_to_world must be a matrix or batch of matrices")
        if camera_to_world.shape[-2:] == (3, 4):
            homogeneous_row = camera_to_world.new_tensor([0.0, 0.0, 0.0, 1.0])
            homogeneous_row = homogeneous_row.reshape(1, 1, 4).expand(
                camera_to_world.shape[0], -1, -1
            )
            camera_to_world = torch.cat((camera_to_world, homogeneous_row), dim=-2)
        elif camera_to_world.shape[-2:] != (4, 4):
            raise ValueError("camera_to_world matrices must have shape [3,4] or [4,4]")
        if inputs.camera_convention == "opengl":
            # gsplat view matrices use OpenCV camera axes. Convert an OpenGL
            # camera-to-world transform by flipping its Y and Z basis axes.
            axis_flip = torch.diag(
                camera_to_world.new_tensor([1.0, -1.0, -1.0, 1.0])
            )
            camera_to_world = camera_to_world @ axis_flip
        elif inputs.camera_convention != "opencv":
            raise ValueError("camera_convention must be opencv or opengl")
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0)
        if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
            raise ValueError("intrinsics must have shape [3,3] or [B,3,3]")
        if camera_to_world.shape[0] != intrinsics.shape[0]:
            raise ValueError("camera and intrinsics batch sizes must match")
        return camera_to_world, intrinsics

    def _features_and_groups(
        self, inputs: RasterizationInput, camera_count: int
    ) -> tuple[Tensor, Tensor | None, bool]:
        colors = inputs.colors
        gaussian_count = inputs.means.shape[0]
        if colors.ndim == 2:
            if colors.shape != (gaussian_count, 3) or camera_count != 1:
                raise ValueError(
                    "colors must have shape [N,3] only for a single camera"
                )
        elif colors.ndim == 3:
            if colors.shape != (camera_count, gaussian_count, 3):
                raise ValueError("batched colors must have shape [B,N,3]")
        else:
            raise ValueError("colors must have shape [N,3] or [B,N,3]")
        if inputs.group_ids is None or self.config.actor_alpha_mode == "none":
            return colors, None, False
        if inputs.group_ids.shape != (gaussian_count,):
            raise ValueError("group_ids must have shape [N]")
        group_ids = inputs.group_ids.to(device=colors.device, dtype=torch.long)
        actor_mask = group_ids >= 0
        if not actor_mask.any():
            return colors, None, False

        if self.config.actor_alpha_mode == "aggregate":
            indicators = actor_mask[:, None].to(colors)
            group_labels = None
        else:
            group_labels = torch.unique(group_ids[actor_mask], sorted=True)
            indicators = (group_ids[:, None] == group_labels[None, :]).to(colors)
        if colors.ndim == 3:
            indicators = indicators.unsqueeze(0).expand(camera_count, -1, -1)
        return torch.cat((colors, indicators), dim=-1), group_labels, True

    def __call__(self, inputs: RasterizationInput) -> RasterizationOutput:
        try:
            import gsplat
        except ImportError as error:
            raise ImportError(
                "GsplatRasterizer requires gsplat; install the integration extra "
                "or use /venv/camosplat on the provided instance"
            ) from error

        count = inputs.means.shape[0]
        if inputs.means.shape != (count, 3):
            raise ValueError("means must have shape [N,3]")
        if inputs.quaternions.shape != (count, 4):
            raise ValueError("quaternions must have shape [N,4]")
        if inputs.scales.shape != (count, 3):
            raise ValueError("scales must have shape [N,3]")
        if inputs.opacities.shape != (count, 1):
            raise ValueError("opacities must have shape [N,1]")
        height, width = inputs.image_size
        if height <= 0 or width <= 0:
            raise ValueError("image dimensions must be positive")

        camera_to_world, intrinsics = self._camera_batch(inputs)
        camera_to_world = camera_to_world.to(inputs.means)
        intrinsics = intrinsics.to(inputs.means)
        view_matrices = torch.linalg.inv(camera_to_world)
        features, group_labels, has_actor_features = self._features_and_groups(
            inputs, camera_to_world.shape[0]
        )

        rasterization = gsplat.rasterization
        arguments = {
            "means": inputs.means,
            "quats": inputs.quaternions,
            "scales": inputs.scales,
            "opacities": inputs.opacities[:, 0],
            "colors": features,
            "viewmats": view_matrices,
            "Ks": intrinsics,
            "width": width,
            "height": height,
            "near_plane": self.config.near_plane,
            "far_plane": self.config.far_plane,
            "radius_clip": self.config.radius_clip,
            "eps2d": self.config.eps2d,
            "sh_degree": None,
            "packed": self.config.packed,
            "tile_size": self.config.tile_size,
            "render_mode": f"RGB+{self.config.depth_mode}",
            "sparse_grad": self.config.sparse_grad,
            "absgrad": self.config.absgrad,
            "rasterize_mode": self.config.rasterize_mode,
            "channel_chunk": self.config.channel_chunk,
        }
        # CamoSplat extends upstream gsplat with rolling-shutter metadata.
        # Forward it when supported, reject silent loss when a caller supplies it
        # to an upstream backend, and still satisfy a required velocities=None.
        signature = inspect.signature(rasterization).parameters
        velocities = inputs.velocities
        if velocities is not None:
            if velocities.shape != (count, 3):
                raise ValueError("velocities must have shape [N,3]")
            velocities = velocities.to(inputs.means)
        if "velocities" in signature:
            arguments["velocities"] = velocities
        elif velocities is not None:
            raise ValueError("installed gsplat does not support Gaussian velocities")

        camera_count = camera_to_world.shape[0]

        def camera_vectors(value: Tensor | None, name: str) -> Tensor | None:
            if value is None:
                return None
            value = value.to(inputs.means)
            if value.shape == (3,) and camera_count == 1:
                value = value.unsqueeze(0)
            if value.shape != (camera_count, 3):
                raise ValueError(f"{name} must have shape [B,3]")
            return value

        rolling_values = {
            "linear_velocity": camera_vectors(
                inputs.camera_linear_velocity, "camera_linear_velocity"
            ),
            "angular_velocity": camera_vectors(
                inputs.camera_angular_velocity, "camera_angular_velocity"
            ),
        }
        shutter_time = inputs.rolling_shutter_time
        if shutter_time is not None:
            shutter_time = shutter_time.to(inputs.means)
            if not torch.isfinite(shutter_time).all():
                raise ValueError("rolling_shutter_time must be finite")
            if torch.any(shutter_time < 0):
                raise ValueError("rolling_shutter_time cannot be negative")
            if shutter_time.ndim == 0 and camera_count == 1:
                shutter_time = shutter_time.reshape(1)
            if shutter_time.shape != (camera_count,):
                raise ValueError("rolling_shutter_time must have shape [B]")
        rolling_values["rolling_shutter_time"] = shutter_time
        for name, value in rolling_values.items():
            if name in signature:
                arguments[name] = value
            elif value is not None:
                raise ValueError(f"installed gsplat does not support {name}")
        if not 1 <= inputs.rolling_shutter_direction <= 5:
            raise ValueError("rolling_shutter_direction must be in [1,5]")
        if shutter_time is not None:
            if "rolling_shutter_direction" not in signature:
                raise ValueError(
                    "installed gsplat does not support rolling_shutter_direction"
                )
            arguments["rolling_shutter_direction"] = (
                inputs.rolling_shutter_direction
            )

        rendered, accumulated_alpha, metadata = rasterization(**arguments)
        feature_count = features.shape[-1]
        rendered_features = rendered[..., :feature_count]
        rendered_depth = rendered[..., feature_count : feature_count + 1]
        rgb = rendered_features[..., :3]
        actor_features = (
            rendered_features[..., 3:] if has_actor_features else None
        )
        if actor_features is not None and self.config.actor_alpha_mode == "per_group":
            group_alpha = actor_features
            actor_alpha = actor_features.sum(dim=-1, keepdim=True)
        else:
            group_alpha = None
            actor_alpha = actor_features
        metadata = dict(metadata)
        metadata["armgs_actor_alpha_mode"] = self.config.actor_alpha_mode
        metadata["armgs_depth_mode"] = self.config.depth_mode
        return RasterizationOutput(
            rgb=rgb,
            depth=rendered_depth,
            accumulated_alpha=accumulated_alpha,
            actor_alpha=actor_alpha,
            group_alpha=group_alpha,
            group_labels=group_labels,
            metadata=metadata,
        )
