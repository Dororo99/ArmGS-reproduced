"""Differentiable explicit cubemap sky representation for ArmGS."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ExplicitCubemapSky(nn.Module):
    """Learnable RGB cubemap sampled by normalized world-space directions.

    Faces are stored channel-last in ``(+X, -X, +Y, -Y, +Z, -Z)`` order with
    shape ``[6, height, width, 3]``. Inputs may have any leading dimensions but
    must have shape ``[..., 3]`` and contain finite unit-length directions.

    The per-face coordinates ``(u, v)`` below follow ``grid_sample`` convention:
    ``(-1, -1)`` is the top-left texel and ``(1, 1)`` is the bottom-right texel.
    Cubemap seams use deterministic dominant-axis selection; sampling within a
    face is bilinear and differentiable with respect to both texels and direction.
    """

    FACE_NAMES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")

    def __init__(
        self,
        resolution: int | tuple[int, int] = 16,
        *,
        initial_color: Sequence[float] = (0.5, 0.5, 0.5),
        normalization_tolerance: float = 1.0e-4,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if isinstance(resolution, int):
            height = width = resolution
        else:
            if len(resolution) != 2:
                raise ValueError("resolution must be an int or (height, width)")
            height, width = resolution
        if height <= 0 or width <= 0:
            raise ValueError("cubemap height and width must be positive")
        if normalization_tolerance <= 0.0:
            raise ValueError("normalization_tolerance must be positive")

        color = torch.as_tensor(initial_color, device=device, dtype=dtype)
        if color.shape != (3,):
            raise ValueError("initial_color must contain exactly three RGB values")
        if not color.is_floating_point():
            color = color.to(torch.get_default_dtype())
        if not torch.isfinite(color).all():
            raise ValueError("initial_color must contain only finite values")

        texels = color.reshape(1, 1, 1, 3).expand(6, height, width, 3).clone()
        self.cubemap = nn.Parameter(texels)
        self.normalization_tolerance = float(normalization_tolerance)

    @property
    def resolution(self) -> tuple[int, int]:
        return self.cubemap.shape[1], self.cubemap.shape[2]

    def _validate_directions(self, directions: Tensor) -> None:
        if directions.ndim < 1 or directions.shape[-1] != 3:
            raise ValueError("directions must have channel-last shape [..., 3]")
        if not directions.is_floating_point():
            raise TypeError("directions must be a floating-point tensor")
        if directions.device != self.cubemap.device:
            raise ValueError("directions and cubemap must be on the same device")
        if directions.dtype != self.cubemap.dtype:
            raise ValueError("directions and cubemap must have the same dtype")
        if directions.numel() == 0:
            return
        detached = directions.detach()
        if not torch.isfinite(detached).all():
            raise ValueError("directions must contain only finite values")
        norms = torch.linalg.vector_norm(detached, dim=-1)
        if not torch.allclose(
            norms,
            torch.ones_like(norms),
            atol=self.normalization_tolerance,
            rtol=self.normalization_tolerance,
        ):
            raise ValueError("directions must be normalized to unit length")

    @staticmethod
    def _face_indices_and_uv(flat_directions: Tensor) -> tuple[Tensor, Tensor]:
        x, y, z = flat_directions.unbind(dim=-1)
        absolute = flat_directions.abs()
        dominant_axis = absolute.argmax(dim=-1)
        dominant_magnitude = absolute.gather(1, dominant_axis[:, None]).squeeze(1)

        positive_x = (dominant_axis == 0) & (x >= 0)
        negative_x = (dominant_axis == 0) & (x < 0)
        positive_y = (dominant_axis == 1) & (y >= 0)
        negative_y = (dominant_axis == 1) & (y < 0)
        positive_z = (dominant_axis == 2) & (z >= 0)

        face_indices = torch.empty_like(dominant_axis)
        face_indices[positive_x] = 0
        face_indices[negative_x] = 1
        face_indices[positive_y] = 2
        face_indices[negative_y] = 3
        face_indices[positive_z] = 4
        face_indices[(dominant_axis == 2) & (z < 0)] = 5

        safe_magnitude = dominant_magnitude.clamp_min(torch.finfo(flat_directions.dtype).tiny)
        u = torch.empty_like(x)
        v = torch.empty_like(y)
        u[positive_x] = -z[positive_x] / safe_magnitude[positive_x]
        v[positive_x] = -y[positive_x] / safe_magnitude[positive_x]
        u[negative_x] = z[negative_x] / safe_magnitude[negative_x]
        v[negative_x] = -y[negative_x] / safe_magnitude[negative_x]
        u[positive_y] = x[positive_y] / safe_magnitude[positive_y]
        v[positive_y] = z[positive_y] / safe_magnitude[positive_y]
        u[negative_y] = x[negative_y] / safe_magnitude[negative_y]
        v[negative_y] = -z[negative_y] / safe_magnitude[negative_y]
        u[positive_z] = x[positive_z] / safe_magnitude[positive_z]
        v[positive_z] = -y[positive_z] / safe_magnitude[positive_z]
        negative_z = (dominant_axis == 2) & (z < 0)
        u[negative_z] = -x[negative_z] / safe_magnitude[negative_z]
        v[negative_z] = -y[negative_z] / safe_magnitude[negative_z]
        return face_indices, torch.stack((u, v), dim=-1).clamp(-1.0, 1.0)

    def forward(self, directions: Tensor) -> Tensor:
        """Return bilinearly sampled sky RGB with the same shape as ``directions``."""

        self._validate_directions(directions)
        output_shape = directions.shape
        flat_directions = directions.reshape(-1, 3)
        if flat_directions.shape[0] == 0:
            return directions.new_empty(output_shape)

        face_indices, uv = self._face_indices_and_uv(flat_directions)
        textures = self.cubemap.permute(0, 3, 1, 2)
        flat_rgb = flat_directions.new_zeros(flat_directions.shape)
        for face_index in range(6):
            point_indices = torch.nonzero(face_indices == face_index, as_tuple=False).squeeze(1)
            if point_indices.numel() == 0:
                continue
            grid = uv.index_select(0, point_indices).reshape(1, -1, 1, 2)
            sampled = F.grid_sample(
                textures[face_index : face_index + 1],
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            sampled = sampled[0, :, :, 0].transpose(0, 1)
            flat_rgb = flat_rgb.index_copy(0, point_indices, sampled)
        return flat_rgb.reshape(output_shape)


__all__ = ["ExplicitCubemapSky"]
