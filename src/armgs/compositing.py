"""Reference front-to-back alpha compositing for tests and backend validation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CompositeResult:
    rgb: Tensor
    accumulated_alpha: Tensor
    weights: Tensor
    depth: Tensor | None = None


def front_to_back_composite(
    colors: Tensor,
    alphas: Tensor,
    *,
    depths: Tensor | None = None,
    eps: float = 1.0e-8,
) -> CompositeResult:
    """Composite already depth-ordered samples along one or more rays.

    The first dimension is sample order. Remaining dimensions (except channels)
    represent arbitrary ray batches. ``colors`` has shape ``[N,...,3]`` and
    ``alphas`` has shape ``[N,...,1]``.
    """

    if colors.ndim < 2 or colors.shape[-1] != 3:
        raise ValueError("colors must have shape [N,...,3]")
    if alphas.shape != (*colors.shape[:-1], 1):
        raise ValueError("alphas must match colors with a final singleton channel")
    if depths is not None and depths.shape not in (
        colors.shape[:-1],
        (*colors.shape[:-1], 1),
    ):
        raise ValueError("depths must have shape [N,...] or [N,...,1]")

    one = torch.ones_like(alphas[:1])
    transmittance = torch.cumprod(
        torch.cat((one, 1.0 - alphas[:-1]), dim=0), dim=0
    )
    weights = transmittance * alphas
    rgb = (weights * colors).sum(dim=0)
    accumulated_alpha = weights.sum(dim=0)
    rendered_depth: Tensor | None = None
    if depths is not None:
        if depths.ndim == colors.ndim - 1:
            depths = depths.unsqueeze(-1)
        depth_numerator = (weights * depths).sum(dim=0)
        rendered_depth = depth_numerator / accumulated_alpha.clamp_min(eps)
    return CompositeResult(
        rgb=rgb,
        accumulated_alpha=accumulated_alpha,
        weights=weights,
        depth=rendered_depth,
    )


def composite_sky(
    foreground_rgb: Tensor, accumulated_alpha: Tensor, sky_rgb: Tensor
) -> Tensor:
    if foreground_rgb.shape[-1] != 3 or sky_rgb.shape != foreground_rgb.shape:
        raise ValueError("foreground_rgb and sky_rgb must share shape [...,3]")
    if accumulated_alpha.shape != (*foreground_rgb.shape[:-1], 1):
        raise ValueError("accumulated_alpha must have shape [...,1]")
    return foreground_rgb + (1.0 - accumulated_alpha) * sky_rgb

