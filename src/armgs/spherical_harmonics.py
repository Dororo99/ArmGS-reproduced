"""Real spherical harmonics in the original 3DGS/gsplat convention.

The responsibility boundary intentionally mirrors the reference renderers:

* camera_to_gaussian_directions forms mean - camera_center and normalizes it
  (camera-to-Gaussian, not Gaussian-to-camera).
* evaluate_spherical_harmonics evaluates the signed degree 0--3 SH series only.
  It does not add an RGB offset or clamp its result.
* spherical_harmonics_to_rgb owns the reference renderer's +0.5 color offset
  and optional lower clamp.

Consequently, callers that pass precomputed RGB to a rasterizer must call
spherical_harmonics_to_rgb exactly once; a rasterizer evaluating raw SH
coefficients internally is responsible for the offset and clamp itself.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

_C0 = 0.28209479177387814
_C1 = 0.4886025119029199
_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def camera_to_gaussian_directions(means: Tensor, camera_center: Tensor) -> Tensor:
    """Return normalized mean - camera_center directions used by 3DGS."""

    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError("means must have shape [N,3]")
    if camera_center.shape != (3,):
        raise ValueError("camera_center must have shape [3]")
    return F.normalize(means - camera_center, dim=-1, eps=1.0e-8)


def evaluate_spherical_harmonics(
    coefficients: Tensor, directions: Tensor, degree: int | None = None
) -> Tensor:
    """Evaluate signed degree 0--3 SH values without +0.5 or clamping.

    Coefficients use the real-SH ordering and signs from the official Inria
    3DGS rasterizer. Directions are normalized internally, matching gsplat's
    public spherical-harmonics operator.
    """

    if coefficients.ndim != 3 or coefficients.shape[-1] != 3:
        raise ValueError("coefficients must have shape [N,K,3]")
    if directions.shape != (coefficients.shape[0], 3):
        raise ValueError("directions must have shape [N,3]")
    coefficient_count = coefficients.shape[1]
    if degree is None:
        degree = int(coefficient_count**0.5) - 1
    if degree < 0 or degree > 3:
        raise ValueError("the reference evaluator supports SH degrees zero through three")
    required = (degree + 1) ** 2
    if coefficient_count < required:
        raise ValueError(f"degree {degree} requires at least {required} coefficients")

    directions = F.normalize(directions, dim=-1, eps=1.0e-8)
    x, y, z = directions.unbind(dim=-1)
    result = _C0 * coefficients[:, 0]
    if degree > 0:
        result = (
            result
            - _C1 * y[:, None] * coefficients[:, 1]
            + _C1 * z[:, None] * coefficients[:, 2]
            - _C1 * x[:, None] * coefficients[:, 3]
        )
    if degree > 1:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result = (
            result
            + _C2[0] * xy[:, None] * coefficients[:, 4]
            + _C2[1] * yz[:, None] * coefficients[:, 5]
            + _C2[2] * (2.0 * zz - xx - yy)[:, None] * coefficients[:, 6]
            + _C2[3] * xz[:, None] * coefficients[:, 7]
            + _C2[4] * (xx - yy)[:, None] * coefficients[:, 8]
        )
    if degree > 2:
        result = (
            result
            + _C3[0] * (y * (3.0 * xx - yy))[:, None] * coefficients[:, 9]
            + _C3[1] * (x * y * z)[:, None] * coefficients[:, 10]
            + _C3[2] * (y * (4.0 * zz - xx - yy))[:, None] * coefficients[:, 11]
            + _C3[3] * (z * (2.0 * zz - 3.0 * xx - 3.0 * yy))[:, None]
            * coefficients[:, 12]
            + _C3[4] * (x * (4.0 * zz - xx - yy))[:, None] * coefficients[:, 13]
            + _C3[5] * (z * (xx - yy))[:, None] * coefficients[:, 14]
            + _C3[6] * (x * (xx - 3.0 * yy))[:, None] * coefficients[:, 15]
        )
    return result


def spherical_harmonics_to_rgb(
    coefficients: Tensor,
    directions: Tensor,
    degree: int | None = None,
    *,
    color_offset: float = 0.5,
    clamp_min: bool = True,
) -> Tensor:
    """Convert signed SH values to renderer-ready RGB.

    The default +0.5 followed by clamp_min(0) matches both the official 3DGS
    preprocessing kernel and gsplat's SH rendering path. Set clamp_min=False
    only when an unclamped affine/color stage explicitly needs the offset
    result.
    """

    colors = evaluate_spherical_harmonics(coefficients, directions, degree)
    colors = colors + color_offset
    return colors.clamp_min(0.0) if clamp_min else colors

