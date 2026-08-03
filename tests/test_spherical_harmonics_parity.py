from __future__ import annotations

import pytest
import torch
from torch import Tensor

from armgs.spherical_harmonics import (
    camera_to_gaussian_directions,
    evaluate_spherical_harmonics,
    spherical_harmonics_to_rgb,
)


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


def _official_3dgs_reference(
    coefficients: Tensor, directions: Tensor, degree: int
) -> Tensor:
    """Transcribe Inria computeColorFromSH before RGB postprocessing."""

    directions = directions / torch.linalg.vector_norm(
        directions, dim=-1, keepdim=True
    )
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
            + _C3[3]
            * (z * (2.0 * zz - 3.0 * xx - 3.0 * yy))[:, None]
            * coefficients[:, 12]
            + _C3[4] * (x * (4.0 * zz - xx - yy))[:, None] * coefficients[:, 13]
            + _C3[5] * (z * (xx - yy))[:, None] * coefficients[:, 14]
            + _C3[6] * (x * (xx - 3.0 * yy))[:, None] * coefficients[:, 15]
        )
    return result


@pytest.mark.parametrize("degree", range(4))
def test_matches_official_3dgs_cpu_reference(degree: int) -> None:
    generator = torch.Generator().manual_seed(100 + degree)
    coefficient_count = (degree + 1) ** 2
    coefficients = torch.randn(
        (17, coefficient_count, 3), generator=generator, dtype=torch.float64
    )
    directions = torch.randn((17, 3), generator=generator, dtype=torch.float64)

    expected = _official_3dgs_reference(coefficients, directions, degree)
    actual = evaluate_spherical_harmonics(coefficients, directions, degree)

    torch.testing.assert_close(actual, expected, atol=1.0e-12, rtol=1.0e-12)


def test_camera_direction_uses_camera_to_gaussian_sign() -> None:
    means = torch.tensor([[2.0, 2.0, 3.0], [1.0, 0.0, 3.0]])
    camera_center = torch.tensor([1.0, 2.0, 3.0])

    directions = camera_to_gaussian_directions(means, camera_center)

    torch.testing.assert_close(
        directions,
        torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
    )
    coefficients = torch.zeros(1, 4, 3)
    coefficients[0, 3, 0] = 1.0
    camera_to_gaussian = evaluate_spherical_harmonics(
        coefficients, directions[:1], degree=1
    )
    reversed_direction = evaluate_spherical_harmonics(
        coefficients, -directions[:1], degree=1
    )
    torch.testing.assert_close(
        camera_to_gaussian[:, 0], torch.tensor([-_C1])
    )
    torch.testing.assert_close(reversed_direction[:, 0], torch.tensor([_C1]))


def test_rgb_offset_and_clamp_are_owned_by_rgb_wrapper() -> None:
    desired_signed_value = torch.tensor([[-1.0, 0.25, 1.0]])
    coefficients = (desired_signed_value / _C0).unsqueeze(1)
    directions = torch.tensor([[0.0, 0.0, 1.0]])

    signed = evaluate_spherical_harmonics(coefficients, directions, degree=0)
    offset_only = spherical_harmonics_to_rgb(
        coefficients, directions, degree=0, clamp_min=False
    )
    renderer_rgb = spherical_harmonics_to_rgb(coefficients, directions, degree=0)

    torch.testing.assert_close(signed, desired_signed_value)
    torch.testing.assert_close(offset_only, desired_signed_value + 0.5)
    torch.testing.assert_close(
        renderer_rgb, torch.tensor([[0.0, 0.75, 1.5]])
    )


@pytest.mark.parametrize("degree", range(4))
def test_matches_installed_gsplat_torch_reference(degree: int) -> None:
    pytest.importorskip("gsplat")
    from gsplat.cuda._torch_impl import _spherical_harmonics

    generator = torch.Generator().manual_seed(200 + degree)
    coefficients = torch.randn(
        (19, (degree + 1) ** 2, 3), generator=generator, dtype=torch.float64
    )
    directions = torch.randn((19, 3), generator=generator, dtype=torch.float64)

    expected = _spherical_harmonics(degree, directions, coefficients)
    actual = evaluate_spherical_harmonics(coefficients, directions, degree)

    torch.testing.assert_close(actual, expected, atol=1.0e-12, rtol=1.0e-12)


@pytest.mark.parametrize("degree", range(4))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat CUDA parity test")
def test_matches_installed_gsplat_cuda_operator(degree: int) -> None:
    pytest.importorskip("gsplat")
    from gsplat.cuda._wrapper import spherical_harmonics as gsplat_sh

    torch.manual_seed(300 + degree)
    coefficients = torch.randn(
        23, (degree + 1) ** 2, 3, device="cuda", dtype=torch.float32
    )
    directions = torch.randn(23, 3, device="cuda", dtype=torch.float32)

    expected = gsplat_sh(degree, directions, coefficients)
    actual = evaluate_spherical_harmonics(coefficients, directions, degree)

    torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=2.0e-5)
