from __future__ import annotations

import pytest
import torch

from armgs.sky import ExplicitCubemapSky


def test_axis_directions_select_documented_faces_and_preserve_shape() -> None:
    sky = ExplicitCubemapSky(resolution=2)
    face_colors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    with torch.no_grad():
        sky.cubemap.copy_(face_colors[:, None, None, :].expand_as(sky.cubemap))

    directions = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        ]
    )
    sampled = sky(directions)

    assert sampled.shape == directions.shape
    torch.testing.assert_close(sampled.reshape(6, 3), face_colors)


def test_bilinear_sampling_and_cubemap_gradient() -> None:
    sky = ExplicitCubemapSky(resolution=(2, 2), initial_color=(0.0, 0.0, 0.0))
    with torch.no_grad():
        sky.cubemap[4, :, :, 0].copy_(torch.tensor([[0.0, 1.0], [2.0, 3.0]]))

    direction = torch.tensor([0.0, 0.0, 1.0])
    sampled = sky(direction)

    torch.testing.assert_close(sampled, torch.tensor([1.5, 0.0, 0.0]))
    sampled.sum().backward()
    assert sky.cubemap.grad is not None
    torch.testing.assert_close(
        sky.cubemap.grad[4, :, :, 0], torch.full((2, 2), 0.25)
    )
    assert torch.count_nonzero(sky.cubemap.grad[:4]) == 0
    assert torch.count_nonzero(sky.cubemap.grad[5]) == 0


@pytest.mark.parametrize(
    ("directions", "error", "message"),
    [
        (torch.ones(2, 2), ValueError, "channel-last"),
        (torch.tensor([1, 0, 0]), TypeError, "floating-point"),
        (torch.tensor([2.0, 0.0, 0.0]), ValueError, "normalized"),
        (torch.tensor([float("nan"), 0.0, 1.0]), ValueError, "finite"),
    ],
)
def test_direction_contract_is_validated(
    directions: torch.Tensor, error: type[Exception], message: str
) -> None:
    sky = ExplicitCubemapSky(resolution=2)
    with pytest.raises(error, match=message):
        sky(directions)
