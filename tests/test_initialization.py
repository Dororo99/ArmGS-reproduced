from __future__ import annotations

from pathlib import Path

import pytest
import torch

from armgs.initialization import (
    GaussianInitializationConfig,
    initialize_gaussians_from_points,
    load_colmap_points3d_text,
    voxel_downsample,
    world_points_to_actor_local,
)
from armgs.spherical_harmonics import spherical_harmonics_to_rgb


def test_point_colors_become_degree_three_sh_dc_exactly() -> None:
    points = torch.tensor([[0.0, 0.0, 2.0], [1.0, 0.0, 3.0]])
    colors = torch.tensor([[0.2, 0.4, 0.6], [0.9, 0.1, 0.3]])
    config = GaussianInitializationConfig(
        sh_degree=3,
        initial_opacity=0.2,
        initial_scale=0.04,
    )
    gaussians = initialize_gaussians_from_points(
        points, colors, config=config
    )

    assert gaussians.means.shape == (2, 3)
    assert gaussians.sh_coefficients.shape == (2, 16, 3)
    torch.testing.assert_close(
        gaussians.quaternions,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1),
    )
    torch.testing.assert_close(gaussians.scales, torch.full((2, 3), 0.04))
    torch.testing.assert_close(gaussians.opacities, torch.full((2, 1), 0.2))
    reconstructed = spherical_harmonics_to_rgb(
        gaussians.sh_coefficients,
        torch.tensor([[0.0, 0.0, 1.0]]).expand(2, -1),
        degree=3,
        clamp_min=False,
    )
    torch.testing.assert_close(reconstructed, colors, atol=1.0e-6, rtol=0.0)


def test_voxel_downsampling_averages_position_and_color() -> None:
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [1.2, 0.0, 0.0]]
    )
    colors = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    downsampled_points, downsampled_colors = voxel_downsample(
        points, colors, 1.0
    )

    order = downsampled_points[:, 0].argsort()
    torch.testing.assert_close(
        downsampled_points[order],
        torch.tensor([[0.1, 0.0, 0.0], [1.2, 0.0, 0.0]]),
    )
    torch.testing.assert_close(
        downsampled_colors[order],
        torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]),
    )


def test_actor_box_points_are_transformed_to_canonical_coordinates() -> None:
    actor_to_world = torch.eye(4)
    actor_to_world[:3, 3] = torch.tensor([10.0, -2.0, 1.0])
    world_points = torch.tensor(
        [[10.5, -2.0, 1.0], [12.0, -2.0, 1.0], [10.0, -2.5, 1.5]]
    )
    local, mask = world_points_to_actor_local(
        world_points,
        actor_to_world,
        box_dimensions=torch.tensor([2.0, 2.0, 2.0]),
    )

    torch.testing.assert_close(mask, torch.tensor([True, False, True]))
    torch.testing.assert_close(
        local, torch.tensor([[0.5, 0.0, 0.0], [0.0, -0.5, 0.5]])
    )


def test_colmap_points3d_text_parser_reads_xyz_and_normalized_rgb(
    tmp_path: Path,
) -> None:
    path = tmp_path / "points3D.txt"
    path.write_text(
        "# comment\n"
        "1 1.0 2.0 3.0 255 128 0 0.1 2 4\n"
        "2 -1.0 0.0 5.0 0 64 255 0.2\n",
        encoding="utf-8",
    )

    points, colors = load_colmap_points3d_text(path)
    torch.testing.assert_close(
        points, torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 5.0]])
    )
    torch.testing.assert_close(
        colors,
        torch.tensor(
            [[1.0, 128.0 / 255.0, 0.0], [0.0, 64.0 / 255.0, 1.0]]
        ),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sh_degree": 4},
        {"initial_opacity": 0.0},
        {"initial_scale": float("nan")},
        {"voxel_size": -1.0},
    ],
)
def test_initialization_assumptions_are_validated(
    kwargs: dict[str, float | int],
) -> None:
    with pytest.raises(ValueError):
        GaussianInitializationConfig(**kwargs)
