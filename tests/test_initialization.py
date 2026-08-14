from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

import pytest
import torch

from armgs.initialization import (
    GaussianInitializationConfig,
    estimate_knn_isotropic_scales,
    initialize_gaussians_from_points,
    load_colmap_points3d_text,
    merge_colored_point_clouds,
    preprocess_streetgs_waymo_background,
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
    torch.testing.assert_close(
        gaussians.scales,
        torch.full((2, 3), 2.0**0.5),
    )
    torch.testing.assert_close(gaussians.opacities, torch.full((2, 1), 0.2))
    reconstructed = spherical_harmonics_to_rgb(
        gaussians.sh_coefficients,
        torch.tensor([[0.0, 0.0, 1.0]]).expand(2, -1),
        degree=3,
        clamp_min=False,
    )
    torch.testing.assert_close(reconstructed, colors, atol=1.0e-6, rtol=0.0)


def test_knn_scales_match_three_neighbour_reference_in_small_chunks() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [4.0, -1.0, 2.0],
        ],
        dtype=torch.float64,
    )
    pairwise_squared = (
        (points[:, None, :] - points[None, :, :]).square().sum(dim=-1)
    )
    pairwise_squared.fill_diagonal_(torch.inf)
    expected = pairwise_squared.topk(
        3, dim=-1, largest=False, sorted=False
    ).values.mean(dim=-1).sqrt()

    scales = estimate_knn_isotropic_scales(
        points,
        neighbor_count=3,
        chunk_size=2,
        backend="torch",
    )

    assert scales.shape == (5, 3)
    torch.testing.assert_close(scales, expected[:, None].expand(-1, 3))
    torch.testing.assert_close(scales[:, 0], scales[:, 1])
    torch.testing.assert_close(scales[:, 1], scales[:, 2])


def test_knn_scales_clamp_duplicate_points_like_official_3dgs() -> None:
    points = torch.zeros(4, 3)

    scales = estimate_knn_isotropic_scales(
        points,
        neighbor_count=3,
        chunk_size=1,
        backend="torch",
    )

    expected = torch.full((4, 3), (1.0e-7) ** 0.5)
    torch.testing.assert_close(scales, expected)


def test_reference_defaults_initialize_opacity_rotation_and_knn_scale() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    gaussians = initialize_gaussians_from_points(points)

    torch.testing.assert_close(
        gaussians.opacities, torch.full((4, 1), 0.1)
    )
    torch.testing.assert_close(
        gaussians.quaternions,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(4, -1),
    )
    expected_squared = torch.tensor(
        [14.0 / 3.0, 16.0 / 3.0, 22.0 / 3.0, 32.0 / 3.0]
    )
    torch.testing.assert_close(
        gaussians.scales,
        expected_squared.sqrt()[:, None].expand(-1, 3),
    )


def test_single_point_cloud_uses_only_degenerate_fallback_scale() -> None:
    config = GaussianInitializationConfig(initial_scale=0.07)
    singleton = initialize_gaussians_from_points(
        torch.tensor([[1.0, 2.0, 3.0]]),
        config=config,
    )
    pair = initialize_gaussians_from_points(
        torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        config=config,
    )

    torch.testing.assert_close(singleton.scales, torch.full((1, 3), 0.07))
    torch.testing.assert_close(pair.scales, torch.full((2, 3), 2.0))


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


def test_lidar_and_sfm_clouds_merge_and_voxel_deduplicate() -> None:
    lidar_points = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    lidar_colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    sfm_points = torch.tensor([[0.2, 0.0, 0.0], [3.0, 0.0, 0.0]])
    sfm_colors = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    points, colors = merge_colored_point_clouds(
        (
            (lidar_points, lidar_colors),
            (sfm_points, sfm_colors),
        ),
        voxel_size=1.0,
    )

    order = points[:, 0].argsort()
    torch.testing.assert_close(
        points[order],
        torch.tensor(
            [[0.1, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        ),
    )
    torch.testing.assert_close(
        colors[order],
        torch.tensor(
            [[0.5, 0.5, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
        ),
    )


def _install_fake_open3d(
    monkeypatch: pytest.MonkeyPatch,
    *,
    filtered_points: np.ndarray,
    filtered_colors: np.ndarray,
) -> dict[str, object]:
    calls: dict[str, object] = {}

    class FakePointCloud:
        def __init__(self) -> None:
            self.points = np.empty((0, 3), dtype=np.float64)
            self.colors = np.empty((0, 3), dtype=np.float64)

        def voxel_down_sample(self, *, voxel_size: float) -> FakePointCloud:
            calls["voxel_size"] = voxel_size
            calls["input_points"] = np.array(self.points, copy=True)
            calls["input_colors"] = np.array(self.colors, copy=True)
            output = FakePointCloud()
            output.points = np.asarray(filtered_points, dtype=np.float64)
            output.colors = np.asarray(filtered_colors, dtype=np.float64)
            return output

        def remove_radius_outlier(
            self, *, nb_points: int, radius: float
        ) -> tuple[FakePointCloud, list[int]]:
            calls["radius_nb_points"] = nb_points
            calls["radius"] = radius
            return self, list(range(len(self.points)))

    fake_open3d = SimpleNamespace(
        geometry=SimpleNamespace(PointCloud=FakePointCloud),
        utility=SimpleNamespace(
            Vector3dVector=lambda values: np.asarray(values, dtype=np.float64)
        ),
    )
    monkeypatch.setitem(sys.modules, "open3d", fake_open3d)
    return calls


def test_streetgs_waymo_background_uses_official_open3d_order_and_no_revoxel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filtered_lidar = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64
    )
    filtered_colors = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    calls = _install_fake_open3d(
        monkeypatch,
        filtered_points=filtered_lidar,
        filtered_colors=filtered_colors,
    )
    lidar_points = torch.tensor(
        [[-0.01, 0.0, 0.0], [2.01, 0.0, 0.0]], dtype=torch.float32
    )
    lidar_colors = torch.tensor(
        [[0.8, 0.1, 0.0], [0.0, 0.7, 0.2]], dtype=torch.float32
    )
    sfm_points = torch.tensor(
        [[0.0, 0.0, 0.0], [2.999, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    sfm_colors = torch.tensor(
        [[0.0, 0.0, 1.0], [0.2, 0.3, 0.4], [1.0, 1.0, 1.0]]
    )

    result = preprocess_streetgs_waymo_background(
        lidar_points,
        lidar_colors,
        sfm_points,
        sfm_colors,
    )

    assert calls["voxel_size"] == 0.15
    assert calls["radius_nb_points"] == 10
    assert calls["radius"] == 0.5
    np.testing.assert_allclose(calls["input_points"], lidar_points.numpy())
    np.testing.assert_allclose(calls["input_colors"], lidar_colors.numpy())
    torch.testing.assert_close(
        result.points,
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.999, 0.0, 0.0],
            ]
        ),
    )
    torch.testing.assert_close(
        result.colors,
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.2, 0.3, 0.4],
            ]
        ),
    )
    assert result.lidar_point_count == 2
    assert result.sfm_input_point_count == 3
    assert result.sfm_retained_point_count == 2
    torch.testing.assert_close(
        result.lidar_aabb_center, torch.tensor([1.0, 0.0, 0.0])
    )
    torch.testing.assert_close(
        result.lidar_aabb_half_diagonal, torch.tensor(1.0)
    )


def test_streetgs_optional_camera_filter_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filtered_lidar = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64
    )
    _install_fake_open3d(
        monkeypatch,
        filtered_points=filtered_lidar,
        filtered_colors=np.full((2, 3), 0.5, dtype=np.float64),
    )
    lidar_points = torch.from_numpy(filtered_lidar).to(torch.float32)
    lidar_colors = torch.full_like(lidar_points, 0.5)
    sfm_points = torch.tensor(
        [[5.0, 0.0, 2.0], [7.0, 0.0, 0.0], [-4.0, 0.0, 3.0]]
    )
    sfm_colors = torch.full_like(sfm_points, 0.25)
    camera_centers = torch.tensor([[5.0, 0.0, 2.0]])

    default = preprocess_streetgs_waymo_background(
        lidar_points,
        lidar_colors,
        sfm_points,
        sfm_colors,
        camera_centers=camera_centers,
        camera_extent=1.0,
    )
    filtered = preprocess_streetgs_waymo_background(
        lidar_points,
        lidar_colors,
        sfm_points,
        sfm_colors,
        camera_centers=camera_centers,
        filter_sfm_near_or_below_cameras=True,
        camera_extent=1.0,
    )

    assert default.sfm_retained_point_count == 3
    assert filtered.sfm_retained_point_count == 1
    torch.testing.assert_close(filtered.points[-1], sfm_points[-1])


def test_streetgs_waymo_background_imports_open3d_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "open3d", None)
    points = torch.tensor([[0.0, 0.0, 0.0]])
    colors = torch.full_like(points, 0.5)

    with pytest.raises(ImportError, match="requires Open3D"):
        preprocess_streetgs_waymo_background(points, colors)


def test_point_cloud_merge_rejects_empty_and_mixed_dtypes() -> None:
    with pytest.raises(ValueError, match="at least one"):
        merge_colored_point_clouds(())

    points = torch.zeros(1, 3)
    colors = torch.zeros_like(points)
    with pytest.raises(ValueError, match="share a dtype"):
        merge_colored_point_clouds(
            (
                (points, colors),
                (points.to(torch.float64), colors.to(torch.float64)),
            )
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
        {"knn_neighbors": 0},
        {"knn_neighbors": True},
        {"knn_chunk_size": 0},
        {"knn_backend": "unknown"},
        {"minimum_squared_distance": 0.0},
    ],
)
def test_initialization_assumptions_are_validated(
    kwargs: dict[str, float | int],
) -> None:
    with pytest.raises(ValueError):
        GaussianInitializationConfig(**kwargs)
