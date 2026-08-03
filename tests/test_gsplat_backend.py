from __future__ import annotations

import pytest
import torch

from armgs.backends import GsplatRasterizer
from armgs.backends.gsplat import GsplatRasterizerConfig
from armgs.pipeline import conservative_frustum_visible_indices
from armgs.structures import GaussianSet, RasterizationInput


gsplat = pytest.importorskip("gsplat")


def make_cpu_inputs(*, camera_convention: str = "opencv") -> RasterizationInput:
    return RasterizationInput(
        means=torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 4.0], [0.0, 0.0, 5.0]]),
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(3, -1),
        scales=torch.full((3, 3), 0.1),
        opacities=torch.full((3, 1), 0.5),
        colors=torch.zeros(3, 3),
        camera_to_world=torch.eye(4),
        intrinsics=torch.eye(3),
        image_size=(2, 2),
        group_ids=torch.tensor([-1, 7, 9]),
        camera_convention=camera_convention,
    )


def test_aggregate_actor_feature_is_constant_width_and_per_group_is_explicit() -> None:
    inputs = make_cpu_inputs()
    aggregate = GsplatRasterizer()
    features, labels, has_actors = aggregate._features_and_groups(inputs, 1)

    assert has_actors
    assert labels is None
    assert features.shape == (3, 4)
    torch.testing.assert_close(features[:, 3], torch.tensor([0.0, 1.0, 1.0]))

    per_group = GsplatRasterizer(
        GsplatRasterizerConfig(actor_alpha_mode="per_group")
    )
    features, labels, has_actors = per_group._features_and_groups(inputs, 1)
    assert has_actors
    assert features.shape == (3, 5)
    torch.testing.assert_close(labels, torch.tensor([7, 9]))
    torch.testing.assert_close(
        features[:, 3:], torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    )


def test_opengl_camera_is_converted_to_opencv_axes() -> None:
    camera_to_world, _ = GsplatRasterizer._camera_batch(
        make_cpu_inputs(camera_convention="opengl")
    )
    expected = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0])).unsqueeze(0)
    torch.testing.assert_close(camera_to_world, expected)

    with pytest.raises(ValueError, match="camera_convention"):
        GsplatRasterizer._camera_batch(make_cpu_inputs(camera_convention="unknown"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat CUDA test")
def test_gsplat_adapter_renders_rgb_depth_and_actor_alpha() -> None:
    device = torch.device("cuda")
    means = torch.tensor([[0.0, 0.0, 3.0]], device=device, requires_grad=True)
    inputs = RasterizationInput(
        means=means,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        scales=torch.full((1, 3), 0.15, device=device),
        opacities=torch.full((1, 1), 0.9, device=device),
        colors=torch.tensor(
            [[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]], device=device
        ),
        camera_to_world=torch.eye(4, device=device)[:3].unsqueeze(0).expand(2, -1, -1),
        intrinsics=torch.tensor(
            [[24.0, 0.0, 8.0], [0.0, 24.0, 8.0], [0.0, 0.0, 1.0]],
            device=device,
        ).unsqueeze(0).expand(2, -1, -1),
        image_size=(16, 16),
        group_ids=torch.tensor([7], device=device),
        velocities=torch.zeros(1, 3, device=device),
        camera_linear_velocity=torch.zeros(2, 3, device=device),
        camera_angular_velocity=torch.zeros(2, 3, device=device),
        rolling_shutter_time=torch.zeros(2, device=device),
        rolling_shutter_direction=1,
    )
    output = GsplatRasterizer()(inputs)

    assert output.rgb.shape == (2, 16, 16, 3)
    assert output.depth.shape == (2, 16, 16, 1)
    assert output.accumulated_alpha.shape == (2, 16, 16, 1)
    assert output.actor_alpha is not None
    assert output.actor_alpha.shape == (2, 16, 16, 1)
    assert output.group_alpha is None
    assert output.group_labels is None
    assert output.metadata is not None
    assert output.metadata["armgs_actor_alpha_mode"] == "aggregate"
    assert output.accumulated_alpha.max() > 0.0
    assert output.actor_alpha.max() > 0.0
    assert torch.isfinite(output.rgb).all()
    assert output.rgb[0, ..., 0].max() > output.rgb[0, ..., 1].max()
    assert output.rgb[1, ..., 1].max() > output.rgb[1, ..., 0].max()

    output.rgb.sum().backward()
    assert means.grad is not None
    assert torch.isfinite(means.grad).all()



@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat CUDA test")
def test_aggregate_actor_alpha_matches_sum_of_per_group_channels() -> None:
    device = torch.device("cuda")
    inputs = RasterizationInput(
        means=torch.tensor(
            [[-0.1, 0.0, 3.0], [0.1, 0.0, 3.5], [0.0, 0.1, 4.0]],
            device=device,
        ),
        quaternions=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], device=device
        ).expand(3, -1),
        scales=torch.full((3, 3), 0.2, device=device),
        opacities=torch.full((3, 1), 0.7, device=device),
        colors=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            device=device,
        ),
        camera_to_world=torch.eye(4, device=device),
        intrinsics=torch.tensor(
            [[24.0, 0.0, 8.0], [0.0, 24.0, 8.0], [0.0, 0.0, 1.0]],
            device=device,
        ),
        image_size=(16, 16),
        group_ids=torch.tensor([-1, 7, 9], device=device),
    )
    aggregate = GsplatRasterizer()(inputs)
    per_group = GsplatRasterizer(
        GsplatRasterizerConfig(actor_alpha_mode="per_group")
    )(inputs)

    assert aggregate.actor_alpha is not None
    assert aggregate.group_alpha is None
    assert per_group.actor_alpha is not None
    assert per_group.group_alpha is not None
    torch.testing.assert_close(per_group.group_labels, torch.tensor([7, 9], device=device))
    torch.testing.assert_close(aggregate.rgb, per_group.rgb)
    torch.testing.assert_close(aggregate.depth, per_group.depth)
    torch.testing.assert_close(
        aggregate.actor_alpha,
        per_group.group_alpha.sum(dim=-1, keepdim=True),
    )
    torch.testing.assert_close(aggregate.actor_alpha, per_group.actor_alpha)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat CUDA test")
def test_default_culler_keeps_eps2d_edge_splat_that_gsplat_renders() -> None:
    device = torch.device("cuda")
    means = torch.tensor([[-0.5625, 0.0, 3.0]], device=device)
    quaternions = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]], device=device
    )
    scales = torch.full((1, 3), 0.001, device=device)
    opacities = torch.full((1, 1), 0.9, device=device)
    gaussians = GaussianSet(
        means=means,
        quaternions=quaternions,
        scales=scales,
        opacities=opacities,
        sh_coefficients=torch.zeros(1, 1, 3, device=device),
    )
    intrinsics = torch.tensor(
        [[24.0, 0.0, 4.0], [0.0, 24.0, 4.0], [0.0, 0.0, 1.0]],
        device=device,
    )
    visible = conservative_frustum_visible_indices(
        gaussians,
        torch.eye(4, device=device),
        intrinsics,
        (8, 8),
        convention="opencv",
    )
    output = GsplatRasterizer()(
        RasterizationInput(
            means=means,
            quaternions=quaternions,
            scales=scales,
            opacities=opacities,
            colors=torch.ones(1, 3, device=device),
            camera_to_world=torch.eye(4, device=device),
            intrinsics=intrinsics,
            image_size=(8, 8),
        )
    )

    torch.testing.assert_close(visible, torch.tensor([0], device=device))
    assert output.accumulated_alpha.max() > 0.0
