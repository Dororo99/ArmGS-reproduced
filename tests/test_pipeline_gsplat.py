from __future__ import annotations

from pathlib import Path

import pytest
import torch

from armgs.backends import GsplatRasterizer
from armgs.config import build_core, build_loss, load_config
from armgs.density import (
    DensityControlThresholds,
    GaussianDensityPolicy,
    GsplatDensityController,
)
from armgs.geometry import PoseTrajectory
from armgs.pipeline import ArmGSCompositeRenderer, CameraView
from armgs.scene import (
    CompositeGaussianScene,
    DynamicActorModel,
    LearnableGaussianSet,
)
from armgs.sky import ExplicitCubemapSky
from armgs.structures import GaussianSet
from armgs.time import TimestampNormalizer


pytest.importorskip("gsplat")
ROOT = Path(__file__).resolve().parents[1]
_C0 = 0.28209479177387814


def make_degree_three_gaussian(
    mean: list[float], rgb: list[float]
) -> GaussianSet:
    coefficients = torch.zeros(1, 16, 3)
    coefficients[:, 0] = (torch.tensor(rgb) - 0.5) / _C0
    return GaussianSet(
        means=torch.tensor([mean]),
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        scales=torch.full((1, 3), 0.2),
        opacities=torch.full((1, 1), 0.8),
        sh_coefficients=coefficients,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat CUDA test")
def test_real_gsplat_full_composite_pipeline_backward() -> None:
    config = load_config(ROOT / "configs" / "armgs_default.yaml")
    core = build_core(config, num_training_frames=1)
    background = LearnableGaussianSet(
        make_degree_three_gaussian([0.0, 0.0, 4.0], [1.0, 0.0, 0.0])
    )
    actor_gaussians = LearnableGaussianSet(
        make_degree_three_gaussian([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    )
    actor = DynamicActorModel(
        actor_gaussians,
        PoseTrajectory(
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 2.0]]),
        ),
        actor_id=0,
    )
    scene = CompositeGaussianScene(
        background,
        [actor],
        TimestampNormalizer.from_timestamps(torch.tensor([0, 1])),
        sky=ExplicitCubemapSky(2, initial_color=(0.0, 0.0, 1.0)),
    )
    renderer = ArmGSCompositeRenderer(
        core, scene, GsplatRasterizer()
    ).to(torch.device("cuda"))

    view = CameraView(
        camera_to_world=torch.eye(4),
        intrinsics=torch.tensor(
            [[24.0, 0.0, 8.0], [0.0, 24.0, 8.0], [0.0, 0.0, 1.0]]
        ),
        image_size=(16, 16),
        timestamp=torch.tensor(0, dtype=torch.int64),
        camera_id=torch.tensor(0),
        training_row=torch.tensor(0),
        camera_convention="opencv",
    )
    output = renderer(view)
    assert output.rgb.shape == (1, 16, 16, 3)
    assert output.actor_alpha is not None
    assert output.actor_alpha.shape == (1, 16, 16, 1)
    assert output.actor_alpha.max() > 0
    assert output.sky_rgb is not None

    loss = build_loss(config)(
        output.rgb,
        torch.full_like(output.rgb, 0.25),
        rendered_depth=output.depth,
        lidar_depth=output.depth.detach() + 0.5,
        non_sky_accumulated_alpha=output.non_sky_accumulated_alpha,
        target_sky_mask=torch.zeros_like(output.non_sky_accumulated_alpha),
        actor_alpha=output.actor_alpha,
        actor_bbox_mask=torch.ones_like(output.actor_alpha, dtype=torch.bool),
    )
    density = GsplatDensityController(
        {
            -1: renderer.scene.background,
            0: renderer.scene.actors[0].gaussians,
        },
        GaussianDensityPolicy(
            DensityControlThresholds(
                position_gradient_threshold=0.0002,
                split_scale_threshold=0.01,
                prune_opacity_threshold=0.005,
                split_children=2,
                split_scale_reduction=1.6,
                opacity_reset_value=0.01,
            )
        ),
    )
    metadata = output.rasterization.metadata
    assert metadata is not None
    density.before_backward(metadata)
    loss.total.backward()
    assert output.composite_gaussians.group_ids is not None
    density.after_backward(
        metadata, output.composite_gaussians.group_ids
    )
    assert density.accumulator(-1).observation_count.sum() > 0
    assert density.accumulator(0).observation_count.sum() > 0

    assert torch.isfinite(loss.total)
    assert renderer.scene.background.means.grad is not None
    assert renderer.scene.actors[0].trajectory.translations.grad is not None
    assert renderer.scene.sky is not None
    assert renderer.scene.sky.cubemap.grad is not None
    assert (
        renderer.core.local_refiner.affine_learner.final_layer.weight.grad
        is not None
    )
    assert (
        renderer.core.global_refiner.affine_learner.final_layer.weight.grad
        is not None
    )
    assert renderer.core.actor_refiner.position_head.final_layer.weight.grad is not None
    assert renderer.core.actor_refiner.sh_head.final_layer.weight.grad is not None
