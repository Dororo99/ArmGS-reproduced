from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from armgs.actor import ActorDeformationRefiner
from armgs.appearance import (
    FrameAppearanceEmbedding,
    GlobalImageAppearanceRefiner,
    LocalGaussianAppearanceRefiner,
    ViewpointEncoder,
)
from armgs.compositing import front_to_back_composite
from armgs.config import build_loss, load_config
from armgs.encodings import HashGridEncoder
from armgs.geometry import PoseTrajectory
from armgs.losses import ArmGSLoss
from armgs.model import ArmGSCore
from armgs.pipeline import (
    ArmGSCompositeRenderer,
    CameraView,
    camera_center_and_forward,
    conservative_frustum_visible_indices,
    world_ray_directions,
)
from armgs.scene import (
    CompositeGaussianScene,
    DynamicActorModel,
    LearnableGaussianSet,
)
from armgs.sampling import StatefulShuffleSampler
from armgs.sky import ExplicitCubemapSky
from armgs.structures import GaussianSet, RasterizationInput, RasterizationOutput
from armgs.time import TimestampNormalizer
from armgs.training import ArmGSTrainer, ArmGSTrainingBatch


_C0 = 0.28209479177387814


def rgb_to_degree_zero_sh(rgb: torch.Tensor) -> torch.Tensor:
    return ((rgb - 0.5) / _C0).reshape(1, 1, 3)


def make_gaussian(mean: list[float], rgb: list[float], opacity: float = 0.8) -> GaussianSet:
    return GaussianSet(
        means=torch.tensor([mean], dtype=torch.float32),
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        scales=torch.full((1, 3), 0.1),
        opacities=torch.tensor([[opacity]]),
        sh_coefficients=rgb_to_degree_zero_sh(torch.tensor(rgb)),
    )


def make_core() -> ArmGSCore:
    embedding_dim = 4
    return ArmGSCore(
        FrameAppearanceEmbedding(1, embedding_dim),
        LocalGaussianAppearanceRefiner(
            HashGridEncoder(
                num_levels=2,
                features_per_level=2,
                log2_hashmap_size=5,
                base_resolution=4,
                max_resolution=8,
            ),
            embedding_dim,
            hidden_dim=8,
            num_layers=3,
        ),
        GlobalImageAppearanceRefiner(
            ViewpointEncoder(position_frequencies=1, direction_frequencies=1),
            embedding_dim,
            hidden_dim=8,
            num_layers=4,
        ),
        ActorDeformationRefiner(
            0,
            hidden_dim=8,
            position_frequencies=1,
            time_frequencies=1,
            encoder_layers=2,
            head_layers=2,
        ),
    )


class OnePixelDepthRasterizer:
    """Exact differentiable rasterizer for paper-order integration tests."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, inputs: RasterizationInput) -> RasterizationOutput:
        self.calls += 1
        assert inputs.colors.ndim == 2
        order = inputs.means[:, 2].argsort()
        colors = inputs.colors.index_select(0, order)
        alphas = inputs.opacities.index_select(0, order)
        depths = inputs.means.index_select(0, order)[:, 2:3]
        composite = front_to_back_composite(colors, alphas, depths=depths)
        assert composite.depth is not None
        actor_alpha = None
        if inputs.group_ids is not None:
            actor_indicator = (inputs.group_ids.index_select(0, order) >= 0).to(
                composite.weights
            )
            actor_alpha = (
                composite.weights[:, 0] * actor_indicator
            ).sum().reshape(1, 1, 1, 1)
        return RasterizationOutput(
            rgb=composite.rgb.reshape(1, 1, 1, 3),
            depth=composite.depth.reshape(1, 1, 1, 1),
            accumulated_alpha=composite.accumulated_alpha.reshape(1, 1, 1, 1),
            actor_alpha=actor_alpha,
            metadata={"sorted_group_ids": inputs.group_ids.index_select(0, order)},
        )


def make_renderer() -> tuple[ArmGSCompositeRenderer, OnePixelDepthRasterizer]:
    core = make_core()
    background = LearnableGaussianSet(
        make_gaussian([0.0, 0.0, 4.0], [1.0, 0.0, 0.0])
    )
    actor_gaussians = LearnableGaussianSet(
        make_gaussian([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    )
    trajectory = PoseTrajectory(
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 2.0]]),
    )
    actor = DynamicActorModel(actor_gaussians, trajectory, actor_id=7)
    sky = ExplicitCubemapSky(resolution=2, initial_color=(0.0, 0.0, 1.0))
    scene = CompositeGaussianScene(
        background,
        [actor],
        TimestampNormalizer.from_timestamps(torch.tensor([0, 1])),
        sky=sky,
    )
    rasterizer = OnePixelDepthRasterizer()
    return ArmGSCompositeRenderer(core, scene, rasterizer), rasterizer


def make_view() -> CameraView:
    return CameraView(
        camera_to_world=torch.eye(4),
        intrinsics=torch.tensor(
            [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]
        ),
        image_size=(1, 1),
        timestamp=torch.tensor(0, dtype=torch.int64),
        camera_id=torch.tensor(0),
        training_row=torch.tensor(0),
        camera_convention="opencv",
    )


def test_conservative_frustum_culling_selects_before_hash_grid() -> None:
    gaussians = GaussianSet(
        means=torch.tensor(
            [[0.0, 0.0, 3.0], [100.0, 0.0, 3.0], [0.0, 0.0, -1.0]]
        ),
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(3, -1),
        scales=torch.full((3, 3), 0.1),
        opacities=torch.full((3, 1), 0.5),
        sh_coefficients=torch.zeros(3, 1, 3),
    )
    intrinsics = torch.tensor(
        [[10.0, 0.0, 8.0], [0.0, 10.0, 8.0], [0.0, 0.0, 1.0]]
    )
    visible = conservative_frustum_visible_indices(
        gaussians,
        torch.eye(4),
        intrinsics,
        (16, 16),
        convention="opencv",
    )
    torch.testing.assert_close(visible, torch.tensor([0]))


def test_frustum_culling_keeps_eps2d_supported_edge_splat() -> None:
    gaussians = make_gaussian(
        [-0.5625, 0.0, 3.0], [1.0, 0.0, 0.0]
    ).with_updates(scales=torch.full((1, 3), 0.001))
    intrinsics = torch.tensor(
        [[24.0, 0.0, 4.0], [0.0, 24.0, 4.0], [0.0, 0.0, 1.0]]
    )
    without_padding = conservative_frustum_visible_indices(
        gaussians,
        torch.eye(4),
        intrinsics,
        (8, 8),
        convention="opencv",
        eps2d=0.0,
    )
    assert without_padding.numel() == 0

    visible = conservative_frustum_visible_indices(
        gaussians,
        torch.eye(4),
        intrinsics,
        (8, 8),
        convention="opencv",
        eps2d=0.3,
    )
    torch.testing.assert_close(visible, torch.tensor([0]))


def test_camera_conventions_are_explicit_and_ray_generation_matches() -> None:
    identity = torch.eye(4)
    _, cv_forward = camera_center_and_forward(identity, "opencv")
    _, gl_forward = camera_center_and_forward(identity, "opengl")
    torch.testing.assert_close(cv_forward, torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(gl_forward, torch.tensor([0.0, 0.0, -1.0]))

    intrinsics = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]
    )
    cv_ray = world_ray_directions(
        identity, intrinsics, (1, 1), convention="opencv"
    )
    gl_ray = world_ray_directions(
        identity, intrinsics, (1, 1), convention="opengl"
    )
    torch.testing.assert_close(cv_ray[0, 0], cv_forward)
    torch.testing.assert_close(gl_ray[0, 0], gl_forward)


def test_single_depth_order_then_sky_then_global_order_is_numerically_fixed() -> None:
    renderer, rasterizer = make_renderer()
    with torch.no_grad():
        local_final = renderer.core.local_refiner.affine_learner.final_layer
        local_final.bias[:3].fill_(1.0)
        local_final.bias[3:].copy_(torch.tensor([0.1, 0.0, 0.0]))
        global_final = renderer.core.global_refiner.affine_learner.final_layer
        global_final.bias[:9].copy_((2.0 * torch.eye(3)).reshape(-1))
        global_final.bias[9:].zero_()

    front = renderer(make_view())
    expected_front = torch.tensor([[[[0.512, 1.6, 0.08]]]])
    torch.testing.assert_close(front.rgb, expected_front, atol=1.0e-5, rtol=0.0)
    assert rasterizer.calls == 1
    assert front.rasterization.metadata is not None
    torch.testing.assert_close(
        front.rasterization.metadata["sorted_group_ids"], torch.tensor([7, -1])
    )

    with torch.no_grad():
        renderer.scene.actors[0].trajectory.translations[0, 2] = 6.0
    behind = renderer(make_view())
    expected_behind = torch.tensor([[[[1.792, 0.32, 0.08]]]])
    torch.testing.assert_close(behind.rgb, expected_behind, atol=1.0e-5, rtol=0.0)
    assert rasterizer.calls == 2
    torch.testing.assert_close(
        behind.rasterization.metadata["sorted_group_ids"], torch.tensor([-1, 7])
    )


def test_full_pipeline_and_equation_nine_backpropagate_end_to_end() -> None:
    renderer, _ = make_renderer()
    output = renderer(make_view())
    assert output.actor_alpha is not None
    target_rgb = torch.full_like(output.rgb, 0.25)
    lidar_depth = output.depth.detach() + 0.5
    sky_mask = torch.zeros_like(output.non_sky_accumulated_alpha)
    actor_bbox = torch.ones_like(output.actor_alpha, dtype=torch.bool)
    losses = ArmGSLoss(require_auxiliary=True)(
        output.rgb,
        target_rgb,
        rendered_depth=output.depth,
        lidar_depth=lidar_depth,
        non_sky_accumulated_alpha=output.non_sky_accumulated_alpha,
        target_sky_mask=sky_mask,
        actor_alpha=output.actor_alpha,
        actor_bbox_mask=actor_bbox,
    )
    losses.total.backward()

    assert torch.isfinite(losses.total)
    assert renderer.scene.background.means.grad is not None
    assert renderer.scene.background.opacity_logits.grad is not None
    assert renderer.scene.actors[0].trajectory.translations.grad is not None
    assert renderer.scene.actors[0].gaussians.sh_coefficients.grad is not None
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


def test_trainer_registers_every_parameter_group_and_resumes_exactly() -> None:
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "armgs_default.yaml"
    )
    renderer, _ = make_renderer()
    trainer = ArmGSTrainer.from_config(renderer, build_loss(config), config)
    group_names = {group["name"] for group in trainer.optimizer.param_groups}
    assert group_names == {
        "means",
        "rotations",
        "scales",
        "opacities",
        "spherical_harmonics",
        "actor_pose",
        "appearance",
        "actor_deformation",
        "sky",
    }
    final_lr = trainer.mean_scheduler.learning_rate_at(
        config["optimization"]["iterations"] - 1
    )
    assert abs(final_lr - config["optimization"]["learning_rates"]["mean_final"]) < 1.0e-12

    with torch.no_grad():
        reference = renderer(make_view())
    assert reference.actor_alpha is not None
    batch = ArmGSTrainingBatch(
        view=make_view(),
        target_rgb=torch.full_like(reference.rgb, 0.25),
        lidar_depth=reference.depth + 0.5,
        target_sky_mask=torch.zeros_like(reference.non_sky_accumulated_alpha),
        actor_bbox_mask=torch.ones_like(reference.actor_alpha, dtype=torch.bool),
    )
    result = trainer.train_step(batch)
    assert result.step == 0
    assert trainer.step == 1
    assert torch.isfinite(result.losses.total)

    state = trainer.state_dict()
    expected_random = torch.rand(4)
    restored_renderer, _ = make_renderer()
    restored = ArmGSTrainer.from_config(
        restored_renderer, build_loss(config), config
    )
    restored.load_state_dict(state)
    assert restored.step == trainer.step
    torch.testing.assert_close(torch.rand(4), expected_random)
    with torch.no_grad():
        expected = trainer.renderer(make_view()).rgb
        actual = restored.renderer(make_view()).rgb
    torch.testing.assert_close(actual, expected)


def test_actor_is_not_ghosted_outside_its_track_interval() -> None:
    renderer, _ = make_renderer()
    base = make_view()
    outside = CameraView(
        camera_to_world=base.camera_to_world,
        intrinsics=base.intrinsics,
        image_size=base.image_size,
        timestamp=torch.tensor(1, dtype=torch.int64),
        camera_id=base.camera_id,
        training_row=base.training_row,
        camera_convention=base.camera_convention,
    )
    output = renderer(outside)
    assert output.composite_gaussians.count == 1
    assert output.actor_alpha is not None
    torch.testing.assert_close(output.actor_alpha, torch.zeros_like(output.actor_alpha))


def test_active_actor_does_not_hide_missing_backend_alpha_support() -> None:
    renderer, rasterizer = make_renderer()

    class NoActorAlphaRasterizer:
        def __call__(self, inputs: RasterizationInput) -> RasterizationOutput:
            return replace(rasterizer(inputs), actor_alpha=None)

    renderer.rasterizer = NoActorAlphaRasterizer()
    output = renderer(make_view())
    assert output.actor_alpha is None


def test_near_zero_gaussian_quaternion_is_rejected() -> None:
    invalid = make_gaussian(
        [0.0, 0.0, 3.0], [1.0, 0.0, 0.0]
    ).with_updates(quaternions=torch.zeros(1, 4))
    with pytest.raises(
        ValueError, match="initial quaternions cannot be near zero"
    ):
        LearnableGaussianSet(invalid)

    valid = LearnableGaussianSet(
        make_gaussian([0.0, 0.0, 3.0], [1.0, 0.0, 0.0])
    )
    with torch.no_grad():
        valid.quaternions.zero_()
    with pytest.raises(ValueError, match="near-zero quaternion"):
        valid.activated()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("rotation", -1.0),
        ("opacity", float("nan")),
        ("mean_final", float("inf")),
    ],
)
def test_trainer_rejects_non_finite_or_non_positive_learning_rates(
    name: str, value: float
) -> None:
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "armgs_default.yaml"
    )
    config["optimization"]["learning_rates"][name] = value
    renderer, _ = make_renderer()
    with pytest.raises(ValueError, match="finite and positive"):
        ArmGSTrainer.from_config(renderer, build_loss(config), config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA RNG state test")
def test_checkpoint_accepts_cuda_mapped_rng_and_extra_device_states() -> None:
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "armgs_default.yaml"
    )
    renderer, _ = make_renderer()
    trainer = ArmGSTrainer.from_config(renderer, build_loss(config), config)
    state = trainer.state_dict()
    mapped_state = dict(state)
    mapped_state["torch_rng_state"] = state["torch_rng_state"].cuda()
    cuda_states = state["cuda_rng_state_all"]
    assert isinstance(cuda_states, list) and cuda_states
    mapped_state["cuda_rng_state_all"] = [
        rng_state.cuda() for rng_state in cuda_states
    ] + [cuda_states[0].cuda()]
    trainer.load_state_dict(mapped_state)


def test_nonzero_rolling_shutter_disables_automatic_frustum_culling() -> None:
    renderer, _ = make_renderer()
    nonzero = replace(
        make_view(),
        rolling_shutter_time=torch.tensor(1.0),
    )
    output = renderer(nonzero)
    assert output.visible_indices is None

    zero = replace(
        make_view(),
        rolling_shutter_time=torch.tensor(0.0),
    )
    assert renderer(zero).visible_indices is not None

    for invalid in (torch.tensor(-1.0), torch.tensor(float("nan"))):
        with pytest.raises(ValueError, match="rolling_shutter_time"):
            renderer(replace(make_view(), rolling_shutter_time=invalid))


def test_trainer_checkpoint_restores_sampler_mid_epoch() -> None:
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "armgs_default.yaml"
    )
    sampler = StatefulShuffleSampler(7, seed=23)
    iterator = iter(sampler)
    consumed = [next(iterator) for _ in range(3)]
    assert len(set(consumed)) == 3

    renderer, _ = make_renderer()
    trainer = ArmGSTrainer.from_config(
        renderer,
        build_loss(config),
        config,
        sampler=sampler,
    )
    state = trainer.state_dict()
    expected_remaining = list(iterator)

    restored_sampler = StatefulShuffleSampler(7, seed=23)
    restored_renderer, _ = make_renderer()
    restored = ArmGSTrainer.from_config(
        restored_renderer,
        build_loss(config),
        config,
        sampler=restored_sampler,
    )
    restored.load_state_dict(state)
    assert list(restored_sampler) == expected_remaining

    trainer_without_sampler = ArmGSTrainer.from_config(
        make_renderer()[0], build_loss(config), config
    )
    with pytest.raises(ValueError, match="trainer has no sampler"):
        trainer_without_sampler.load_state_dict(state)
