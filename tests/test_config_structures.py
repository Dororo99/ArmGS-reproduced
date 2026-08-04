from __future__ import annotations

from pathlib import Path

import torch

from armgs.config import (
    build_core,
    build_density_policy,
    build_initialization_config,
    build_loss,
    build_sampler,
    build_sky,
    load_config,
)
from armgs.structures import GaussianSet


ROOT = Path(__file__).resolve().parents[1]


def make_set(offset: float) -> GaussianSet:
    return GaussianSet(
        means=torch.full((1, 3), offset),
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        scales=torch.ones(1, 3),
        opacities=torch.ones(1, 1),
        sh_coefficients=torch.zeros(1, 16, 3),
        group_ids=torch.tensor([int(offset)]),
    )


def test_default_config_builds_paper_layer_counts() -> None:
    config = load_config(ROOT / "configs" / "armgs_default.yaml")
    core = build_core(
        config,
        num_training_frames=3,
        training_camera_ids=torch.tensor([0, 1, 0]),
        training_timestamps=torch.tensor([0.0, 1.0, 2.0]),
    )
    loss = build_loss(config)

    assert len(core.local_refiner.affine_learner.layers) == 3
    assert len(core.global_refiner.affine_learner.layers) == 4
    assert len(core.actor_refiner.spatial_temporal_encoder.layers) == 2
    assert loss.lambda_ssim == 0.2
    assert loss.require_auxiliary
    assert core.local_refiner.hash_chunk_size == 65536
    assert core.local_refiner.scale_delta_limit == 1.0
    assert core.global_refiner.matrix_delta_limit == 0.5
    novel_embedding = core.novel_view_embedding(
        torch.tensor(0), torch.tensor(1.8)
    )
    expected_embedding = core.training_frame_embedding(torch.tensor(2))
    torch.testing.assert_close(novel_embedding, expected_embedding)


def test_nuscenes_scene_0061_enables_strict_paper_auxiliary_losses() -> None:
    config = load_config(ROOT / "configs" / "armgs_nuscenes_scene_0061.yaml")
    loss = build_loss(config)

    assert loss.require_auxiliary
    assert loss.lambda_depth == 0.01
    assert loss.lambda_sky == 0.05
    assert loss.lambda_foreground == 0.1


def test_gaussian_sets_concatenate_and_preserve_groups() -> None:
    combined = GaussianSet.concatenate([make_set(0.0), make_set(1.0)])

    assert combined.count == 2
    assert combined.sh_degree == 3
    torch.testing.assert_close(combined.group_ids, torch.tensor([0, 1]))


def test_default_config_builds_data_initialization_and_density_contracts() -> None:
    config = load_config(ROOT / "configs" / "armgs_default.yaml")

    initialization = build_initialization_config(config)
    assert initialization.sh_degree == 3
    assert initialization.initial_opacity == 0.1
    assert initialization.voxel_size == 0.1

    sky = build_sky(config)
    assert sky.resolution == (256, 256)

    policy = build_density_policy(config, scene_scale=10.0)
    assert not policy.schedule.is_due(500)
    assert policy.schedule.is_due(600)
    assert policy.schedule.is_due(14_900)
    assert not policy.schedule.is_due(15_000)
    assert policy.thresholds.position_gradient_threshold == 0.0002
    assert policy.thresholds.split_scale_threshold == 0.1
    assert policy.thresholds.prune_opacity_threshold == 0.005
    assert policy.thresholds.max_screen_radius == 20.0
    assert policy.thresholds.max_world_scale == 1.0
    assert policy.thresholds.prune_large_after_step == 3_000
    assert policy.thresholds.minimum_gaussians == 0

    sampler = build_sampler(config, dataset_size=5)
    assert sampler.dataset_size == 5
    assert sampler.seed == 0
    assert sampler.shuffle
