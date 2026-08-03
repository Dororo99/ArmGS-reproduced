from __future__ import annotations

import math

import torch

from armgs.actor import ActorDeformationRefiner
from armgs.geometry import (
    PoseTrajectory,
    rotate_points,
    transform_actor_gaussians,
)
from armgs.spherical_harmonics import spherical_harmonics_to_rgb
from armgs.structures import GaussianSet


def make_gaussians(count: int = 2, degree: int = 1) -> GaussianSet:
    return GaussianSet(
        means=torch.zeros(count, 3),
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(count, -1).clone(),
        scales=torch.ones(count, 3),
        opacities=torch.full((count, 1), 0.5),
        sh_coefficients=torch.randn(count, (degree + 1) ** 2, 3),
    )


def test_actor_refiner_starts_with_no_deformation() -> None:
    actor = make_gaussians()
    refiner = ActorDeformationRefiner(
        sh_degree=1,
        hidden_dim=16,
        position_frequencies=2,
        time_frequencies=2,
    )
    deformation = refiner(actor.means, actor.sh_coefficients, torch.tensor(0.25))

    torch.testing.assert_close(deformation.means, actor.means)
    torch.testing.assert_close(deformation.sh_coefficients, actor.sh_coefficients)
    deformation.means.sum().backward()
    assert refiner.position_head.final_layer.weight.grad is not None


def test_actor_pose_transform_uses_quaternion_composition() -> None:
    actor = make_gaussians(count=1, degree=0).with_updates(
        means=torch.tensor([[1.0, 0.0, 0.0]])
    )
    half_angle = math.pi / 4.0
    pose = torch.tensor([math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)])
    transformed = transform_actor_gaussians(
        actor, pose, torch.tensor([1.0, 2.0, 0.0])
    )

    torch.testing.assert_close(
        transformed.means, torch.tensor([[1.0, 3.0, 0.0]]), atol=1.0e-6, rtol=0.0
    )
    torch.testing.assert_close(transformed.quaternions, pose[None], atol=1.0e-6, rtol=0.0)


def test_pose_trajectory_linearly_interpolates_translation_and_slerps_rotation() -> None:
    trajectory = PoseTrajectory(
        timestamps=torch.tensor([0, 2]),
        quaternions=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        ),
        translations=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )
    pose = trajectory.interpolate(torch.tensor(0.5))
    rotated = rotate_points(pose.quaternions, torch.tensor([[1.0, 0.0, 0.0]]))

    torch.testing.assert_close(
        pose.translations, torch.tensor([[0.5, 0.0, 0.0]])
    )
    torch.testing.assert_close(
        rotated, torch.tensor([[0.70710678, 0.70710678, 0.0]]), atol=1.0e-5, rtol=0.0
    )


def test_pose_trajectory_preserves_large_integer_timestamp_differences() -> None:
    start = 1_700_000_000_000_000_000
    trajectory = PoseTrajectory(
        timestamps=torch.tensor([start, start + 2_000_000], dtype=torch.int64),
        quaternions=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        ),
        translations=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )
    query = torch.tensor(start + 500_000, dtype=torch.int64)
    pose = trajectory.interpolate(query)

    torch.testing.assert_close(pose.translations, torch.tensor([[0.5, 0.0, 0.0]]))


def test_degree_zero_sh_color() -> None:
    coefficients = torch.tensor([[[1.0, 2.0, 3.0]]])
    direction = torch.tensor([[0.0, 0.0, 1.0]])
    rgb = spherical_harmonics_to_rgb(
        coefficients, direction, degree=0, clamp_min=False
    )
    expected = 0.28209479177387814 * coefficients[:, 0] + 0.5
    torch.testing.assert_close(rgb, expected)



def test_pose_module_float_preserves_absolute_timestamp_precision() -> None:
    start = 1_700_000_000_000_000_000
    trajectory = PoseTrajectory(
        timestamps=torch.tensor([start, start + 2_000_000], dtype=torch.int64),
        quaternions=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        ),
        translations=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    ).float()
    assert trajectory.timestamps.dtype == torch.float64
    pose = trajectory.interpolate(
        torch.tensor(start + 500_000, dtype=torch.int64)
    )
    torch.testing.assert_close(pose.translations, torch.tensor([[0.5, 0.0, 0.0]]))
