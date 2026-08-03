from __future__ import annotations

import torch

from armgs.actor import ActorDeformationRefiner
from armgs.appearance import (
    GlobalImageAppearanceRefiner,
    LocalGaussianAppearanceRefiner,
    ViewpointEncoder,
)
from armgs.encodings import HashGridEncoder


def test_all_three_refinement_levels_can_overfit_tiny_targets() -> None:
    torch.manual_seed(2)
    positions = torch.rand(6, 3) * 2.0 - 1.0
    colors = torch.rand(6, 3)
    embedding = torch.randn(5)

    local = LocalGaussianAppearanceRefiner(
        HashGridEncoder(
            num_levels=2,
            features_per_level=2,
            log2_hashmap_size=5,
            base_resolution=4,
            max_resolution=8,
        ),
        frame_embedding_dim=5,
        hidden_dim=16,
    )
    local_target = colors + 0.15 * positions
    optimizer = torch.optim.Adam(local.parameters(), lr=0.03)
    for _ in range(80):
        optimizer.zero_grad()
        loss = (local(positions, colors, embedding) - local_target).square().mean()
        loss.backward()
        optimizer.step()
    assert loss.detach() < 1.0e-4

    global_refiner = GlobalImageAppearanceRefiner(
        ViewpointEncoder(position_frequencies=2, direction_frequencies=2),
        frame_embedding_dim=5,
        hidden_dim=16,
    )
    image = torch.rand(4, 5, 3)
    target_matrix = torch.tensor(
        [[1.1, 0.05, 0.0], [0.0, 0.9, 0.02], [0.03, 0.0, 1.05]]
    )
    global_target = image @ target_matrix.T + torch.tensor([0.1, -0.05, 0.02])
    camera_center = torch.tensor([0.1, 0.2, 0.3])
    view_direction = torch.tensor([0.0, 0.0, -1.0])
    optimizer = torch.optim.Adam(global_refiner.parameters(), lr=0.03)
    for _ in range(80):
        optimizer.zero_grad()
        output = global_refiner(
            image, embedding, camera_center, view_direction
        )
        loss = (output - global_target).square().mean()
        loss.backward()
        optimizer.step()
    assert loss.detach() < 1.0e-4

    actor = ActorDeformationRefiner(
        sh_degree=1,
        hidden_dim=16,
        position_frequencies=2,
        time_frequencies=2,
    )
    sh_coefficients = torch.rand(6, 4, 3)
    target_means = positions * 1.1
    target_sh = sh_coefficients - 0.04
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.03)
    for _ in range(100):
        optimizer.zero_grad()
        deformation = actor(positions, sh_coefficients, torch.tensor(0.3))
        loss = (deformation.means - target_means).square().mean()
        loss = loss + (deformation.sh_coefficients - target_sh).square().mean()
        loss.backward()
        optimizer.step()
    assert loss.detach() < 1.0e-4
