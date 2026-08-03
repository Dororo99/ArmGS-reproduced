from __future__ import annotations

import pytest
import torch

from armgs.appearance import (
    GlobalAffine,
    GlobalImageAppearanceRefiner,
    LocalGaussianAppearanceRefiner,
    NearestFrameLookup,
    ViewpointEncoder,
)
from armgs.encodings import HashGridEncoder


def make_hash_encoder() -> HashGridEncoder:
    return HashGridEncoder(
        num_levels=2,
        features_per_level=2,
        log2_hashmap_size=5,
        base_resolution=4,
        max_resolution=8,
    )


def test_nearest_frame_is_restricted_to_same_camera() -> None:
    lookup = NearestFrameLookup(
        camera_ids=torch.tensor([0, 1, 0, 1]),
        timestamps=torch.tensor([0.0, 0.0, 2.0, 3.0]),
    )
    rows = lookup(torch.tensor([0, 1]), torch.tensor([1.6, 2.5]))
    torch.testing.assert_close(rows, torch.tensor([2, 3]))

    with pytest.raises(ValueError, match="no training frame"):
        lookup(torch.tensor(4), torch.tensor(0.0))


def test_nearest_frame_module_float_preserves_absolute_timestamps() -> None:
    start = 1_700_000_000_000_000_000
    lookup = NearestFrameLookup(
        camera_ids=torch.tensor([0, 0]),
        timestamps=torch.tensor(
            [start, start + 50_000_000], dtype=torch.int64
        ),
    ).float()
    assert lookup.timestamps.dtype == torch.float64
    row = lookup(
        torch.tensor(0),
        torch.tensor(start + 40_000_000, dtype=torch.int64),
    )
    torch.testing.assert_close(row, torch.tensor(1))


def test_local_refiner_starts_as_exact_identity() -> None:
    refiner = LocalGaussianAppearanceRefiner(
        make_hash_encoder(), frame_embedding_dim=5, hidden_dim=16, num_layers=3
    )
    positions = torch.rand(7, 3) * 2.0 - 1.0
    colors = torch.rand(7, 3)
    embedding = torch.randn(5)
    affine = refiner.predict_affine(positions, colors, embedding)
    output = refiner(positions, colors, embedding)

    torch.testing.assert_close(affine.scale, torch.ones_like(affine.scale))
    torch.testing.assert_close(affine.bias, torch.zeros_like(affine.bias))
    torch.testing.assert_close(output, colors)
    output.sum().backward()
    assert refiner.affine_learner.final_layer.weight.grad is not None


def test_global_refiner_starts_as_exact_identity() -> None:
    encoder = ViewpointEncoder(position_frequencies=2, direction_frequencies=2)
    refiner = GlobalImageAppearanceRefiner(
        encoder, frame_embedding_dim=5, hidden_dim=16, num_layers=4
    )
    image = torch.rand(4, 6, 3)
    output = refiner(
        image,
        torch.randn(5),
        torch.tensor([0.1, -0.2, 0.3]),
        torch.tensor([0.0, 0.0, -1.0]),
    )

    torch.testing.assert_close(output, image)
    output.sum().backward()
    assert refiner.affine_learner.final_layer.weight.grad is not None


def test_global_matrix_uses_column_color_convention() -> None:
    image = torch.tensor([[[1.0, 2.0, 3.0]]])
    affine = GlobalAffine(
        matrix=torch.tensor(
            [[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]]
        ),
        bias=torch.zeros(1, 3),
    )
    transformed = GlobalImageAppearanceRefiner.apply_affine(image, affine)
    torch.testing.assert_close(transformed, torch.tensor([[[2.0, 3.0, 1.0]]]))



def test_optional_affine_bounds_remain_identity_centered_and_finite() -> None:
    local = LocalGaussianAppearanceRefiner(
        make_hash_encoder(),
        frame_embedding_dim=3,
        hidden_dim=8,
        scale_delta_limit=1.0,
        bias_limit=0.25,
    )
    with torch.no_grad():
        local.affine_learner.final_layer.bias.fill_(100.0)
    local_affine = local.predict_affine(
        torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(3)
    )
    assert torch.all(local_affine.scale <= 2.0)
    assert torch.all(local_affine.scale > 1.0)
    assert torch.all(local_affine.bias <= 0.25)

    global_refiner = GlobalImageAppearanceRefiner(
        ViewpointEncoder(position_frequencies=1, direction_frequencies=1),
        frame_embedding_dim=3,
        hidden_dim=8,
        matrix_delta_limit=0.5,
        bias_limit=0.25,
    )
    with torch.no_grad():
        global_refiner.affine_learner.final_layer.bias.fill_(-100.0)
    global_affine = global_refiner.predict_affine(
        torch.zeros(3),
        torch.zeros(3),
        torch.tensor([0.0, 0.0, 1.0]),
    )
    identity = torch.eye(3).unsqueeze(0)
    assert torch.all(torch.abs(global_affine.matrix - identity) <= 0.5)
    assert torch.all(torch.abs(global_affine.bias) <= 0.25)
