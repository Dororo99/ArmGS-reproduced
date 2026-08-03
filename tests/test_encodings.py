from __future__ import annotations

import torch

from armgs.encodings import HashGridEncoder, SinusoidalEncoder


def test_sinusoidal_encoder_shape_and_input_prefix() -> None:
    encoder = SinusoidalEncoder(3, 4, include_input=True)
    inputs = torch.tensor([[0.1, -0.2, 0.3]])
    encoded = encoder(inputs)

    assert encoded.shape == (1, encoder.output_dim)
    torch.testing.assert_close(encoded[:, :3], inputs)


def test_hash_grid_is_differentiable_in_positions_and_table() -> None:
    encoder = HashGridEncoder(
        num_levels=2,
        features_per_level=2,
        log2_hashmap_size=5,
        base_resolution=4,
        max_resolution=8,
        aabb_min=(0.0, 0.0, 0.0),
        aabb_max=(1.0, 1.0, 1.0),
    )
    positions = torch.tensor(
        [[0.17, 0.41, 0.73], [0.82, 0.29, 0.56]], requires_grad=True
    )
    encoded = encoder(positions)

    assert encoded.shape == (2, 4)
    encoded.square().sum().backward()
    assert positions.grad is not None
    assert torch.isfinite(positions.grad).all()
    assert encoder.embeddings.grad is not None
    assert torch.isfinite(encoder.embeddings.grad).all()
