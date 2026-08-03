from __future__ import annotations

import pytest
import torch

from armgs.encodings import HashGridEncoder


def _make_encoder() -> HashGridEncoder:
    return HashGridEncoder(
        num_levels=3,
        features_per_level=2,
        log2_hashmap_size=6,
        base_resolution=4,
        max_resolution=16,
        aabb_min=(0.0, 0.0, 0.0),
        aabb_max=(1.0, 1.0, 1.0),
    )


def test_chunked_forward_matches_dense_output_and_gradients() -> None:
    torch.manual_seed(7)
    dense_encoder = _make_encoder()
    chunked_encoder = _make_encoder()
    chunked_encoder.load_state_dict(dense_encoder.state_dict())

    dense_positions = torch.rand(11, 3, requires_grad=True)
    chunked_positions = dense_positions.detach().clone().requires_grad_(True)
    weights = torch.randn(11, dense_encoder.output_dim)

    dense_output = dense_encoder(dense_positions)
    chunked_output = chunked_encoder(chunked_positions, chunk_size=3)
    torch.testing.assert_close(chunked_output, dense_output)

    (dense_output * weights).sum().backward()
    (chunked_output * weights).sum().backward()
    torch.testing.assert_close(chunked_positions.grad, dense_positions.grad)
    torch.testing.assert_close(chunked_encoder.embeddings.grad, dense_encoder.embeddings.grad)


def test_visible_subset_matches_dense_selection_and_gradients() -> None:
    torch.manual_seed(11)
    dense_encoder = _make_encoder()
    visible_encoder = _make_encoder()
    visible_encoder.load_state_dict(dense_encoder.state_dict())

    dense_positions = torch.rand(13, 3, requires_grad=True)
    visible_positions = dense_positions.detach().clone().requires_grad_(True)
    visible_indices = torch.tensor([9, 1, 7, 3], dtype=torch.long)
    weights = torch.randn(visible_indices.numel(), dense_encoder.output_dim)

    expected = dense_encoder(dense_positions)[visible_indices]
    actual = visible_encoder.forward_visible(visible_positions, visible_indices, chunk_size=2)
    torch.testing.assert_close(actual, expected)

    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    torch.testing.assert_close(visible_positions.grad, dense_positions.grad)
    torch.testing.assert_close(visible_encoder.embeddings.grad, dense_encoder.embeddings.grad)


def test_visible_keyword_boolean_mask_and_empty_selection_are_safe() -> None:
    encoder = _make_encoder()
    positions = torch.rand(5, 3, requires_grad=True)
    mask = torch.tensor([True, False, True, False, False])

    masked = encoder(positions, visible_indices=mask, chunk_size=1)
    torch.testing.assert_close(masked, encoder(positions)[mask])

    empty = encoder.forward_visible(positions, torch.empty(0, dtype=torch.long), chunk_size=2)
    assert empty.shape == (0, encoder.output_dim)
    empty.sum().backward()
    assert positions.grad is not None
    torch.testing.assert_close(positions.grad, torch.zeros_like(positions))


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_chunk_size_must_be_positive(chunk_size: int) -> None:
    encoder = _make_encoder()
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        encoder(torch.rand(2, 3), chunk_size=chunk_size)


def test_chunk_size_must_be_an_integer() -> None:
    encoder = _make_encoder()
    with pytest.raises(TypeError, match="chunk_size must be an integer"):
        encoder(torch.rand(2, 3), chunk_size=1.5)  # type: ignore[arg-type]
