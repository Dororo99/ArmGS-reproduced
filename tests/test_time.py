from __future__ import annotations

import pytest
import torch

from armgs.time import TimestampNormalizer


def test_large_integer_nanoseconds_are_normalized_before_float32_cast() -> None:
    timestamps = torch.tensor(
        [1_700_000_000_000_000_000, 1_700_000_000_050_000_000],
        dtype=torch.int64,
    )
    normalizer = TimestampNormalizer.from_timestamps(timestamps)
    reference = torch.zeros((), dtype=torch.float32)
    normalized = normalizer(timestamps, reference=reference)

    assert normalized.dtype == torch.float32
    torch.testing.assert_close(normalized, torch.tensor([-1.0, 1.0]))


def test_single_timestamp_maps_to_zero_and_extrapolation_is_not_clamped() -> None:
    normalizer = TimestampNormalizer.from_timestamps(torch.tensor([10]))
    torch.testing.assert_close(
        normalizer(torch.tensor(10)), torch.tensor(0.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        normalizer(torch.tensor(12)), torch.tensor(2.0, dtype=torch.float64)
    )


def test_invalid_timestamp_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        TimestampNormalizer(0.0, 0.0)
    with pytest.raises(ValueError, match="non-empty"):
        TimestampNormalizer.from_timestamps(torch.empty(0))
    with pytest.raises(ValueError, match="either reference"):
        TimestampNormalizer(0.0, 1.0)(
            torch.tensor(0.0), reference=torch.tensor(0.0), dtype=torch.float32
        )



def test_module_float_does_not_quantize_absolute_time_buffers() -> None:
    timestamps = torch.tensor(
        [1_700_000_000_000_000_000, 1_700_000_000_050_000_000],
        dtype=torch.int64,
    )
    normalizer = TimestampNormalizer.from_timestamps(timestamps).float()
    assert normalizer.origin.dtype == torch.float64
    assert normalizer.scale.dtype == torch.float64
    torch.testing.assert_close(
        normalizer(timestamps, dtype=torch.float32),
        torch.tensor([-1.0, 1.0]),
    )
