"""Numerically stable timestamp contracts for dynamic-scene conditioning."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TimestampNormalizer(nn.Module):
    """Map absolute timestamps to a stable, dimensionless interval.

    Dataset timestamps are often stored as large integer microseconds or
    nanoseconds. Subtracting them after a float32 cast can erase frame-to-frame
    differences. This module therefore keeps origin and scale in float64,
    performs centering in float64, and only then casts the normalized result to
    the requested training dtype/device.

    By default the observed timestamp interval maps to [-1, 1]. Values outside
    the fitted interval are intentionally not clamped so interpolation or
    extrapolation policy remains explicit at the caller.
    """

    def __init__(
        self,
        origin: float | Tensor,
        scale: float | Tensor,
        *,
        output_center: float = 0.0,
        output_half_range: float = 1.0,
    ) -> None:
        super().__init__()
        origin_tensor = torch.as_tensor(origin, dtype=torch.float64).reshape(())
        scale_tensor = torch.as_tensor(scale, dtype=torch.float64).reshape(())
        if not torch.isfinite(origin_tensor):
            raise ValueError("timestamp origin must be finite")
        if not torch.isfinite(scale_tensor) or scale_tensor <= 0:
            raise ValueError("timestamp scale must be finite and positive")
        if not torch.isfinite(torch.tensor(output_center)):
            raise ValueError("output_center must be finite")
        if (
            not torch.isfinite(torch.tensor(output_half_range))
            or output_half_range <= 0
        ):
            raise ValueError("output_half_range must be finite and positive")
        self.register_buffer("origin", origin_tensor)
        self.register_buffer("scale", scale_tensor)
        self.output_center = float(output_center)
        self.output_half_range = float(output_half_range)

    def _apply(self, fn):  # type: ignore[no-untyped-def]
        # Module.float()/half() must not quantize absolute time before centering.
        origin = self.origin
        scale = self.scale
        result = super()._apply(fn)
        device = self.origin.device
        self._buffers["origin"] = origin.to(device=device, dtype=torch.float64)
        self._buffers["scale"] = scale.to(device=device, dtype=torch.float64)
        return result

    @classmethod
    def from_timestamps(cls, timestamps: Tensor) -> "TimestampNormalizer":
        """Fit a normalizer whose min/max map to -1/+1."""

        if timestamps.numel() == 0:
            raise ValueError("timestamps must be non-empty")
        values = timestamps.detach().to(dtype=torch.float64)
        if not torch.isfinite(values).all():
            raise ValueError("timestamps must be finite")
        minimum = values.min()
        maximum = values.max()
        span = maximum - minimum
        if span == 0:
            return cls(minimum, 1.0)
        midpoint = minimum + span / 2.0
        return cls(midpoint, span / 2.0)

    def forward(
        self,
        timestamps: Tensor,
        *,
        reference: Tensor | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Normalize timestamps, optionally matching a reference tensor."""

        if reference is not None and (dtype is not None or device is not None):
            raise ValueError("use either reference or explicit dtype/device")
        values = timestamps.to(device=self.origin.device, dtype=torch.float64)
        if not torch.isfinite(values).all():
            raise ValueError("timestamps must be finite")
        normalized = (values - self.origin) / self.scale
        normalized = normalized * self.output_half_range + self.output_center
        if reference is not None:
            return normalized.to(device=reference.device, dtype=reference.dtype)
        if dtype is not None or device is not None:
            return normalized.to(dtype=dtype, device=device)
        return normalized


__all__ = ["TimestampNormalizer"]
