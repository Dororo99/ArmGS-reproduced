"""Position and time encodings used by ArmGS refinement modules."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral

import torch
from torch import Tensor, nn


class SinusoidalEncoder(nn.Module):
    """NeRF-style sinusoidal encoding for any final input dimension."""

    def __init__(
        self,
        input_dim: int,
        num_frequencies: int,
        *,
        include_input: bool = True,
        min_frequency_exp: float = 0.0,
        max_frequency_exp: float | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_frequencies < 0:
            raise ValueError("num_frequencies cannot be negative")
        if max_frequency_exp is None:
            max_frequency_exp = float(max(num_frequencies - 1, 0))

        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        if num_frequencies:
            exponents = torch.linspace(
                min_frequency_exp, max_frequency_exp, num_frequencies
            )
            frequencies = torch.pow(2.0, exponents) * math.pi
        else:
            frequencies = torch.empty(0)
        self.register_buffer("frequencies", frequencies, persistent=False)

    @property
    def output_dim(self) -> int:
        encoded = self.input_dim * self.num_frequencies * 2
        return encoded + (self.input_dim if self.include_input else 0)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected final dimension {self.input_dim}, got {inputs.shape[-1]}"
            )
        parts: list[Tensor] = []
        if self.include_input:
            parts.append(inputs)
        if self.num_frequencies:
            angles = inputs.unsqueeze(-1) * self.frequencies.to(inputs)
            parts.extend((angles.sin().flatten(-2), angles.cos().flatten(-2)))
        if not parts:
            return inputs.new_empty((*inputs.shape[:-1], 0))
        return torch.cat(parts, dim=-1)


class HashGridEncoder(nn.Module):
    """A differentiable, pure-PyTorch multi-resolution 3-D hash grid.

    This follows the Instant-NGP trilinear lookup idea while intentionally
    avoiding a tiny-cuda-nn dependency. It is a correct reference path and can
    later be replaced by a fused backend without changing tensor contracts.
    """

    _HASH_PRIMES = (1, 2_654_435_761, 805_459_861)

    def __init__(
        self,
        *,
        num_levels: int = 8,
        features_per_level: int = 2,
        log2_hashmap_size: int = 15,
        base_resolution: int = 16,
        max_resolution: int = 2048,
        aabb_min: Sequence[float] = (-1.0, -1.0, -1.0),
        aabb_max: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        if num_levels <= 0:
            raise ValueError("num_levels must be positive")
        if features_per_level <= 0:
            raise ValueError("features_per_level must be positive")
        if log2_hashmap_size <= 0:
            raise ValueError("log2_hashmap_size must be positive")
        if base_resolution <= 0 or max_resolution < base_resolution:
            raise ValueError("resolutions must satisfy 0 < base <= max")

        minimum = torch.as_tensor(aabb_min, dtype=torch.float32)
        maximum = torch.as_tensor(aabb_max, dtype=torch.float32)
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("AABB endpoints must each contain three values")
        if torch.any(maximum <= minimum):
            raise ValueError("aabb_max must be greater than aabb_min")

        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.hashmap_size = 1 << log2_hashmap_size
        if num_levels == 1:
            resolutions = torch.tensor([base_resolution], dtype=torch.long)
        else:
            scale = math.exp(
                math.log(max_resolution / base_resolution) / (num_levels - 1)
            )
            resolutions = torch.tensor(
                [math.floor(base_resolution * scale**level) for level in range(num_levels)],
                dtype=torch.long,
            )

        self.register_buffer("aabb_min", minimum)
        self.register_buffer("aabb_max", maximum)
        self.register_buffer("resolutions", resolutions, persistent=True)
        self.register_buffer(
            "corner_offsets",
            torch.tensor(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1],
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.embeddings = nn.Parameter(
            torch.empty(num_levels, self.hashmap_size, features_per_level)
        )
        nn.init.uniform_(self.embeddings, -1.0e-4, 1.0e-4)

    @property
    def output_dim(self) -> int:
        return self.num_levels * self.features_per_level

    def set_aabb(self, minimum: Tensor, maximum: Tensor) -> None:
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("AABB tensors must have shape [3]")
        if torch.any(maximum <= minimum):
            raise ValueError("maximum must be greater than minimum")
        self.aabb_min.copy_(minimum.to(self.aabb_min))
        self.aabb_max.copy_(maximum.to(self.aabb_max))

    def _hash(self, integer_coordinates: Tensor) -> Tensor:
        coordinates = integer_coordinates.to(torch.long)
        hashed = coordinates[..., 0] * self._HASH_PRIMES[0]
        hashed = torch.bitwise_xor(
            hashed, coordinates[..., 1] * self._HASH_PRIMES[1]
        )
        hashed = torch.bitwise_xor(
            hashed, coordinates[..., 2] * self._HASH_PRIMES[2]
        )
        return torch.remainder(hashed, self.hashmap_size)

    @staticmethod
    def _validate_chunk_size(chunk_size: int | None) -> int | None:
        if chunk_size is None:
            return None
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral):
            raise TypeError("chunk_size must be an integer or None")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        return int(chunk_size)

    def _forward_flat(self, flat_positions: Tensor) -> Tensor:
        """Encode a flat ``[N, 3]`` tensor without allocating output chunks."""

        minimum = self.aabb_min.to(flat_positions)
        maximum = self.aabb_max.to(flat_positions)
        normalized = (flat_positions - minimum) / (maximum - minimum)
        # Clamping makes the AABB behavior explicit and prevents the upper corner
        # from crossing into an additional cell at exactly one.
        normalized = normalized.clamp(0.0, 1.0 - 1.0e-6)

        level_features: list[Tensor] = []
        offsets = self.corner_offsets.to(device=flat_positions.device)
        for level, resolution_tensor in enumerate(self.resolutions):
            resolution = resolution_tensor.to(flat_positions)
            grid_position = normalized * resolution
            lower = torch.floor(grid_position).to(torch.long)
            fraction = grid_position - lower.to(grid_position)

            corners = lower[:, None, :] + offsets[None, :, :]
            indices = self._hash(corners)
            corner_features = self.embeddings[level][indices]
            per_axis_weights = torch.where(
                offsets[None, :, :].bool(),
                fraction[:, None, :],
                1.0 - fraction[:, None, :],
            )
            weights = per_axis_weights.prod(dim=-1, keepdim=True)
            level_features.append((corner_features * weights).sum(dim=1))

        return torch.cat(level_features, dim=-1)

    def _forward_flat_chunked(
        self, flat_positions: Tensor, chunk_size: int | None
    ) -> Tensor:
        chunk_size = self._validate_chunk_size(chunk_size)
        if chunk_size is None or flat_positions.shape[0] <= chunk_size:
            return self._forward_flat(flat_positions)
        return torch.cat(
            [
                self._forward_flat(flat_positions[start : start + chunk_size])
                for start in range(0, flat_positions.shape[0], chunk_size)
            ],
            dim=0,
        )

    def forward_visible(
        self,
        positions: Tensor,
        visible_indices: Tensor,
        *,
        chunk_size: int | None = None,
    ) -> Tensor:
        """Encode only selected positions as a flat ``[K, output_dim]`` tensor.

        ``positions`` may have any leading dimensions; ``visible_indices`` addresses
        their flattened order. A one-dimensional boolean mask is also accepted.
        Selection happens before hash-grid corner allocation, which is the important
        memory-saving property for large Gaussian sets.
        """

        if positions.shape[-1] != 3:
            raise ValueError("positions must have final dimension three")
        if not isinstance(visible_indices, Tensor):
            raise TypeError("visible_indices must be a tensor")
        if visible_indices.ndim != 1:
            raise ValueError("visible_indices must be one-dimensional")

        flat_positions = positions.reshape(-1, 3)
        if visible_indices.dtype == torch.bool:
            if visible_indices.numel() != flat_positions.shape[0]:
                raise ValueError(
                    "boolean visible_indices must match the flattened position count"
                )
            selected = flat_positions[visible_indices.to(device=flat_positions.device)]
        else:
            integer_dtypes = {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }
            if visible_indices.dtype not in integer_dtypes:
                raise TypeError("visible_indices must contain integers or booleans")
            indices = visible_indices.to(
                device=flat_positions.device, dtype=torch.long
            )
            if indices.numel() and (
                torch.any(indices < 0) or torch.any(indices >= flat_positions.shape[0])
            ):
                raise IndexError("visible_indices contains an out-of-range index")
            selected = torch.index_select(flat_positions, 0, indices)

        return self._forward_flat_chunked(selected, chunk_size)

    def forward(
        self,
        positions: Tensor,
        *,
        chunk_size: int | None = None,
        visible_indices: Tensor | None = None,
    ) -> Tensor:
        """Encode positions, optionally in chunks or after visible selection.

        The default call preserves the original leading dimensions. Supplying
        ``visible_indices`` returns the selected positions in flat index order, as
        documented by :meth:`forward_visible`.
        """

        if positions.shape[-1] != 3:
            raise ValueError("positions must have final dimension three")
        if visible_indices is not None:
            return self.forward_visible(
                positions, visible_indices, chunk_size=chunk_size
            )

        original_shape = positions.shape[:-1]
        flat_positions = positions.reshape(-1, 3)
        encoded = self._forward_flat_chunked(flat_positions, chunk_size)
        return encoded.reshape(*original_shape, self.output_dim)
