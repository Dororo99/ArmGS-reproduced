"""Dynamic actor refinement from ArmGS equations (7) and (8)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .encodings import SinusoidalEncoder
from .networks import MLP


@dataclass(frozen=True)
class ActorDeformation:
    means: Tensor
    sh_coefficients: Tensor
    delta_means: Tensor
    delta_sh: Tensor


class ActorDeformationRefiner(nn.Module):
    """Shared lightweight spatial-temporal actor deformation network.

    The paper's default sinusoidal position path is implemented here. The
    class-wise hash-grid hypernetwork is left as an optional future backend
    because its parameterization is not specified by the paper.
    """

    def __init__(
        self,
        sh_degree: int,
        *,
        hidden_dim: int = 64,
        position_frequencies: int = 6,
        time_frequencies: int = 6,
        encoder_layers: int = 2,
        head_layers: int = 2,
    ) -> None:
        super().__init__()
        if sh_degree < 0:
            raise ValueError("sh_degree cannot be negative")
        self.sh_degree = sh_degree
        self.sh_coefficient_count = (sh_degree + 1) ** 2
        self.sh_dim = self.sh_coefficient_count * 3
        self.position_encoder = SinusoidalEncoder(3, position_frequencies)
        self.time_encoder = SinusoidalEncoder(1, time_frequencies)
        encoder_input_dim = (
            self.position_encoder.output_dim + self.time_encoder.output_dim + self.sh_dim
        )
        self.spatial_temporal_encoder = MLP(
            encoder_input_dim,
            hidden_dim,
            hidden_dim,
            encoder_layers,
        )
        self.position_head = MLP(hidden_dim, hidden_dim, 3, head_layers)
        self.sh_head = MLP(hidden_dim, hidden_dim, self.sh_dim, head_layers)
        self._initialize_no_deformation()

    def _initialize_no_deformation(self) -> None:
        for head in (self.position_head, self.sh_head):
            nn.init.zeros_(head.final_layer.weight)
            nn.init.zeros_(head.final_layer.bias)

    @staticmethod
    def _expand_timestamps(timestamps: Tensor, count: int) -> Tensor:
        if timestamps.ndim == 0:
            return timestamps.reshape(1, 1).expand(count, 1)
        if timestamps.ndim == 1:
            if timestamps.numel() == 1:
                return timestamps.reshape(1, 1).expand(count, 1)
            if timestamps.numel() == count:
                return timestamps.reshape(count, 1)
        if timestamps.shape == (1, 1):
            return timestamps.expand(count, 1)
        if timestamps.shape == (count, 1):
            return timestamps
        raise ValueError("timestamps must be scalar, [1], [N], [1,1], or [N,1]")

    def predict_offsets(
        self, means: Tensor, sh_coefficients: Tensor, timestamps: Tensor
    ) -> tuple[Tensor, Tensor]:
        if means.ndim != 2 or means.shape[-1] != 3:
            raise ValueError("means must have shape [N,3]")
        expected_sh_shape = (means.shape[0], self.sh_coefficient_count, 3)
        if sh_coefficients.shape != expected_sh_shape:
            raise ValueError(
                f"sh_coefficients must have shape {expected_sh_shape}, "
                f"got {tuple(sh_coefficients.shape)}"
            )
        expanded_time = self._expand_timestamps(timestamps, means.shape[0]).to(means)
        features = torch.cat(
            (
                self.position_encoder(means),
                self.time_encoder(expanded_time),
                # Keep the feature width explicit: reshape(0, -1) is
                # ambiguous after density pruning removes every Gaussian
                # from an actor.
                sh_coefficients.reshape(means.shape[0], self.sh_dim),
            ),
            dim=-1,
        )
        representation = self.spatial_temporal_encoder(features)
        delta_means = self.position_head(representation)
        delta_sh = self.sh_head(representation).reshape(expected_sh_shape)
        return delta_means, delta_sh

    def forward(
        self, means: Tensor, sh_coefficients: Tensor, timestamps: Tensor
    ) -> ActorDeformation:
        delta_means, delta_sh = self.predict_offsets(
            means, sh_coefficients, timestamps
        )
        return ActorDeformation(
            means=means + delta_means,
            sh_coefficients=sh_coefficients + delta_sh,
            delta_means=delta_means,
            delta_sh=delta_sh,
        )
