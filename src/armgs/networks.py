"""Small neural-network building blocks used by the paper modules."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, nn


class MLP(nn.Module):
    """An MLP where ``num_layers`` counts all linear layers.

    The ArmGS paper reports only the number of linear layers, so keeping this
    interpretation in one class avoids subtle off-by-one architecture changes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation_factory: Callable[[], nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("MLP dimensions must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be at least one")

        dimensions = [input_dim]
        if num_layers > 1:
            dimensions.extend([hidden_dim] * (num_layers - 1))
        dimensions.append(output_dim)

        self.layers = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1])
            for index in range(num_layers)
        )
        self.activation = activation_factory()

    @property
    def final_layer(self) -> nn.Linear:
        return self.layers[-1]

    def forward(self, inputs: Tensor) -> Tensor:
        values = inputs
        for layer in self.layers[:-1]:
            values = self.activation(layer(values))
        return self.final_layer(values)

