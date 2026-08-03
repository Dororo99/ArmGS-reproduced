"""Deterministic index sampling with exact mid-epoch checkpoint resume."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from numbers import Integral
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Sampler


class StatefulShuffleSampler(Sampler[int]):
    """A single-process sampler whose permutation and cursor are checkpointable.

    The sampler owns a CPU permutation for the current epoch. Its cursor is
    advanced before yielding each index, so a checkpoint taken immediately
    after a batch resumes at the next unseen sample. This provides deterministic
    index order across process restarts without mutating PyTorch's global RNG.

    DataLoader worker RNG and in-flight prefetch queues are intentionally out of
    scope; exact resume requires num_workers=0 or application-managed worker
    state.
    """

    _STATE_VERSION = 1

    def __init__(
        self,
        dataset_size: int,
        *,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        super().__init__(None)
        if (
            isinstance(dataset_size, bool)
            or not isinstance(dataset_size, Integral)
            or dataset_size <= 0
        ):
            raise ValueError("dataset_size must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("seed must be an integer")
        self.dataset_size = int(dataset_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.cursor = 0
        self._order = self._order_for_epoch(self.epoch)

    def _order_for_epoch(self, epoch: int) -> Tensor:
        if not self.shuffle:
            return torch.arange(self.dataset_size, dtype=torch.long)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + epoch)
        return torch.randperm(
            self.dataset_size, generator=generator, dtype=torch.long
        )

    def __iter__(self) -> Iterator[int]:
        while self.cursor < self.dataset_size:
            index = int(self._order[self.cursor].item())
            self.cursor += 1
            yield index
        self.epoch += 1
        self.cursor = 0
        self._order = self._order_for_epoch(self.epoch)

    def __len__(self) -> int:
        return self.dataset_size - self.cursor

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self._STATE_VERSION,
            "dataset_size": self.dataset_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "order": self._order.clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        version = int(state.get("version", -1))
        if version != self._STATE_VERSION:
            raise ValueError(f"unsupported sampler state version {version}")
        if int(state["dataset_size"]) != self.dataset_size:
            raise ValueError("sampler dataset_size does not match checkpoint")
        if int(state["seed"]) != self.seed:
            raise ValueError("sampler seed does not match checkpoint")
        if bool(state["shuffle"]) != self.shuffle:
            raise ValueError("sampler shuffle mode does not match checkpoint")

        epoch = int(state["epoch"])
        cursor = int(state["cursor"])
        if epoch < 0:
            raise ValueError("sampler epoch cannot be negative")
        if not 0 <= cursor <= self.dataset_size:
            raise ValueError("sampler cursor is out of range")

        order = state["order"]
        if not isinstance(order, Tensor):
            raise ValueError("sampler order must be a tensor")
        order = order.detach().to(device="cpu", dtype=torch.long)
        if order.shape != (self.dataset_size,):
            raise ValueError("sampler order has the wrong shape")
        expected = torch.arange(self.dataset_size, dtype=torch.long)
        if not torch.equal(torch.sort(order).values, expected):
            raise ValueError("sampler order must be a permutation")

        self.epoch = epoch
        self.cursor = cursor
        self._order = order.clone()


__all__ = ["StatefulShuffleSampler"]
