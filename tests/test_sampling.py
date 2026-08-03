from __future__ import annotations

import pytest
import torch

from armgs.sampling import StatefulShuffleSampler


def test_same_seed_produces_same_epoch_orders_without_global_rng_mutation() -> None:
    torch.manual_seed(123)
    expected_random = torch.rand(4)
    torch.manual_seed(123)

    left = StatefulShuffleSampler(9, seed=17)
    right = StatefulShuffleSampler(9, seed=17)
    assert list(left) == list(right)
    assert left.epoch == right.epoch == 1
    torch.testing.assert_close(torch.rand(4), expected_random)

    second_left = list(left)
    second_right = list(right)
    assert second_left == second_right
    assert second_left != list(range(9))


def test_mid_epoch_checkpoint_resumes_at_exact_next_index() -> None:
    sampler = StatefulShuffleSampler(11, seed=5)
    iterator = iter(sampler)
    prefix = [next(iterator) for _ in range(4)]
    assert len(set(prefix)) == 4
    state = sampler.state_dict()
    expected_remaining = list(iterator)

    restored = StatefulShuffleSampler(11, seed=5)
    restored.load_state_dict(state)
    assert list(restored) == expected_remaining
    assert restored.epoch == 1
    assert restored.cursor == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA state mapping test")
def test_sampler_accepts_cuda_mapped_order_state() -> None:
    sampler = StatefulShuffleSampler(5, seed=2)
    iterator = iter(sampler)
    next(iterator)
    state = sampler.state_dict()
    state["order"] = state["order"].cuda()

    restored = StatefulShuffleSampler(5, seed=2)
    restored.load_state_dict(state)
    assert restored.state_dict()["order"].device.type == "cpu"
    assert list(restored) == list(iterator)


def test_sequential_mode_and_exhausted_checkpoint_advance_cleanly() -> None:
    sampler = StatefulShuffleSampler(4, shuffle=False)
    iterator = iter(sampler)
    assert [next(iterator) for _ in range(4)] == [0, 1, 2, 3]
    state = sampler.state_dict()
    assert state["cursor"] == 4

    restored = StatefulShuffleSampler(4, shuffle=False)
    restored.load_state_dict(state)
    assert list(restored) == []
    assert restored.epoch == 1
    assert list(restored) == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "state_change, message",
    [
        ({"dataset_size": 4}, "dataset_size"),
        ({"seed": 8}, "seed"),
        ({"cursor": 6}, "cursor"),
        ({"order": torch.tensor([0, 0, 1, 2, 3])}, "permutation"),
    ],
)
def test_incompatible_or_corrupt_state_is_rejected(
    state_change: dict[str, object], message: str
) -> None:
    sampler = StatefulShuffleSampler(5, seed=7)
    state = sampler.state_dict()
    state.update(state_change)

    with pytest.raises(ValueError, match=message):
        sampler.load_state_dict(state)
