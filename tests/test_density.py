from __future__ import annotations

import math

import pytest
import torch

from armgs.density import (
    DensificationSchedule,
    DensityControlThresholds,
    GaussianDensityPolicy,
    GaussianDensityStats,
    GaussianTopologyUpdatePlan,
    apply_gaussian_topology_update,
)
from armgs.scene import LearnableGaussianSet
from armgs.structures import GaussianSet


def make_module() -> LearnableGaussianSet:
    count = 4
    return LearnableGaussianSet(
        GaussianSet(
            means=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0],
                ]
            ),
            quaternions=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]]
            ).expand(count, -1).clone(),
            scales=torch.tensor(
                [
                    [0.1, 0.1, 0.1],
                    [2.0, 2.0, 2.0],
                    [0.3, 0.3, 0.3],
                    [0.4, 0.4, 0.4],
                ]
            ),
            opacities=torch.tensor(
                [[0.9], [0.8], [0.7], [0.01]]
            ),
            sh_coefficients=torch.arange(
                count * 3, dtype=torch.float32
            ).reshape(count, 1, 3),
            group_ids=torch.tensor([10, 11, 12, 13]),
        )
    )


def thresholds() -> DensityControlThresholds:
    return DensityControlThresholds(
        position_gradient_threshold=0.5,
        split_scale_threshold=1.0,
        prune_opacity_threshold=0.05,
        split_children=2,
        split_scale_reduction=2.0,
        opacity_reset_value=0.2,
        minimum_gaussians=1,
    )


def manual_plan(
    *, opacity_reset_value: float | None = 0.2
) -> GaussianTopologyUpdatePlan:
    return GaussianTopologyUpdatePlan(
        source_count=4,
        retain_indices=torch.tensor([0, 2]),
        duplicate_indices=torch.tensor([0]),
        split_indices=torch.tensor([1]),
        split_offsets=torch.tensor(
            [[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]]
        ),
        split_scale_reduction=2.0,
        opacity_reset_value=opacity_reset_value,
    )


def test_published_schedule_is_inclusive_and_unscheduled_plan_is_noop() -> None:
    schedule = DensificationSchedule()
    assert schedule.is_due(500)
    assert schedule.is_due(600)
    assert schedule.is_due(15_000)
    assert not schedule.is_due(499)
    assert not schedule.is_due(501)
    assert not schedule.is_due(15_001)

    module = make_module()
    policy = GaussianDensityPolicy(thresholds())
    plan = policy.make_plan(module, None, step=499)
    assert plan.is_noop

    original_parameters = tuple(module.parameters())
    optimizer = torch.optim.Adam(module.parameters(), lr=1.0e-3)
    result = apply_gaussian_topology_update(
        module, plan, optimizer=optimizer
    )
    assert result.optimizer_migrated
    assert not result.topology_changed
    assert tuple(module.parameters()) == original_parameters


def test_policy_separates_duplicate_split_and_prune_deterministically() -> None:
    first = make_module()
    second = make_module()
    stats = GaussianDensityStats(
        position_gradient_norm=torch.tensor([1.0, 1.0, 0.0, 1.0])
    )
    policy = GaussianDensityPolicy(thresholds())
    first_plan = policy.make_plan(
        first,
        stats,
        step=500,
        generator=torch.Generator().manual_seed(1234),
    )
    second_plan = policy.make_plan(
        second,
        stats,
        step=500,
        generator=torch.Generator().manual_seed(1234),
    )

    torch.testing.assert_close(
        first_plan.retain_indices, torch.tensor([0, 2])
    )
    torch.testing.assert_close(
        first_plan.duplicate_indices, torch.tensor([0])
    )
    torch.testing.assert_close(
        first_plan.split_indices, torch.tensor([1])
    )
    torch.testing.assert_close(
        first_plan.split_offsets, second_plan.split_offsets
    )
    assert first_plan.pruned_count == 1
    assert first_plan.opacity_reset_value is None


def test_explicit_plan_updates_raw_shapes_values_groups_and_opacity() -> None:
    module = make_module()
    old_sh = module.sh_coefficients.detach().clone()
    result = apply_gaussian_topology_update(
        module, manual_plan()
    )

    assert module.count == 5
    assert result.old_count == 4
    assert result.new_count == 5
    assert result.retained_count == 2
    assert result.duplicated_count == 1
    assert result.split_parent_count == 1
    assert result.split_child_count == 2
    assert result.pruned_count == 1
    torch.testing.assert_close(
        result.source_indices, torch.tensor([0, 2, 0, 1, 1])
    )
    torch.testing.assert_close(
        result.new_row_mask,
        torch.tensor([False, False, True, True, True]),
    )
    torch.testing.assert_close(
        module.means.detach(),
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [6.0, 7.0, 8.0],
                [0.0, 0.0, 0.0],
                [4.0, 4.0, 5.0],
                [2.0, 4.0, 5.0],
            ]
        ),
    )
    expected_scales = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [0.3, 0.3, 0.3],
            [0.1, 0.1, 0.1],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    torch.testing.assert_close(
        module.log_scales.detach().exp(), expected_scales
    )
    torch.testing.assert_close(
        module.opacity_logits.detach().sigmoid(),
        torch.full((5, 1), 0.2),
    )
    torch.testing.assert_close(
        module.sh_coefficients.detach(),
        old_sh.index_select(0, result.source_indices),
    )
    torch.testing.assert_close(
        module._group_ids, torch.tensor([10, 12, 10, 11, 11])
    )


def _populate_adam_state(
    optimizer: torch.optim.Adam,
    modules: list[LearnableGaussianSet],
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = sum(
        parameter.sum()
        for module in modules
        for parameter in module.parameters()
    )
    loss.backward()
    optimizer.step()


def _fill_row_state(
    optimizer: torch.optim.Adam,
    parameter: torch.nn.Parameter,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = optimizer.state[parameter]
    rows = torch.arange(
        parameter.shape[0], dtype=parameter.dtype
    )
    expand_shape = (
        parameter.shape[0],
        *([1] * (parameter.ndim - 1)),
    )
    exp_avg = rows.reshape(expand_shape).expand_as(parameter).clone()
    exp_avg_sq = (rows + 10).reshape(expand_shape).expand_as(
        parameter
    ).clone()
    state["exp_avg"] = exp_avg
    state["exp_avg_sq"] = exp_avg_sq
    return exp_avg, exp_avg_sq, state["step"].clone()


def test_adam_state_migration_preserves_kept_rows_and_zeros_new_rows() -> None:
    background = make_module()
    actor = make_module()
    optimizer = torch.optim.Adam(
        [*background.parameters(), *actor.parameters()],
        lr=1.0e-3,
    )
    _populate_adam_state(optimizer, [background, actor])

    old_background_means = background.means
    old_actor_means = actor.means
    means_avg, means_sq, old_step = _fill_row_state(
        optimizer, old_background_means
    )
    opacity_avg, _, _ = _fill_row_state(
        optimizer, background.opacity_logits
    )

    result = apply_gaussian_topology_update(
        background,
        manual_plan(),
        optimizer=optimizer,
    )
    assert result.optimizer_migrated
    assert background.count == 5
    assert actor.count == 4
    assert any(
        parameter is old_actor_means
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    assert not any(
        parameter is old_background_means
        for group in optimizer.param_groups
        for parameter in group["params"]
    )

    means_state = optimizer.state[background.means]
    torch.testing.assert_close(
        means_state["exp_avg"][:2],
        means_avg.index_select(0, torch.tensor([0, 2])),
    )
    torch.testing.assert_close(
        means_state["exp_avg_sq"][:2],
        means_sq.index_select(0, torch.tensor([0, 2])),
    )
    torch.testing.assert_close(
        means_state["exp_avg"][2:],
        torch.zeros_like(means_state["exp_avg"][2:]),
    )
    torch.testing.assert_close(means_state["step"], old_step)

    opacity_state = optimizer.state[background.opacity_logits]
    assert torch.count_nonzero(opacity_avg) > 0
    assert torch.count_nonzero(opacity_state["exp_avg"]) == 0
    assert torch.count_nonzero(opacity_state["exp_avg_sq"]) == 0

    actor_plan = GaussianTopologyUpdatePlan(
        source_count=4,
        retain_indices=torch.tensor([0, 1, 2]),
        duplicate_indices=torch.empty(0, dtype=torch.long),
        split_indices=torch.empty(0, dtype=torch.long),
        split_offsets=torch.empty(0, 2, 3),
    )
    actor_result = apply_gaussian_topology_update(
        actor, actor_plan, optimizer=optimizer
    )
    assert actor_result.optimizer_migrated
    assert actor.count == 3
    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert {
        id(parameter)
        for parameter in [*background.parameters(), *actor.parameters()]
    }.issubset(optimized_ids)


def test_opacity_only_reset_replaces_one_parameter_and_clears_its_state() -> None:
    module = make_module()
    optimizer = torch.optim.Adam(module.parameters(), lr=1.0e-3)
    _populate_adam_state(optimizer, [module])
    old_parameters = {
        name: parameter
        for name, parameter in module.named_parameters()
    }
    plan = GaussianTopologyUpdatePlan.no_op(
        module.count,
        device=module.means.device,
        dtype=module.means.dtype,
    )
    plan = GaussianTopologyUpdatePlan(
        source_count=plan.source_count,
        retain_indices=plan.retain_indices,
        duplicate_indices=plan.duplicate_indices,
        split_indices=plan.split_indices,
        split_offsets=plan.split_offsets,
        opacity_reset_value=0.1,
    )
    result = apply_gaussian_topology_update(
        module, plan, optimizer=optimizer
    )

    assert set(result.parameter_replacements) == {"opacity_logits"}
    assert module.means is old_parameters["means"]
    assert module.opacity_logits is not old_parameters["opacity_logits"]
    torch.testing.assert_close(
        module.opacity_logits.sigmoid(),
        torch.full((4, 1), 0.1),
    )
    state = optimizer.state[module.opacity_logits]
    assert torch.count_nonzero(state["exp_avg"]) == 0
    assert torch.count_nonzero(state["exp_avg_sq"]) == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"position_gradient_threshold": float("nan")},
            "position_gradient_threshold",
        ),
        ({"prune_opacity_threshold": 1.1}, "must not exceed"),
        ({"split_children": 1}, "at least two"),
        ({"opacity_reset_value": 0.0}, "strictly between"),
    ],
)
def test_unpublished_threshold_contract_is_validated(
    kwargs: dict[str, float | int | None],
    match: str,
) -> None:
    values: dict[str, float | int | None] = {
        "position_gradient_threshold": 0.1,
        "split_scale_threshold": 0.2,
        "prune_opacity_threshold": 0.01,
        "split_children": 2,
        "split_scale_reduction": 1.6,
        "opacity_reset_value": None,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        DensityControlThresholds(**values)  # type: ignore[arg-type]
