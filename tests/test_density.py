from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import pytest
import torch

from armgs.config import build_density_controller
from armgs.density import (
    DensificationSchedule,
    DensityControlThresholds,
    GaussianDensityPolicy,
    GaussianDensityStats,
    GaussianTopologyUpdatePlan,
    GsplatDensityController,
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
        minimum_gaussians=0,
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


def test_reference_schedule_uses_strict_bounds_and_global_modulo() -> None:
    schedule = DensificationSchedule()
    assert not schedule.is_due(500)
    assert schedule.is_due(600)
    assert schedule.is_due(14_900)
    assert not schedule.is_due(15_000)
    assert not schedule.is_due(499)
    assert not schedule.is_due(501)
    assert not schedule.is_due(15_001)
    assert [
        step for step in range(0, 15_101) if schedule.is_due(step)
    ] == list(range(600, 15_000, 100))

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
        step=600,
        generator=torch.Generator().manual_seed(1234),
    )
    second_plan = policy.make_plan(
        second,
        stats,
        step=600,
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


def test_opacity_pruning_uses_strict_reference_threshold() -> None:
    module = make_module()
    opacity = torch.tensor([[0.0049], [0.005], [0.01], [0.8]])
    with torch.no_grad():
        module.opacity_logits.copy_(torch.logit(opacity))
    represented_threshold = float(
        module.opacity_logits.sigmoid()[1].item()
    )
    policy = GaussianDensityPolicy(
        replace(
            thresholds(),
            position_gradient_threshold=10.0,
            prune_opacity_threshold=represented_threshold,
        )
    )
    plan = policy.make_plan(
        module,
        GaussianDensityStats(torch.zeros(module.count)),
        step=600,
    )

    torch.testing.assert_close(
        plan.retain_indices, torch.tensor([1, 2, 3])
    )
    assert plan.pruned_count == 1


def test_large_pruning_starts_after_3000_and_follows_densify_then_prune() -> None:
    module = make_module()
    with torch.no_grad():
        module.log_scales.copy_(
            torch.tensor(
                [[0.2, 0.2, 0.2], [2.0, 2.0, 2.0],
                 [4.0, 4.0, 4.0], [1.0, 1.0, 1.0]]
            ).log()
        )
        module.opacity_logits.fill_(torch.logit(torch.tensor(0.8)))
    policy = GaussianDensityPolicy(
        replace(
            thresholds(),
            prune_opacity_threshold=0.005,
            max_screen_radius=20.0,
            max_world_scale=1.5,
            prune_large_after_step=3_000,
        )
    )
    stats = GaussianDensityStats(
        position_gradient_norm=torch.tensor([1.0, 1.0, 1.0, 0.0]),
        max_screen_radius=torch.tensor([25.0, 25.0, 25.0, 1.0]),
    )

    at_boundary = policy.make_plan(
        module,
        stats,
        step=3_000,
        generator=torch.Generator().manual_seed(1),
    )
    torch.testing.assert_close(
        at_boundary.retain_indices, torch.tensor([0, 3])
    )
    torch.testing.assert_close(
        at_boundary.duplicate_indices, torch.tensor([0])
    )
    torch.testing.assert_close(
        at_boundary.split_indices, torch.tensor([1, 2])
    )

    after_boundary = policy.make_plan(
        module,
        stats,
        step=3_100,
        generator=torch.Generator().manual_seed(1),
    )
    # The screen-large clone parent is removed, but its new zero-radius clone
    # survives. The 2m split parent produces 1m children; the 4m parent would
    # still produce 2m children and is therefore fully pruned.
    torch.testing.assert_close(
        after_boundary.retain_indices, torch.tensor([3])
    )
    torch.testing.assert_close(
        after_boundary.duplicate_indices, torch.tensor([0])
    )
    torch.testing.assert_close(
        after_boundary.split_indices, torch.tensor([1])
    )
    assert after_boundary.pruned_count == 2


def test_actor_box_pruning_samples_rotated_support_after_large_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = LearnableGaussianSet(
        GaussianSet(
            means=torch.zeros(2, 3),
            # The first non-unit quaternion normalizes to a 90-degree z turn.
            quaternions=torch.tensor(
                [
                    [2.0**0.5, 0.0, 0.0, 2.0**0.5],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            ),
            scales=torch.full((2, 3), 0.1),
            opacities=torch.full((2, 1), 0.8),
            sh_coefficients=torch.zeros(2, 1, 3),
        )
    )
    policy = GaussianDensityPolicy(
        replace(
            thresholds(),
            position_gradient_threshold=10.0,
            prune_opacity_threshold=0.0,
            prune_large_after_step=3_000,
        ),
        actor_box_half_extents=(2.0, 0.5, 2.0),
    )
    stats = GaussianDensityStats(torch.zeros(module.count))
    calls = 0

    def fixed_normal(
        mean: torch.Tensor,
        std: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        assert mean.shape == std.shape == (2, 2, 3)
        assert generator is not None
        samples = torch.zeros_like(mean)
        # One of the first Gaussian's two support samples rotates from +x to
        # +y, crossing the y half-extent. Every sample of row one stays in.
        samples[0, 1, 0] = 1.0
        return samples

    monkeypatch.setattr(torch, "normal", fixed_normal)
    boundary = policy.make_plan(
        module,
        stats,
        step=3_000,
        generator=torch.Generator().manual_seed(4),
    )
    torch.testing.assert_close(
        boundary.retain_indices, torch.tensor([0, 1])
    )
    assert calls == 0

    after = policy.make_plan(
        module,
        stats,
        step=3_100,
        generator=torch.Generator().manual_seed(4),
    )
    torch.testing.assert_close(after.retain_indices, torch.tensor([1]))
    assert after.pruned_count == 1
    assert calls == 1

    background_policy = GaussianDensityPolicy(policy.thresholds)
    background = background_policy.make_plan(
        module,
        stats,
        step=3_100,
        generator=torch.Generator().manual_seed(4),
    )
    torch.testing.assert_close(
        background.retain_indices, torch.tensor([0, 1])
    )
    assert calls == 1


def test_density_controller_uses_planar_scaled_actor_bounds_only() -> None:
    config = {
        "optimization": {
            "densification": {
                "start_iteration": 500,
                "end_iteration": 15_000,
                "interval": 100,
                "position_gradient_threshold": 0.0002,
                "split_scale_fraction_of_scene": 0.01,
                "prune_opacity_threshold": 0.005,
                "split_children": 2,
                "split_scale_reduction": 1.6,
                "opacity_reset_value": 0.01,
                "prune_actor_outside_box": True,
            }
        }
    }
    background = make_module()
    actor_gaussians = make_module()
    actor = SimpleNamespace(
        actor_id=7,
        gaussians=actor_gaussians,
        dimensions_lwh=torch.tensor([4.0, 2.0, 1.5]),
    )
    scene = SimpleNamespace(background=background, actors=[actor])
    controller = build_density_controller(
        config,
        scene,
        scene_scale=20.0,
        actor_box_scale=2.0,
        group_scene_scales={-1: 20.0, 7: 3.0},
    )

    assert controller.policy_for(-1).actor_box_half_extents is None
    assert controller.policy_for(7).actor_box_half_extents == (
        4.0,
        2.0,
        0.75,
    )
    state = controller.state_dict()
    assert "actor_box_half_extents" not in state["policies"][-1]
    assert state["policies"][7]["actor_box_half_extents"] == (
        4.0,
        2.0,
        0.75,
    )


def test_reset_is_capped_and_disabled_at_densification_endpoint() -> None:
    module = make_module()
    policy = GaussianDensityPolicy(
        replace(
            thresholds(),
            position_gradient_threshold=10.0,
            prune_opacity_threshold=0.0,
        )
    )
    endpoint = policy.make_plan(
        module, None, step=15_000, reset_opacity=True
    )
    assert endpoint.is_noop

    reset = policy.make_plan(
        module,
        GaussianDensityStats(torch.zeros(module.count)),
        step=3_000,
        reset_opacity=True,
    )
    apply_gaussian_topology_update(module, reset)
    torch.testing.assert_close(
        module.opacity_logits.sigmoid(),
        torch.tensor([[0.2], [0.2], [0.2], [0.01]]),
    )


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
        torch.tensor([[0.1], [0.1], [0.1], [0.01]]),
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

def make_count_module(count: int) -> LearnableGaussianSet:
    return LearnableGaussianSet(
        GaussianSet(
            means=torch.arange(
                count * 3, dtype=torch.float32
            ).reshape(count, 3),
            quaternions=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]]
            ).expand(count, -1).clone(),
            scales=torch.full((count, 3), 0.25),
            opacities=torch.full((count, 1), 0.8),
            sh_coefficients=torch.zeros(count, 1, 3),
        )
    )


def test_controller_applies_group_specific_extent_policies() -> None:
    background = make_count_module(1)
    actor = make_count_module(1)
    background_policy = GaussianDensityPolicy(
        replace(thresholds(), split_scale_threshold=0.1)
    )
    actor_policy = GaussianDensityPolicy(
        replace(thresholds(), split_scale_threshold=1.0)
    )
    controller = GsplatDensityController(
        {-1: background, 7: actor},
        {-1: background_policy, 7: actor_policy},
    )
    for group_id in (-1, 7):
        accumulator = controller.accumulator(group_id)
        accumulator.gradient_sum.fill_(1.0)
        accumulator.observation_count.fill_(1.0)
    optimizer = torch.optim.Adam(
        [*background.parameters(), *actor.parameters()], lr=1.0e-3
    )

    results = controller.apply_scheduled_updates(
        step=600,
        optimizer=optimizer,
        generator=torch.Generator().manual_seed(7),
    )

    assert controller.policy is background_policy
    assert controller.policy_for(7) is actor_policy
    assert results[-1].split_parent_count == 1
    assert results[-1].duplicated_count == 0
    assert results[7].split_parent_count == 0
    assert results[7].duplicated_count == 1


def test_packed_means2d_statistics_use_clean_gsplat_normalization() -> None:
    background = make_count_module(2)
    actor = make_count_module(2)
    controller = GsplatDensityController(
        {-1: background, 7: actor},
        GaussianDensityPolicy(thresholds()),
    )
    projected = torch.zeros(5, 2, requires_grad=True)
    means2d = projected + 0.0
    metadata = {
        "means2d": means2d,
        "gaussian_ids": torch.tensor([0, 1, 1, 2, 3]),
        "radii": torch.tensor(
            [
                [10, 5],
                [20, 1],
                [30, 1],
                [40, 2],
                [50, 3],
            ]
        ),
        "width": 100,
        "height": 50,
        "n_cameras": 2,
    }
    upstream = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )
    controller.before_backward(metadata)
    (means2d * upstream).sum().backward()
    controller.after_backward(
        metadata, torch.tensor([-1, -1, 7, 7])
    )

    background_state = controller.accumulator(-1)
    expected_second = 50.0 + math.sqrt(100.0**2 + 50.0**2)
    torch.testing.assert_close(
        background_state.gradient_sum,
        torch.tensor([100.0, expected_second]),
    )
    torch.testing.assert_close(
        background_state.observation_count,
        torch.tensor([1.0, 2.0]),
    )
    torch.testing.assert_close(
        background_state.max_screen_radius,
        torch.tensor([10.0, 30.0]),
    )
    torch.testing.assert_close(
        controller.accumulator(7).gradient_sum,
        torch.tensor([200.0, 100.0]),
    )
    torch.testing.assert_close(
        controller.accumulator(7).max_screen_radius,
        torch.tensor([40.0, 50.0]),
    )
    torch.testing.assert_close(
        controller.statistics(-1).position_gradient_norm,
        torch.tensor([100.0, expected_second / 2.0]),
    )


def _run_unpacked_frame(
    controller: GsplatDensityController,
    radii: torch.Tensor,
    upstream: torch.Tensor,
    actor_id: int = 7,
) -> None:
    means2d = torch.zeros_like(upstream, requires_grad=True)
    metadata = {
        "means2d": means2d,
        "gaussian_ids": None,
        "radii": radii,
        "width": 8,
        "height": 4,
        "n_cameras": means2d.shape[0],
    }
    controller.before_backward(metadata)
    (means2d * upstream).sum().backward()
    controller.after_backward(
        metadata, torch.tensor([-1, actor_id, actor_id]), packed=False
    )


def test_unpacked_statistics_filter_visibility_and_max_across_frames() -> None:
    controller = GsplatDensityController(
        {-1: make_count_module(1), 7: make_count_module(2)},
        GaussianDensityPolicy(thresholds()),
    )
    first_radii = torch.tensor(
        [
            [[2, 1], [0, 5], [4, 2]],
            [[3, 2], [6, 1], [0, 0]],
        ]
    )
    _run_unpacked_frame(
        controller,
        first_radii,
        torch.ones(2, 3, 2),
    )
    first_norm = math.sqrt(8.0**2 + 4.0**2)
    torch.testing.assert_close(
        controller.accumulator(-1).gradient_sum,
        torch.tensor([2.0 * first_norm]),
    )
    torch.testing.assert_close(
        controller.accumulator(-1).observation_count,
        torch.tensor([2.0]),
    )
    torch.testing.assert_close(
        controller.accumulator(7).observation_count,
        torch.tensor([1.0, 1.0]),
    )

    second_radii = torch.tensor(
        [[[1, 1], [2, 1], [7, 1]]]
    )
    second_upstream = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]
    )
    _run_unpacked_frame(controller, second_radii, second_upstream)
    torch.testing.assert_close(
        controller.accumulator(-1).observation_count,
        torch.tensor([3.0]),
    )
    torch.testing.assert_close(
        controller.accumulator(7).observation_count,
        torch.tensor([2.0, 2.0]),
    )
    torch.testing.assert_close(
        controller.accumulator(-1).max_screen_radius,
        torch.tensor([3.0]),
    )
    torch.testing.assert_close(
        controller.accumulator(7).max_screen_radius,
        torch.tensor([6.0, 7.0]),
    )


def test_controller_applies_both_group_plans_and_resizes_statistics() -> None:
    background = make_module()
    actor = make_module()
    policy = GaussianDensityPolicy(thresholds())
    controller = GsplatDensityController(
        {-1: background, 7: actor}, policy
    )
    optimizer = torch.optim.Adam(
        [*background.parameters(), *actor.parameters()],
        lr=1.0e-3,
    )
    means2d = torch.zeros(1, 8, 2, requires_grad=True)
    metadata = {
        "means2d": means2d,
        "gaussian_ids": None,
        "radii": torch.ones(1, 8, 2),
        "width": 2,
        "height": 2,
        "n_cameras": 1,
    }
    per_module_gradient = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
    )
    upstream = torch.cat(
        (per_module_gradient, per_module_gradient), dim=0
    ).unsqueeze(0)
    groups = torch.tensor([-1, -1, -1, -1, 7, 7, 7, 7])
    controller.before_backward(metadata)
    (means2d * upstream).sum().backward()
    controller.after_backward(metadata, groups, packed=False)

    assert controller.apply_scheduled_updates(
        step=499, optimizer=optimizer
    ) == {}
    results = controller.apply_scheduled_updates(
        step=600,
        optimizer=optimizer,
        generator=torch.Generator().manual_seed(9),
    )
    assert set(results) == {-1, 7}
    assert all(result.new_count == 5 for result in results.values())
    assert background.count == actor.count == 5
    for group_id in (-1, 7):
        accumulator = controller.accumulator(group_id)
        assert accumulator.count == 5
        assert torch.count_nonzero(accumulator.gradient_sum) == 0
        assert torch.count_nonzero(accumulator.observation_count) == 0
        assert torch.count_nonzero(accumulator.max_screen_radius) == 0

    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert {
        id(parameter)
        for parameter in [*background.parameters(), *actor.parameters()]
    }.issubset(optimized_ids)


def test_controller_state_dict_restores_statistics_exactly() -> None:
    modules = {-1: make_count_module(1), 3: make_count_module(2)}
    policy = GaussianDensityPolicy(thresholds())
    controller = GsplatDensityController(modules, policy)
    radii = torch.tensor(
        [[[2, 1], [3, 1], [4, 1]]]
    )
    upstream = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]
    )
    _run_unpacked_frame(controller, radii, upstream, actor_id=3)
    state = controller.state_dict()

    restored = GsplatDensityController(modules, policy)
    restored.load_state_dict(state)
    for group_id in (-1, 3):
        expected = controller.accumulator(group_id)
        actual = restored.accumulator(group_id)
        torch.testing.assert_close(
            actual.gradient_sum, expected.gradient_sum
        )
        torch.testing.assert_close(
            actual.observation_count, expected.observation_count
        )
        torch.testing.assert_close(
            actual.max_screen_radius,
            expected.max_screen_radius,
        )

    state["accumulators"][-1]["gradient_sum"].add_(100)
    assert not torch.equal(
        state["accumulators"][-1]["gradient_sum"],
        controller.accumulator(-1).gradient_sum,
    )


def test_controller_rejects_checkpoint_with_pending_backward() -> None:
    controller = GsplatDensityController(
        {-1: make_count_module(1)},
        GaussianDensityPolicy(thresholds()),
    )
    means2d = torch.zeros(1, 1, 2, requires_grad=True)
    metadata = {
        "means2d": means2d,
        "gaussian_ids": None,
        "radii": torch.ones(1, 1, 2),
        "width": 1,
        "height": 1,
        "n_cameras": 1,
    }
    controller.before_backward(metadata)
    with pytest.raises(RuntimeError, match="cannot checkpoint"):
        controller.state_dict()
