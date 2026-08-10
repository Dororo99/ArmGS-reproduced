"""Density-control policy and topology-safe optimizer migration for ArmGS.

ArmGS reports *when* densification runs, but not the thresholds that decide
which Gaussian is duplicated, split, or pruned. The published schedule and the
experiment-owned thresholds therefore have separate types in this module.

Topology changes replace nn.Parameter objects. Call
apply_gaussian_topology_update with its optimizer argument, or pass its result
to migrate_adam_optimizer so Adam never keeps state keyed by stale parameters.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .geometry import normalize_quaternion, rotate_points
from .scene import LearnableGaussianSet


_RAW_PARAMETER_NAMES = (
    "means",
    "quaternions",
    "log_scales",
    "opacity_logits",
    "sh_coefficients",
)


@dataclass(frozen=True)
class DensificationSchedule:
    """ArmGS/3DGS densification window with exclusive boundaries.

    ArmGS reports densification every 100 steps from step 500 to 15,000. The
    inherited 3DGS training loop implements those values as strict bounds and
    tests the global iteration modulo the interval. Consequently, the actual
    default topology-update events are 600, 700, ..., 14,900.
    """

    start_step: int = 500
    end_step: int = 15_000
    interval: int = 100

    def __post_init__(self) -> None:
        values = (self.start_step, self.end_step, self.interval)
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values
        ):
            raise TypeError("densification schedule values must be integers")
        if self.start_step < 0:
            raise ValueError("densification start_step cannot be negative")
        if self.end_step <= self.start_step:
            raise ValueError(
                "densification end_step must follow start_step"
            )
        if self.interval <= 0:
            raise ValueError("densification interval must be positive")

    def is_due(self, step: int) -> bool:
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step < 0:
            raise ValueError("step cannot be negative")
        return (
            self.start_step < step < self.end_step
            and step % self.interval == 0
        )


@dataclass(frozen=True)
class DensityControlThresholds:
    """Experiment settings not specified by the ArmGS paper.

    Required fields force callers to record every unpublished choice.
    split_scale_threshold is an absolute world-space maximum scale.
    """

    position_gradient_threshold: float
    split_scale_threshold: float
    prune_opacity_threshold: float
    split_children: int
    split_scale_reduction: float
    opacity_reset_value: float | None
    max_screen_radius: float | None = None
    max_world_scale: float | None = None
    prune_large_after_step: int = 3_000
    minimum_gaussians: int = 0

    def __post_init__(self) -> None:
        finite_nonnegative = {
            "position_gradient_threshold": self.position_gradient_threshold,
            "split_scale_threshold": self.split_scale_threshold,
            "prune_opacity_threshold": self.prune_opacity_threshold,
        }
        for name, value in finite_nonnegative.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.prune_opacity_threshold > 1.0:
            raise ValueError("prune_opacity_threshold must not exceed one")
        if isinstance(self.split_children, bool) or self.split_children < 2:
            raise ValueError("split_children must be at least two")
        if (
            not math.isfinite(self.split_scale_reduction)
            or self.split_scale_reduction <= 0.0
        ):
            raise ValueError(
                "split_scale_reduction must be finite and positive"
            )
        if self.opacity_reset_value is not None and (
            not math.isfinite(self.opacity_reset_value)
            or not 0.0 < self.opacity_reset_value < 1.0
        ):
            raise ValueError(
                "opacity_reset_value must lie strictly between zero and one"
            )
        if self.max_screen_radius is not None and (
            not math.isfinite(self.max_screen_radius)
            or self.max_screen_radius <= 0.0
        ):
            raise ValueError("max_screen_radius must be finite and positive")
        if self.max_world_scale is not None and (
            not math.isfinite(self.max_world_scale)
            or self.max_world_scale <= 0.0
        ):
            raise ValueError("max_world_scale must be finite and positive")
        if (
            isinstance(self.prune_large_after_step, bool)
            or not isinstance(self.prune_large_after_step, int)
            or self.prune_large_after_step < 0
        ):
            raise ValueError(
                "prune_large_after_step must be a non-negative integer"
            )
        if (
            isinstance(self.minimum_gaussians, bool)
            or self.minimum_gaussians < 0
        ):
            raise ValueError("minimum_gaussians must be non-negative")


@dataclass(frozen=True)
class GaussianDensityStats:
    """Per-Gaussian statistics accumulated by the integration training loop."""

    position_gradient_norm: Tensor
    max_screen_radius: Tensor | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("position_gradient_norm", self.position_gradient_norm),
            ("max_screen_radius", self.max_screen_radius),
        ):
            if value is None:
                continue
            if value.ndim != 1:
                raise ValueError(f"{name} must have shape [N]")
            if not value.is_floating_point():
                raise ValueError(f"{name} must be floating point")
            if (
                not torch.isfinite(value.detach()).all()
                or torch.any(value.detach() < 0)
            ):
                raise ValueError(
                    f"{name} must contain finite non-negative values"
                )


def _validate_indices(name: str, indices: Tensor, source_count: int) -> None:
    if indices.ndim != 1 or indices.dtype != torch.long:
        raise ValueError(
            f"{name} must be a one-dimensional torch.long tensor"
        )
    if indices.numel() and (
        int(indices.detach().min()) < 0
        or int(indices.detach().max()) >= source_count
    ):
        raise ValueError(f"{name} contains an out-of-range source index")
    if indices.unique().numel() != indices.numel():
        raise ValueError(f"{name} cannot contain duplicate indices")


@dataclass(frozen=True)
class GaussianTopologyUpdatePlan:
    """A fully materialized Gaussian row-topology update.

    Output rows are ordered as retained originals, duplicates, then split
    children. A duplicate can replace a screen-space-oversized parent because
    newly cloned rows have zero accumulated screen radius in official 3DGS.
    Split parents are replaced by their children. Unrepresented source rows
    are pruned.
    """

    source_count: int
    retain_indices: Tensor
    duplicate_indices: Tensor
    split_indices: Tensor
    split_offsets: Tensor
    split_scale_reduction: float = 1.0
    opacity_reset_value: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_count, bool)
            or not isinstance(self.source_count, int)
            or self.source_count < 0
        ):
            raise ValueError("source_count must be a non-negative integer")
        for name, indices in (
            ("retain_indices", self.retain_indices),
            ("duplicate_indices", self.duplicate_indices),
            ("split_indices", self.split_indices),
        ):
            _validate_indices(name, indices, self.source_count)
        if (
            self.split_offsets.ndim != 3
            or self.split_offsets.shape[2] != 3
        ):
            raise ValueError("split_offsets must have shape [S, children, 3]")
        if self.split_offsets.shape[0] != self.split_indices.numel():
            raise ValueError(
                "split_offsets first dimension must match split_indices"
            )
        if self.split_offsets.shape[1] < 2:
            raise ValueError(
                "each split parent must produce at least two children"
            )
        if not self.split_offsets.is_floating_point():
            raise ValueError("split_offsets must be floating point")
        if not torch.isfinite(self.split_offsets.detach()).all():
            raise ValueError("split_offsets must be finite")
        if (
            not math.isfinite(self.split_scale_reduction)
            or self.split_scale_reduction <= 0.0
        ):
            raise ValueError(
                "split_scale_reduction must be finite and positive"
            )
        if self.opacity_reset_value is not None and (
            not math.isfinite(self.opacity_reset_value)
            or not 0.0 < self.opacity_reset_value < 1.0
        ):
            raise ValueError(
                "opacity_reset_value must lie strictly between zero and one"
            )

        duplicated = set(self.duplicate_indices.detach().cpu().tolist())
        split = set(self.split_indices.detach().cpu().tolist())
        retained = set(self.retain_indices.detach().cpu().tolist())
        if retained.intersection(split):
            raise ValueError("split parents cannot also be retained")
        if duplicated.intersection(split):
            raise ValueError("split parents cannot also be duplicated")

    @property
    def split_children(self) -> int:
        return self.split_offsets.shape[1]

    @property
    def output_count(self) -> int:
        return (
            self.retain_indices.numel()
            + self.duplicate_indices.numel()
            + self.split_indices.numel() * self.split_children
        )

    @property
    def pruned_count(self) -> int:
        represented = (
            self.retain_indices.numel() + self.split_indices.numel()
        )
        return self.source_count - represented

    @property
    def is_noop(self) -> bool:
        expected = torch.arange(
            self.source_count,
            dtype=torch.long,
            device=self.retain_indices.device,
        )
        return (
            self.opacity_reset_value is None
            and self.duplicate_indices.numel() == 0
            and self.split_indices.numel() == 0
            and torch.equal(self.retain_indices, expected)
        )

    @classmethod
    def no_op(
        cls,
        source_count: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "GaussianTopologyUpdatePlan":
        return cls(
            source_count=source_count,
            retain_indices=torch.arange(
                source_count, dtype=torch.long, device=device
            ),
            duplicate_indices=torch.empty(
                0, dtype=torch.long, device=device
            ),
            split_indices=torch.empty(
                0, dtype=torch.long, device=device
            ),
            split_offsets=torch.empty(
                0, 2, 3, dtype=dtype, device=device
            ),
        )


@dataclass(frozen=True)
class GaussianTopologyUpdateResult:
    """Materialized topology mapping and stale-to-live parameter references."""

    old_count: int
    new_count: int
    retained_count: int
    duplicated_count: int
    split_parent_count: int
    split_child_count: int
    pruned_count: int
    retained_source_indices: Tensor
    source_indices: Tensor
    new_row_mask: Tensor
    opacity_was_reset: bool
    parameter_replacements: Mapping[
        str, tuple[nn.Parameter, nn.Parameter]
    ]
    optimizer_migrated: bool = False

    @property
    def topology_changed(self) -> bool:
        return bool(
            self.duplicated_count
            or self.split_parent_count
            or self.pruned_count
        )


class GaussianDensityPolicy:
    """Convert accumulated statistics into an explicit topology plan."""

    def __init__(
        self,
        thresholds: DensityControlThresholds,
        *,
        schedule: DensificationSchedule | None = None,
        actor_box_half_extents: tuple[float, float, float] | None = None,
    ) -> None:
        self.thresholds = thresholds
        self.schedule = schedule or DensificationSchedule()
        if actor_box_half_extents is not None:
            if len(actor_box_half_extents) != 3:
                raise ValueError(
                    "actor_box_half_extents must contain three values"
                )
            resolved_extents = tuple(
                float(value) for value in actor_box_half_extents
            )
            if any(
                not math.isfinite(value) or value <= 0.0
                for value in resolved_extents
            ):
                raise ValueError(
                    "actor_box_half_extents must be finite and positive"
                )
            actor_box_half_extents = resolved_extents
        self.actor_box_half_extents = actor_box_half_extents

    def _outside_actor_box_mask(
        self,
        module: LearnableGaussianSet,
        *,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Sample StreetGS actor support and flag rows outside its local box."""

        if self.actor_box_half_extents is None:
            return torch.zeros(
                module.count,
                dtype=torch.bool,
                device=module.means.device,
            )
        repeat_num = 2
        scales = module.log_scales.detach().exp()
        standard_deviations = scales[:, None, :].expand(
            -1, repeat_num, -1
        )
        zero_means = torch.zeros_like(standard_deviations)
        samples = torch.normal(
            mean=zero_means,
            std=standard_deviations,
            generator=generator,
        )
        quaternions = normalize_quaternion(module.quaternions.detach())
        expanded_quaternions = quaternions[:, None, :].expand(
            -1, repeat_num, -1
        )
        rotated_samples = rotate_points(
            expanded_quaternions.reshape(-1, 4),
            samples.reshape(-1, 3),
        ).reshape(module.count, repeat_num, 3)
        sample_positions = rotated_samples + module.means.detach()[:, None, :]
        half_extents = sample_positions.new_tensor(
            self.actor_box_half_extents
        )
        inside = (
            (sample_positions >= -half_extents)
            & (sample_positions <= half_extents)
        ).all(dim=-1).all(dim=-1)
        return ~inside

    def make_plan(
        self,
        module: LearnableGaussianSet,
        stats: GaussianDensityStats | None,
        *,
        step: int,
        reset_opacity: bool = False,
        generator: torch.Generator | None = None,
    ) -> GaussianTopologyUpdatePlan:
        density_due = self.schedule.is_due(step)
        # The reference loop guards both reset and densification with
        # iteration < densify_until_iter. In particular, iteration 15,000 is
        # neither a density event nor an opacity reset.
        reset_due = reset_opacity and step < self.schedule.end_step
        reset_value: float | None = None
        if reset_due:
            reset_value = self.thresholds.opacity_reset_value
            if reset_value is None:
                raise ValueError(
                    "reset_opacity was requested but "
                    "opacity_reset_value is disabled"
                )
        if not density_due and reset_value is None:
            return GaussianTopologyUpdatePlan.no_op(
                module.count,
                device=module.means.device,
                dtype=module.means.dtype,
            )
        if not density_due:
            base = GaussianTopologyUpdatePlan.no_op(
                module.count,
                device=module.means.device,
                dtype=module.means.dtype,
            )
            return replace(base, opacity_reset_value=reset_value)

        if stats is None:
            raise ValueError(
                "density statistics are required on a scheduled step"
            )
        if stats.position_gradient_norm.shape != (module.count,):
            raise ValueError(
                "position_gradient_norm must have one value per Gaussian"
            )
        if stats.position_gradient_norm.device != module.means.device:
            raise ValueError(
                "density statistics must share the Gaussian device"
            )
        prune_large = step > self.thresholds.prune_large_after_step
        if prune_large and self.thresholds.max_screen_radius is not None:
            if stats.max_screen_radius is None:
                raise ValueError(
                    "max_screen_radius statistics are required "
                    "by the thresholds"
                )
            if stats.max_screen_radius.shape != (module.count,):
                raise ValueError(
                    "max_screen_radius must have one value per Gaussian"
                )
            if stats.max_screen_radius.device != module.means.device:
                raise ValueError(
                    "density statistics must share the Gaussian device"
                )

        with torch.no_grad():
            opacity = (
                module.opacity_logits.detach().sigmoid().squeeze(-1)
            )
            maximum_scale = (
                module.log_scales.detach().exp().amax(dim=-1)
            )
            low_opacity = (
                opacity < self.thresholds.prune_opacity_threshold
            )
            screen_too_large = torch.zeros_like(
                low_opacity, dtype=torch.bool
            )
            world_too_large = torch.zeros_like(
                low_opacity, dtype=torch.bool
            )
            outside_actor_box = torch.zeros_like(
                low_opacity, dtype=torch.bool
            )
            if prune_large and self.thresholds.max_screen_radius is not None:
                assert stats.max_screen_radius is not None
                screen_too_large = (
                    stats.max_screen_radius.detach()
                    > self.thresholds.max_screen_radius
                )
            if prune_large and self.thresholds.max_world_scale is not None:
                world_too_large = (
                    maximum_scale > self.thresholds.max_world_scale
                )
            if prune_large and self.actor_box_half_extents is not None:
                outside_actor_box = self._outside_actor_box_mask(
                    module, generator=generator
                )

            high_gradient = (
                stats.position_gradient_norm.detach()
                >= self.thresholds.position_gradient_threshold
            )
            selected_split = high_gradient & (
                maximum_scale
                > self.thresholds.split_scale_threshold
            )
            selected_clone = high_gradient & ~selected_split

            # 3DGS densifies first, then prunes. Newly cloned/split rows have
            # zero screen-radius history. World-space pruning still applies to
            # their (possibly reduced) scales.
            duplicate_mask = selected_clone & ~low_opacity
            if prune_large and self.thresholds.max_world_scale is not None:
                duplicate_mask &= ~world_too_large
            duplicate_mask &= ~outside_actor_box

            split_children_too_large = torch.zeros_like(
                selected_split, dtype=torch.bool
            )
            if prune_large and self.thresholds.max_world_scale is not None:
                split_children_too_large = (
                    maximum_scale / self.thresholds.split_scale_reduction
                    > self.thresholds.max_world_scale
                )
            split_mask = (
                selected_split
                & ~low_opacity
                & ~split_children_too_large
                & ~outside_actor_box
            )

            prune_original = (
                low_opacity
                | screen_too_large
                | world_too_large
                | outside_actor_box
            )
            retain_mask = ~selected_split & ~prune_original

            output_count = int(retain_mask.sum()) + int(
                duplicate_mask.sum()
            ) + int(split_mask.sum()) * self.thresholds.split_children
            minimum = min(self.thresholds.minimum_gaussians, module.count)
            if output_count < minimum:
                rescue_count = minimum - output_count
                # Actor box containment is a hard geometric invariant.  A
                # non-reference minimum-count safeguard may rescue opacity or
                # size-pruned rows, but never a row outside the tracking box.
                rescue_candidates = (
                    ~retain_mask & ~outside_actor_box
                ).nonzero(
                    as_tuple=False
                ).squeeze(-1)
                rescue_order = torch.argsort(
                    opacity.index_select(0, rescue_candidates),
                    descending=True,
                    stable=True,
                )
                rescued = rescue_candidates.index_select(
                    0,
                    rescue_order[
                        : min(rescue_count, rescue_candidates.numel())
                    ],
                )
                retain_mask[rescued] = True
                duplicate_mask[rescued] = False
                split_mask[rescued] = False

            all_split_indices = selected_split.nonzero(
                as_tuple=False
            ).squeeze(-1)
            surviving_split = split_mask.index_select(
                0, all_split_indices
            )
            split_indices = all_split_indices[surviving_split]
            duplicate_indices = duplicate_mask.nonzero(
                as_tuple=False
            ).squeeze(-1)
            retain_indices = retain_mask.nonzero(
                as_tuple=False
            ).squeeze(-1)

            # Draw for all selected parents before applying post-densification
            # pruning, matching the reference RNG consumption.
            all_standard_offsets = torch.randn(
                all_split_indices.numel(),
                self.thresholds.split_children,
                3,
                device=module.means.device,
                dtype=module.means.dtype,
                generator=generator,
            )
            standard_offsets = all_standard_offsets[surviving_split]
            split_count = split_indices.numel()
            parent_scales = (
                module.log_scales.detach()
                .exp()
                .index_select(0, split_indices)
            )
            local_offsets = (
                standard_offsets * parent_scales[:, None, :]
            )
            parent_quaternions = normalize_quaternion(
                module.quaternions.detach().index_select(
                    0, split_indices
                )
            )
            expanded_quaternions = (
                parent_quaternions[:, None, :].expand(
                    -1, self.thresholds.split_children, -1
                )
            )
            split_offsets = rotate_points(
                expanded_quaternions.reshape(-1, 4),
                local_offsets.reshape(-1, 3),
            ).reshape(
                split_count, self.thresholds.split_children, 3
            )

        return GaussianTopologyUpdatePlan(
            source_count=module.count,
            retain_indices=retain_indices,
            duplicate_indices=duplicate_indices,
            split_indices=split_indices,
            split_offsets=split_offsets,
            split_scale_reduction=(
                self.thresholds.split_scale_reduction
            ),
            opacity_reset_value=reset_value,
        )


def _validate_plan_for_module(
    module: LearnableGaussianSet,
    plan: GaussianTopologyUpdatePlan,
) -> None:
    if plan.source_count != module.count:
        raise ValueError(
            f"topology plan expected {plan.source_count} "
            f"Gaussians, got {module.count}"
        )
    for name in _RAW_PARAMETER_NAMES:
        parameter = getattr(module, name)
        if parameter.shape[0] != module.count:
            raise RuntimeError(
                f"raw Gaussian parameter {name!r} has inconsistent rows"
            )


def _empty_result(
    module: LearnableGaussianSet,
) -> GaussianTopologyUpdateResult:
    source = torch.arange(
        module.count, dtype=torch.long, device=module.means.device
    )
    return GaussianTopologyUpdateResult(
        old_count=module.count,
        new_count=module.count,
        retained_count=module.count,
        duplicated_count=0,
        split_parent_count=0,
        split_child_count=0,
        pruned_count=0,
        retained_source_indices=source,
        source_indices=source,
        new_row_mask=torch.zeros(
            module.count, dtype=torch.bool, device=source.device
        ),
        opacity_was_reset=False,
        parameter_replacements={},
    )


def _build_result(
    plan: GaussianTopologyUpdatePlan,
    retain: Tensor,
    duplicate: Tensor,
    split: Tensor,
    source_indices: Tensor,
    replacements: Mapping[
        str, tuple[nn.Parameter, nn.Parameter]
    ],
) -> GaussianTopologyUpdateResult:
    retained_count = retain.numel()
    new_row_mask = torch.zeros(
        source_indices.numel(),
        dtype=torch.bool,
        device=source_indices.device,
    )
    new_row_mask[retained_count:] = True
    return GaussianTopologyUpdateResult(
        old_count=plan.source_count,
        new_count=source_indices.numel(),
        retained_count=retained_count,
        duplicated_count=duplicate.numel(),
        split_parent_count=split.numel(),
        split_child_count=(
            split.numel() * plan.split_children
        ),
        pruned_count=plan.pruned_count,
        retained_source_indices=retain,
        source_indices=source_indices,
        new_row_mask=new_row_mask,
        opacity_was_reset=(
            plan.opacity_reset_value is not None
        ),
        parameter_replacements=replacements,
    )


def apply_gaussian_topology_update(
    module: LearnableGaussianSet,
    plan: GaussianTopologyUpdatePlan,
    *,
    optimizer: torch.optim.Adam | None = None,
) -> GaussianTopologyUpdateResult:
    """Apply a plan to raw 3DGS parameters and optionally migrate Adam.

    Kept rows retain their Adam moments. Duplicate and split-child rows start
    with zero moments. An opacity reset clears all opacity moments. Call this
    between optimization steps, after gradients have been consumed.
    """

    _validate_plan_for_module(module, plan)
    if plan.is_noop:
        result = _empty_result(module)
        return replace(
            result, optimizer_migrated=optimizer is not None
        )

    device = module.means.device
    retain = plan.retain_indices.to(device=device)
    duplicate = plan.duplicate_indices.to(device=device)
    split = plan.split_indices.to(device=device)
    split_sources = split.repeat_interleave(plan.split_children)
    source_indices = torch.cat(
        (retain, duplicate, split_sources), dim=0
    )
    topology_changed = bool(
        duplicate.numel() or split.numel() or plan.pruned_count
    )
    replacement_names = (
        _RAW_PARAMETER_NAMES
        if topology_changed
        else ("opacity_logits",)
    )
    split_output_start = retain.numel() + duplicate.numel()
    split_offsets = plan.split_offsets.to(
        device=device, dtype=module.means.dtype
    ).reshape(-1, 3)

    replacements: dict[
        str, tuple[nn.Parameter, nn.Parameter]
    ] = {}
    for name in replacement_names:
        old = getattr(module, name)
        value = (
            old.detach().index_select(0, source_indices).clone()
            if topology_changed
            else old.detach().clone()
        )
        if name == "means" and split_sources.numel():
            value[split_output_start:] += split_offsets
        elif name == "log_scales" and split_sources.numel():
            value[split_output_start:] -= math.log(
                plan.split_scale_reduction
            )
        elif (
            name == "opacity_logits"
            and plan.opacity_reset_value is not None
        ):
            reset_cap = value.new_tensor(plan.opacity_reset_value)
            capped_opacity = torch.minimum(
                value.sigmoid(), reset_cap
            )
            value.copy_(torch.logit(capped_opacity))
        replacements[name] = (
            old,
            nn.Parameter(value, requires_grad=old.requires_grad),
        )

    result = _build_result(
        plan,
        retain,
        duplicate,
        split,
        source_indices,
        replacements,
    )
    if optimizer is not None:
        result = migrate_adam_optimizer(optimizer, result)

    for name, (_, new) in replacements.items():
        setattr(module, name, new)
    if topology_changed and module._group_ids.numel():
        module._buffers["_group_ids"] = (
            module._group_ids.index_select(
                0, source_indices
            ).clone()
        )
    return result


def _migrated_adam_state(
    state: dict[object, object],
    *,
    name: str,
    old: nn.Parameter,
    new: nn.Parameter,
    result: GaussianTopologyUpdateResult,
) -> dict[object, object]:
    migrated: dict[object, object] = {}
    for key, value in state.items():
        if not isinstance(value, Tensor):
            migrated[key] = deepcopy(value)
            continue
        if value.ndim == 0:
            migrated[key] = value.detach().clone()
            continue
        if value.shape != old.shape:
            raise RuntimeError(
                f"unsupported Adam state shape {tuple(value.shape)} "
                f"for {name} parameter shape {tuple(old.shape)}"
            )
        target = value.new_zeros(new.shape)
        if not (
            name == "opacity_logits" and result.opacity_was_reset
        ):
            retained = result.retained_source_indices.to(
                device=value.device
            )
            target[: result.retained_count] = value.index_select(
                0, retained
            )
        migrated[key] = target
    return migrated


def migrate_adam_optimizer(
    optimizer: torch.optim.Adam,
    result: GaussianTopologyUpdateResult,
) -> GaussianTopologyUpdateResult:
    """Replace stale parameters and row-migrate Adam state transactionally."""

    if not isinstance(optimizer, torch.optim.Adam):
        raise TypeError(
            "density state migration currently supports torch.optim.Adam"
        )
    if result.optimizer_migrated:
        raise ValueError(
            "this topology result has already migrated an optimizer"
        )
    if not result.parameter_replacements:
        return replace(result, optimizer_migrated=True)

    locations: dict[
        str, tuple[dict[str, object], int]
    ] = {}
    migrated_states: dict[
        str, dict[object, object]
    ] = {}
    for name, (old, new) in result.parameter_replacements.items():
        matches: list[tuple[dict[str, object], int]] = []
        for group in optimizer.param_groups:
            parameters = group["params"]
            assert isinstance(parameters, list)
            for index, parameter in enumerate(parameters):
                if parameter is old:
                    matches.append((group, index))
        if len(matches) != 1:
            raise ValueError(
                f"old {name} parameter must occur exactly once "
                "in the Adam optimizer"
            )
        locations[name] = matches[0]
        migrated_states[name] = _migrated_adam_state(
            optimizer.state.get(old, {}),
            name=name,
            old=old,
            new=new,
            result=result,
        )

    for name, (old, new) in result.parameter_replacements.items():
        group, index = locations[name]
        parameters = group["params"]
        assert isinstance(parameters, list)
        parameters[index] = new
        optimizer.state.pop(old, None)
        if migrated_states[name]:
            optimizer.state[new] = migrated_states[name]
    return replace(result, optimizer_migrated=True)


__all__ = [
    "DensificationSchedule",
    "DensityControlThresholds",
    "GaussianDensityPolicy",
    "GaussianDensityStats",
    "GaussianTopologyUpdatePlan",
    "GaussianTopologyUpdateResult",
    "apply_gaussian_topology_update",
    "migrate_adam_optimizer",
]



@dataclass
class GaussianDensityAccumulator:
    """Running 3DGS screen-space statistics for one Gaussian module."""

    gradient_sum: Tensor
    observation_count: Tensor
    max_screen_radius: Tensor

    def __post_init__(self) -> None:
        count = self.gradient_sum.numel()
        expected = (count,)
        for name, value in (
            ("gradient_sum", self.gradient_sum),
            ("observation_count", self.observation_count),
            ("max_screen_radius", self.max_screen_radius),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
            if not value.is_floating_point():
                raise ValueError(f"{name} must be floating point")
            if not torch.isfinite(value.detach()).all():
                raise ValueError(f"{name} must be finite")
            if torch.any(value.detach() < 0):
                raise ValueError(f"{name} must be non-negative")
        devices = {
            self.gradient_sum.device,
            self.observation_count.device,
            self.max_screen_radius.device,
        }
        if len(devices) != 1:
            raise ValueError("density accumulator tensors must share a device")

    @classmethod
    def zeros(
        cls,
        count: int,
        *,
        device: torch.device | str,
    ) -> "GaussianDensityAccumulator":
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("accumulator count must be a non-negative integer")
        return cls(
            gradient_sum=torch.zeros(count, dtype=torch.float32, device=device),
            observation_count=torch.zeros(
                count, dtype=torch.float32, device=device
            ),
            max_screen_radius=torch.zeros(
                count, dtype=torch.float32, device=device
            ),
        )

    @property
    def max_normalized_radius(self) -> Tensor:
        """Deprecated compatibility alias; values are now raw pixels."""

        return self.max_screen_radius

    @property
    def count(self) -> int:
        return self.gradient_sum.numel()

    def ensure_layout(self, module: LearnableGaussianSet) -> None:
        if self.count != module.count:
            raise RuntimeError(
                "density statistics no longer match Gaussian topology; "
                "apply topology changes through GsplatDensityController"
            )
        device = module.means.device
        if self.gradient_sum.device != device:
            self.gradient_sum = self.gradient_sum.to(device=device)
            self.observation_count = self.observation_count.to(device=device)
            self.max_screen_radius = self.max_screen_radius.to(
                device=device
            )

    def add_(
        self,
        gradient_sum: Tensor,
        observation_count: Tensor,
        max_screen_radius: Tensor,
    ) -> None:
        expected = (self.count,)
        values = (
            gradient_sum,
            observation_count,
            max_screen_radius,
        )
        if any(value.shape != expected for value in values):
            raise ValueError("frame statistics must match accumulator shape")
        gradient_sum = gradient_sum.to(self.gradient_sum)
        observation_count = observation_count.to(self.observation_count)
        max_screen_radius = max_screen_radius.to(
            self.max_screen_radius
        )
        if any(
            not torch.isfinite(value.detach()).all()
            or torch.any(value.detach() < 0)
            for value in (
                gradient_sum,
                observation_count,
                max_screen_radius,
            )
        ):
            raise ValueError("frame statistics must be finite and non-negative")
        self.gradient_sum.add_(gradient_sum)
        self.observation_count.add_(observation_count)
        self.max_screen_radius.copy_(
            torch.maximum(self.max_screen_radius, max_screen_radius)
        )

    def reset_(
        self,
        count: int | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        target_count = self.count if count is None else count
        target_device = (
            self.gradient_sum.device if device is None else torch.device(device)
        )
        replacement = self.zeros(target_count, device=target_device)
        self.gradient_sum = replacement.gradient_sum
        self.observation_count = replacement.observation_count
        self.max_screen_radius = replacement.max_screen_radius

    def statistics(self) -> GaussianDensityStats:
        average = self.gradient_sum / self.observation_count.clamp_min(1.0)
        return GaussianDensityStats(
            position_gradient_norm=average,
            max_screen_radius=self.max_screen_radius,
        )

    def state_dict(self) -> dict[str, Tensor]:
        return {
            "gradient_sum": self.gradient_sum.detach().clone(),
            "observation_count": self.observation_count.detach().clone(),
            "max_screen_radius": (
                self.max_screen_radius.detach().clone()
            ),
        }

    def load_state_dict(
        self,
        state: Mapping[str, object],
        *,
        count: int,
        device: torch.device | str,
    ) -> None:
        expected_keys = {
            "gradient_sum",
            "observation_count",
            "max_screen_radius",
        }
        if set(state) != expected_keys:
            raise ValueError("invalid density accumulator state keys")
        tensors: dict[str, Tensor] = {}
        for name in expected_keys:
            value = state[name]
            if not isinstance(value, Tensor):
                raise ValueError(f"{name} state must be a tensor")
            value = value.detach().to(device=device, dtype=torch.float32)
            if value.shape != (count,):
                raise ValueError(
                    f"{name} checkpoint shape must be {(count,)}"
                )
            tensors[name] = value.clone()
        loaded = GaussianDensityAccumulator(**tensors)
        self.gradient_sum = loaded.gradient_sum
        self.observation_count = loaded.observation_count
        self.max_screen_radius = loaded.max_screen_radius


def _metadata_positive_integer(metadata: Mapping[str, Any], key: str) -> int:
    if key not in metadata:
        raise KeyError(f"gsplat metadata is missing {key!r}")
    value = metadata[key]
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"metadata {key!r} must be scalar")
        value = value.detach().item()
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"metadata {key!r} must be a positive integer")
    return value


def _radius_values(radii: Tensor, base_shape: tuple[int, ...]) -> Tensor:
    if radii.shape == base_shape:
        return radii
    if radii.ndim == len(base_shape) + 1 and radii.shape[:-1] == base_shape:
        if radii.shape[-1] < 1:
            raise ValueError("radii final dimension cannot be empty")
        return radii.amax(dim=-1)
    raise ValueError(
        "radii shape must match means2d rows, with an optional axis dimension"
    )


def _frame_screen_statistics(
    metadata: Mapping[str, Any],
    means2d_gradient: Tensor,
    *,
    gaussian_count: int,
    packed: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    width = _metadata_positive_integer(metadata, "width")
    height = _metadata_positive_integer(metadata, "height")
    camera_count = _metadata_positive_integer(metadata, "n_cameras")
    radii = metadata.get("radii")
    if not isinstance(radii, Tensor):
        raise ValueError("gsplat metadata 'radii' must be a tensor")
    radii = radii.detach().to(device=means2d_gradient.device)
    if radii.is_floating_point() and not torch.isfinite(radii).all():
        raise ValueError("radii must be finite")
    if torch.any(radii < 0):
        raise ValueError("radii must be non-negative")

    gradients = means2d_gradient.detach().clone()
    if not gradients.is_floating_point() or not torch.isfinite(gradients).all():
        raise ValueError("means2d gradients must be finite floating-point values")
    gradients[..., 0] *= width / 2.0 * camera_count
    gradients[..., 1] *= height / 2.0 * camera_count

    if packed:
        gaussian_ids = metadata.get("gaussian_ids")
        if not isinstance(gaussian_ids, Tensor):
            raise ValueError(
                "packed gsplat metadata requires tensor gaussian_ids"
            )
        if gradients.ndim != 2 or gradients.shape[-1] != 2:
            raise ValueError("packed means2d gradients must have shape [nnz,2]")
        if gaussian_ids.shape != (gradients.shape[0],):
            raise ValueError("gaussian_ids must have shape [nnz]")
        ids = gaussian_ids.detach().to(
            device=gradients.device, dtype=torch.long
        )
        radius_values = _radius_values(
            radii, (gradients.shape[0],)
        ).to(dtype=torch.float32)
    else:
        if (
            gradients.ndim != 3
            or gradients.shape
            != (camera_count, gaussian_count, 2)
        ):
            raise ValueError(
                "unpacked means2d gradients must have shape [C,N,2]"
            )
        radius_grid = _radius_values(
            radii, (camera_count, gaussian_count)
        )
        visible_radii = (
            radii
            if radii.shape == (camera_count, gaussian_count)
            else radii[..., 0]
        )
        visible = visible_radii > 0
        ids = torch.where(visible)[1]
        gradients = gradients[visible]
        radius_values = radius_grid[visible].to(dtype=torch.float32)

    if ids.numel() and (
        int(ids.min()) < 0 or int(ids.max()) >= gaussian_count
    ):
        raise ValueError("gaussian_ids contain an out-of-range index")
    gradient_sum = torch.zeros(
        gaussian_count,
        dtype=torch.float32,
        device=gradients.device,
    )
    observation_count = torch.zeros_like(gradient_sum)
    max_radius = torch.zeros_like(gradient_sum)
    if ids.numel():
        gradient_sum.index_add_(
            0, ids, gradients.norm(dim=-1).to(dtype=torch.float32)
        )
        observation_count.index_add_(
            0, ids, torch.ones_like(ids, dtype=torch.float32)
        )
        max_radius.scatter_reduce_(
            0,
            ids,
            radius_values,
            reduce="amax",
            include_self=True,
        )
    return gradient_sum, observation_count, max_radius


def _serialized_density_policy(
    policy: GaussianDensityPolicy,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "schedule": asdict(policy.schedule),
        "thresholds": asdict(policy.thresholds),
    }
    # Keep generic checkpoint payloads byte-for-byte compatible with the
    # pre-bounds controller while making enabled actor geometry strict on
    # resume.
    if policy.actor_box_half_extents is not None:
        state["actor_box_half_extents"] = policy.actor_box_half_extents
    return state


class GsplatDensityController:
    """Trainer-facing screen-statistics and scheduled topology lifecycle.

    Group -1 is the static background and non-negative groups are actor IDs.
    The controller consumes gradients of gsplat metadata['means2d']; raw
    world-space mean gradients are never used as the densification proxy.
    """

    _STATE_VERSION = 2

    def __init__(
        self,
        modules: Mapping[int, LearnableGaussianSet],
        policy: (
            GaussianDensityPolicy
            | Mapping[int, GaussianDensityPolicy]
        ),
    ) -> None:
        if -1 not in modules:
            raise ValueError("density modules must include background group -1")
        if any(
            isinstance(group_id, bool)
            or not isinstance(group_id, int)
            or group_id < -1
            for group_id in modules
        ):
            raise ValueError(
                "density group IDs must be -1 or non-negative integers"
            )
        if len({id(module) for module in modules.values()}) != len(modules):
            raise ValueError("each density group must reference a unique module")
        self.modules = dict(sorted(modules.items()))
        if isinstance(policy, GaussianDensityPolicy):
            self.policies = {
                group_id: policy for group_id in self.modules
            }
        else:
            if set(policy) != set(self.modules):
                raise ValueError(
                    "density policy groups must match density modules"
                )
            if any(
                not isinstance(group_policy, GaussianDensityPolicy)
                for group_policy in policy.values()
            ):
                raise TypeError(
                    "each density group policy must be GaussianDensityPolicy"
                )
            self.policies = {
                group_id: policy[group_id] for group_id in self.modules
            }
        # Compatibility for trainer code that only needs the common lifecycle.
        # The background schedule is authoritative for that legacy property;
        # topology decisions always use policies[group_id].
        self.policy = self.policies[-1]
        self._accumulators = {
            group_id: GaussianDensityAccumulator.zeros(
                module.count, device=module.means.device
            )
            for group_id, module in self.modules.items()
        }
        self._pending_means2d: Tensor | None = None

    def accumulator(self, group_id: int) -> GaussianDensityAccumulator:
        if group_id not in self._accumulators:
            raise KeyError(f"unknown density group {group_id}")
        accumulator = self._accumulators[group_id]
        accumulator.ensure_layout(self.modules[group_id])
        return accumulator

    def statistics(self, group_id: int) -> GaussianDensityStats:
        return self.accumulator(group_id).statistics()

    def policy_for(self, group_id: int) -> GaussianDensityPolicy:
        if group_id not in self.policies:
            raise KeyError(f"unknown density group {group_id}")
        return self.policies[group_id]

    def before_backward(self, metadata: Mapping[str, Any]) -> None:
        if self._pending_means2d is not None:
            raise RuntimeError(
                "before_backward called before consuming prior metadata"
            )
        means2d = metadata.get("means2d")
        if not isinstance(means2d, Tensor):
            raise ValueError("gsplat metadata 'means2d' must be a tensor")
        if not means2d.requires_grad:
            raise ValueError("gsplat means2d must require gradients")
        means2d.retain_grad()
        self._pending_means2d = means2d

    def after_backward(
        self,
        metadata: Mapping[str, Any],
        composite_group_ids: Tensor,
        *,
        packed: bool | None = None,
    ) -> None:
        means2d = metadata.get("means2d")
        if self._pending_means2d is None:
            raise RuntimeError("before_backward must precede after_backward")
        if means2d is not self._pending_means2d:
            raise ValueError("after_backward received different means2d metadata")
        gradient = self._pending_means2d.grad
        if gradient is None:
            raise RuntimeError("means2d gradient is unavailable after backward")
        self._pending_means2d = None

        if (
            composite_group_ids.ndim != 1
            or composite_group_ids.dtype != torch.long
        ):
            raise ValueError("composite_group_ids must be a torch.long [N] tensor")
        gaussian_count = composite_group_ids.numel()
        inferred_packed = isinstance(metadata.get("gaussian_ids"), Tensor)
        use_packed = inferred_packed if packed is None else packed
        frame = _frame_screen_statistics(
            metadata,
            gradient,
            gaussian_count=gaussian_count,
            packed=use_packed,
        )
        group_ids = composite_group_ids.detach().to(
            device=frame[0].device
        )
        observed_groups = set(group_ids.unique().cpu().tolist())
        unknown = observed_groups - set(self.modules)
        if unknown:
            raise ValueError(
                f"composite metadata contains unknown groups {sorted(unknown)}"
            )

        selections: dict[int, Tensor] = {}
        for group_id in observed_groups:
            positions = torch.where(group_ids == group_id)[0]
            if positions.numel() != self.modules[group_id].count:
                raise ValueError(
                    f"group {group_id} has {positions.numel()} composite rows "
                    f"but module has {self.modules[group_id].count}"
                )
            selections[group_id] = positions

        for group_id, positions in selections.items():
            accumulator = self.accumulator(group_id)
            accumulator.add_(
                *(value.index_select(0, positions) for value in frame)
            )

    def apply_scheduled_updates(
        self,
        *,
        step: int,
        optimizer: torch.optim.Adam,
        reset_opacity: bool = False,
        generator: torch.Generator | None = None,
    ) -> dict[int, GaussianTopologyUpdateResult]:
        if self._pending_means2d is not None:
            raise RuntimeError(
                "cannot update topology before after_backward completes"
            )
        density_due = {
            group_id: policy.schedule.is_due(step)
            for group_id, policy in self.policies.items()
        }
        reset_due = {
            group_id: (
                reset_opacity and step < policy.schedule.end_step
            )
            for group_id, policy in self.policies.items()
        }
        if not any(density_due.values()) and not any(reset_due.values()):
            return {}

        results: dict[int, GaussianTopologyUpdateResult] = {}
        for group_id, module in self.modules.items():
            if not density_due[group_id] and not reset_due[group_id]:
                continue
            accumulator = self.accumulator(group_id)
            plan = self.policies[group_id].make_plan(
                module,
                accumulator.statistics(),
                step=step,
                reset_opacity=reset_due[group_id],
                generator=generator,
            )
            result = apply_gaussian_topology_update(
                module, plan, optimizer=optimizer
            )
            results[group_id] = result
            if density_due[group_id] or result.topology_changed:
                accumulator.reset_(
                    module.count, device=module.means.device
                )
        return results

    def reset_statistics(self) -> None:
        if self._pending_means2d is not None:
            raise RuntimeError(
                "cannot reset statistics while backward metadata is pending"
            )
        for group_id, module in self.modules.items():
            self._accumulators[group_id].reset_(
                module.count, device=module.means.device
            )

    def state_dict(self) -> dict[str, Any]:
        if self._pending_means2d is not None:
            raise RuntimeError(
                "cannot checkpoint between before_backward and after_backward"
            )
        return {
            "version": self._STATE_VERSION,
            "policies": {
                group_id: _serialized_density_policy(policy)
                for group_id, policy in self.policies.items()
            },
            "accumulators": {
                group_id: accumulator.state_dict()
                for group_id, accumulator in self._accumulators.items()
            },
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        strict_config: bool = True,
    ) -> None:
        if self._pending_means2d is not None:
            raise RuntimeError(
                "cannot restore while backward metadata is pending"
            )
        if state.get("version") != self._STATE_VERSION:
            raise ValueError("unsupported density controller state version")
        expected_policies = {
            group_id: _serialized_density_policy(policy)
            for group_id, policy in self.policies.items()
        }
        if strict_config and state.get("policies") != expected_policies:
            raise ValueError(
                "density policies differ from checkpoint"
            )
        saved = state.get("accumulators")
        if not isinstance(saved, Mapping):
            raise ValueError("checkpoint accumulators must be a mapping")
        if set(saved) != set(self.modules):
            raise ValueError(
                "checkpoint density groups differ from controller modules"
            )
        loaded: dict[int, GaussianDensityAccumulator] = {}
        for group_id, module in self.modules.items():
            group_state = saved[group_id]
            if not isinstance(group_state, Mapping):
                raise ValueError(
                    f"checkpoint group {group_id} must be a mapping"
                )
            accumulator = GaussianDensityAccumulator.zeros(
                module.count, device=module.means.device
            )
            accumulator.load_state_dict(
                group_state,
                count=module.count,
                device=module.means.device,
            )
            loaded[group_id] = accumulator
        self._accumulators = loaded


__all__ += [
    "GaussianDensityAccumulator",
    "GsplatDensityController",
]
