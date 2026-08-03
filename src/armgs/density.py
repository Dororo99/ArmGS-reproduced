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
from dataclasses import dataclass, replace
import math
from typing import Mapping

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
    """Paper-reported inclusive densification window.

    Defaults are the ArmGS values: every 100 optimization steps from step 500
    through step 15,000. Selection thresholds intentionally live elsewhere.
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
        if self.end_step < self.start_step:
            raise ValueError(
                "densification end_step must not precede start_step"
            )
        if self.interval <= 0:
            raise ValueError("densification interval must be positive")

    def is_due(self, step: int) -> bool:
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step < 0:
            raise ValueError("step cannot be negative")
        return (
            self.start_step <= step <= self.end_step
            and (step - self.start_step) % self.interval == 0
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
    minimum_gaussians: int = 1

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
    children. Duplicate parents must be retained. Split parents are replaced
    by their children. Unrepresented source rows are pruned.
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

        retained = set(self.retain_indices.detach().cpu().tolist())
        duplicated = set(self.duplicate_indices.detach().cpu().tolist())
        split = set(self.split_indices.detach().cpu().tolist())
        if not duplicated.issubset(retained):
            raise ValueError(
                "duplicate_indices must be a subset of retain_indices"
            )
        if retained.intersection(split):
            raise ValueError("split parents cannot also be retained")

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
    ) -> None:
        self.thresholds = thresholds
        self.schedule = schedule or DensificationSchedule()

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
        reset_value: float | None = None
        if reset_opacity:
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
        if self.thresholds.max_screen_radius is not None:
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
            prune = opacity < self.thresholds.prune_opacity_threshold
            if self.thresholds.max_screen_radius is not None:
                assert stats.max_screen_radius is not None
                prune = prune | (
                    stats.max_screen_radius.detach()
                    > self.thresholds.max_screen_radius
                )

            minimum = min(
                self.thresholds.minimum_gaussians, module.count
            )
            if int((~prune).sum()) < minimum:
                keep_best = torch.argsort(
                    opacity, descending=True, stable=True
                )[:minimum]
                prune[keep_best] = False

            high_gradient = (
                stats.position_gradient_norm.detach()
                >= self.thresholds.position_gradient_threshold
            ) & ~prune
            split_mask = high_gradient & (
                maximum_scale
                > self.thresholds.split_scale_threshold
            )
            duplicate_mask = high_gradient & ~split_mask
            split_indices = split_mask.nonzero(
                as_tuple=False
            ).squeeze(-1)
            duplicate_indices = duplicate_mask.nonzero(
                as_tuple=False
            ).squeeze(-1)
            retain_indices = (
                ~prune & ~split_mask
            ).nonzero(as_tuple=False).squeeze(-1)

            split_count = split_indices.numel()
            standard_offsets = torch.randn(
                split_count,
                self.thresholds.split_children,
                3,
                device=module.means.device,
                dtype=module.means.dtype,
                generator=generator,
            )
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
            value.fill_(
                torch.logit(
                    value.new_tensor(plan.opacity_reset_value)
                )
            )
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
