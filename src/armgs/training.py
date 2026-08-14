"""Minimal end-to-end ArmGS training step and optimizer contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .density import (
    GaussianTopologyUpdateResult,
    GsplatDensityController,
)
from .losses import ArmGSLoss, LossBreakdown
from .pipeline import ArmGSCompositeRenderer, ArmGSRenderOutput, CameraView
from .sampling import StatefulShuffleSampler
from .scene import LearnableGaussianSet


@dataclass(frozen=True)
class ArmGSTrainingBatch:
    view: CameraView
    target_rgb: Tensor
    lidar_depth: Tensor | None = None
    depth_valid_mask: Tensor | None = None
    target_sky_mask: Tensor | None = None
    actor_bbox_mask: Tensor | None = None
    sky_valid_mask: Tensor | None = None


@dataclass(frozen=True)
class TrainingStepOutput:
    rendering: ArmGSRenderOutput
    losses: LossBreakdown
    step: int
    density_updates: dict[int, GaussianTopologyUpdateResult] | None = None


def _gaussian_modules(renderer: ArmGSCompositeRenderer) -> list[LearnableGaussianSet]:
    return [
        renderer.scene.background,
        *(actor.gaussians for actor in renderer.scene.actors),
    ]

_RAW_GAUSSIAN_PARAMETER_NAMES = (
    "means",
    "quaternions",
    "log_scales",
    "opacity_logits",
    "sh_coefficients",
)

_CONFIGURABLE_GROUP_LR_SCHEDULES = frozenset(
    {
        "actor_pose_translation",
        "actor_pose_rotation",
        "sky",
    }
)


@dataclass(frozen=True)
class ExponentialGroupLRScheduleSpec:
    """One named optimizer group's global-step exponential schedule."""

    initial: float
    final: float
    max_steps: int
    warmup_steps: int = 0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.initial)
            or not math.isfinite(self.final)
            or self.initial <= 0.0
            or self.final <= 0.0
        ):
            raise ValueError(
                "scheduled learning rates must be finite and positive"
            )
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps <= 0
        ):
            raise ValueError("schedule max_steps must be a positive integer")
        if (
            isinstance(self.warmup_steps, bool)
            or not isinstance(self.warmup_steps, int)
            or self.warmup_steps < 0
            or self.warmup_steps > self.max_steps
        ):
            raise ValueError(
                "schedule warmup_steps must be an integer in [0, max_steps]"
            )


def _group_lr_schedule_specs(
    config: Mapping[str, Any],
) -> dict[str, ExponentialGroupLRScheduleSpec]:
    optimization = config["optimization"]
    raw_schedules = optimization.get("learning_rate_schedules")
    if raw_schedules is None:
        return {}
    if not isinstance(raw_schedules, Mapping):
        raise TypeError("optimization.learning_rate_schedules must be a mapping")
    unknown = set(raw_schedules) - _CONFIGURABLE_GROUP_LR_SCHEDULES
    if unknown:
        raise ValueError(
            "unsupported optimizer group LR schedules: "
            + ", ".join(sorted(str(name) for name in unknown))
        )

    specs: dict[str, ExponentialGroupLRScheduleSpec] = {}
    for group_name, raw_spec in raw_schedules.items():
        if not isinstance(raw_spec, Mapping):
            raise TypeError(
                f"LR schedule for {group_name!r} must be a mapping"
            )
        allowed = {"initial", "final", "max_steps", "warmup_steps"}
        unexpected = set(raw_spec) - allowed
        if unexpected:
            raise ValueError(
                f"unexpected LR schedule keys for {group_name!r}: "
                + ", ".join(sorted(str(name) for name in unexpected))
            )
        missing = {"initial", "final"} - set(raw_spec)
        if missing:
            raise ValueError(
                f"LR schedule for {group_name!r} is missing: "
                + ", ".join(sorted(missing))
            )
        specs[str(group_name)] = ExponentialGroupLRScheduleSpec(
            initial=float(raw_spec["initial"]),
            final=float(raw_spec["final"]),
            max_steps=int(
                raw_spec.get("max_steps", optimization["iterations"])
            ),
            warmup_steps=int(raw_spec.get("warmup_steps", 0)),
        )
    return specs


def _resize_gaussians_for_checkpoint(
    renderer: ArmGSCompositeRenderer,
    optimizer: torch.optim.Optimizer,
    renderer_state: dict[str, Tensor],
) -> None:
    """Match Gaussian row topology before strict model/optimizer restore."""

    for module_name, module in renderer.named_modules():
        if not isinstance(module, LearnableGaussianSet):
            continue
        prefix = f"{module_name}." if module_name else ""
        saved_count: int | None = None
        for parameter_name in _RAW_GAUSSIAN_PARAMETER_NAMES:
            state_key = prefix + parameter_name
            saved = renderer_state.get(state_key)
            if not isinstance(saved, Tensor):
                continue
            if saved.ndim == 0:
                raise ValueError(
                    f"checkpoint Gaussian parameter {state_key!r} must have rows"
                )
            if saved_count is None:
                saved_count = saved.shape[0]
            elif saved.shape[0] != saved_count:
                raise ValueError(
                    f"checkpoint Gaussian rows disagree for module {module_name!r}"
                )
            old = getattr(module, parameter_name)
            if old.shape == saved.shape:
                continue
            matches: list[tuple[dict[str, Any], int]] = []
            for group in optimizer.param_groups:
                parameters = group["params"]
                for index, parameter in enumerate(parameters):
                    if parameter is old:
                        matches.append((group, index))
            if len(matches) != 1:
                raise ValueError(
                    f"Gaussian parameter {state_key!r} must occur once in optimizer"
                )
            new = nn.Parameter(
                torch.empty(
                    saved.shape, device=old.device, dtype=old.dtype
                ),
                requires_grad=old.requires_grad,
            )
            group, index = matches[0]
            group["params"][index] = new
            optimizer.state.pop(old, None)
            setattr(module, parameter_name, new)

        group_state = renderer_state.get(prefix + "_group_ids")
        if isinstance(group_state, Tensor) and module._group_ids.shape != group_state.shape:
            module._buffers["_group_ids"] = torch.empty(
                group_state.shape,
                device=module._group_ids.device,
                dtype=torch.long,
            )


def build_armgs_optimizer(
    renderer: ArmGSCompositeRenderer,
    config: dict[str, Any],
    *,
    background_extent: float = 1.0,
    actor_box_scale: float = 1.0,
) -> torch.optim.Adam:
    """Create reference 3DGS parameter groups without omitting parameters.

    The position learning rates are spatially scaled per scene component:
    camera-normalization radius for the background and box-derived extent for
    each object-centric actor, matching the StreetGS composite convention.
    """

    learning_rates = config["optimization"]["learning_rates"]
    group_lr_schedules = _group_lr_schedule_specs(config)
    gaussian_modules = _gaussian_modules(renderer)
    groups: list[dict[str, Any]] = []
    for name, value in (
        ("background_extent", background_extent),
        ("actor_box_scale", actor_box_scale),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    def add_group(
        name: str,
        parameters: list[Tensor],
        learning_rate: float,
        **metadata: Any,
    ) -> None:
        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        if trainable:
            if not math.isfinite(learning_rate) or learning_rate <= 0.0:
                raise ValueError(
                    f"learning rate for trainable group {name!r} must be finite and positive"
                )
            groups.append(
                {
                    "name": name,
                    "params": trainable,
                    "lr": learning_rate,
                    **metadata,
                }
            )

    add_group(
        "means/background",
        [renderer.scene.background.means],
        float(learning_rates["mean_initial"]) * background_extent,
        spatial_lr_scale=float(background_extent),
    )
    for actor in renderer.scene.actors:
        extent = (
            actor.density_extent(actor_box_scale=actor_box_scale)
            if (
                hasattr(actor, "density_extent")
                and getattr(actor, "dimensions_lwh", None) is not None
            )
            else float(background_extent)
        )
        add_group(
            f"means/actor/{actor.actor_id}",
            [actor.gaussians.means],
            float(learning_rates["mean_initial"]) * extent,
            spatial_lr_scale=float(extent),
        )
    add_group(
        "rotations",
        [module.quaternions for module in gaussian_modules],
        float(learning_rates["rotation"]),
    )
    add_group(
        "scales",
        [module.log_scales for module in gaussian_modules],
        float(learning_rates["scale"]),
    )
    add_group(
        "opacities",
        [module.opacity_logits for module in gaussian_modules],
        float(learning_rates["opacity"]),
    )
    add_group(
        "spherical_harmonics",
        [module.sh_coefficients for module in gaussian_modules],
        float(learning_rates["sh"]),
    )
    actor_schedule_names = {
        "actor_pose_translation",
        "actor_pose_rotation",
    }
    if actor_schedule_names & group_lr_schedules.keys():
        translation_schedule = group_lr_schedules.get(
            "actor_pose_translation"
        )
        rotation_schedule = group_lr_schedules.get("actor_pose_rotation")
        add_group(
            "actor_pose_translation",
            [
                actor.trajectory.translations
                for actor in renderer.scene.actors
            ],
            (
                translation_schedule.initial
                if translation_schedule is not None
                else float(learning_rates["actor_pose"])
            ),
        )
        add_group(
            "actor_pose_rotation",
            [
                actor.trajectory.quaternions
                for actor in renderer.scene.actors
            ],
            (
                rotation_schedule.initial
                if rotation_schedule is not None
                else float(learning_rates["actor_pose"])
            ),
        )
    else:
        add_group(
            "actor_pose",
            [
                parameter
                for actor in renderer.scene.actors
                for parameter in (
                    actor.trajectory.quaternions,
                    actor.trajectory.translations,
                )
            ],
            float(learning_rates["actor_pose"]),
        )
    add_group(
        "appearance",
        [
            *renderer.core.frame_embeddings.parameters(),
            *renderer.core.local_refiner.parameters(),
            *renderer.core.global_refiner.parameters(),
        ],
        float(learning_rates["appearance"]),
    )
    add_group(
        "actor_deformation",
        list(renderer.core.actor_refiner.parameters()),
        float(learning_rates["actor_deformation"]),
    )
    if renderer.scene.sky is not None:
        sky_schedule = group_lr_schedules.get("sky")
        add_group(
            "sky",
            list(renderer.scene.sky.parameters()),
            (
                sky_schedule.initial
                if sky_schedule is not None
                else float(
                    learning_rates.get("sky", learning_rates["appearance"])
                )
            ),
        )

    grouped_ids = [
        id(parameter)
        for group in groups
        for parameter in group["params"]
    ]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError("a trainable parameter appears in multiple optimizer groups")
    expected = {
        id(parameter)
        for parameter in renderer.parameters()
        if parameter.requires_grad
    }
    actual = set(grouped_ids)
    if actual != expected:
        missing = len(expected - actual)
        unexpected = len(actual - expected)
        raise RuntimeError(
            "optimizer parameter coverage mismatch: "
            f"{missing} missing, {unexpected} unexpected"
        )
    return torch.optim.Adam(groups, eps=1.0e-15)


class ExponentialMeanLRScheduler:
    """Paper-reported exponential mean LR decay with explicit step indexing."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        initial: float,
        final: float,
        total_steps: int,
    ) -> None:
        if (
            not math.isfinite(initial)
            or not math.isfinite(final)
            or initial <= 0
            or final <= 0
        ):
            raise ValueError(
                "mean learning rates must be finite and positive"
            )
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        self.optimizer = optimizer
        self.initial = float(initial)
        self.final = float(final)
        self.total_steps = int(total_steps)

    def learning_rate_at(self, step: int) -> float:
        if step < 0:
            raise ValueError("step cannot be negative")
        denominator = max(self.total_steps - 1, 1)
        fraction = min(step, self.total_steps - 1) / denominator
        return self.initial * (self.final / self.initial) ** fraction

    def set_step(self, step: int) -> float:
        learning_rate = self.learning_rate_at(step)
        found = False
        for group in self.optimizer.param_groups:
            if str(group.get("name", "")).startswith("means/"):
                spatial_scale = float(group.get("spatial_lr_scale", 1.0))
                if not math.isfinite(spatial_scale) or spatial_scale <= 0.0:
                    raise RuntimeError(
                        "mean optimizer group has an invalid spatial_lr_scale"
                    )
                group["lr"] = learning_rate * spatial_scale
                found = True
        if not found:
            raise RuntimeError("optimizer has no component mean parameter groups")
        return learning_rate


class ExponentialGroupLRScheduler:
    """Update explicitly named optimizer groups from a global training step.

    Warmed-up groups stay frozen at zero before the boundary. Once active, the
    exponential fraction remains step / max_steps instead of restarting at the
    warmup boundary, matching the StreetGS pose schedule.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        schedules: Mapping[str, ExponentialGroupLRScheduleSpec],
    ) -> None:
        if not schedules:
            raise ValueError("at least one optimizer group schedule is required")
        self.optimizer = optimizer
        self.schedules = dict(schedules)
        self._validate_optimizer_groups()

    def _optimizer_groups(self) -> dict[str, dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for group in self.optimizer.param_groups:
            name = str(group.get("name", ""))
            if name in groups:
                raise RuntimeError(f"duplicate optimizer group name {name!r}")
            groups[name] = group
        return groups

    def _validate_optimizer_groups(self) -> None:
        groups = self._optimizer_groups()
        missing = set(self.schedules) - set(groups)
        if missing:
            raise RuntimeError(
                "optimizer is missing scheduled groups: "
                + ", ".join(sorted(missing))
            )

    def learning_rate_at(self, group_name: str, step: int) -> float:
        if step < 0:
            raise ValueError("step cannot be negative")
        try:
            spec = self.schedules[group_name]
        except KeyError as error:
            raise KeyError(
                f"optimizer group {group_name!r} is not scheduled"
            ) from error
        if step < spec.warmup_steps:
            return 0.0
        fraction = min(step, spec.max_steps) / spec.max_steps
        return spec.initial * (spec.final / spec.initial) ** fraction

    def set_step(self, step: int) -> dict[str, float]:
        self._validate_optimizer_groups()
        groups = self._optimizer_groups()
        learning_rates = {
            name: self.learning_rate_at(name, step)
            for name in self.schedules
        }
        for name, learning_rate in learning_rates.items():
            groups[name]["lr"] = learning_rate
        return learning_rates

    def state_dict(self) -> dict[str, Any]:
        return {
            "schedules": {
                name: {
                    "initial": spec.initial,
                    "final": spec.final,
                    "max_steps": spec.max_steps,
                    "warmup_steps": spec.warmup_steps,
                }
                for name, spec in sorted(self.schedules.items())
            }
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if dict(state) != self.state_dict():
            raise ValueError(
                "checkpoint optimizer group LR schedules differ from trainer"
            )


class ArmGSTrainer:
    """One-view training loop that connects rendering and equation (9)."""

    def __init__(
        self,
        renderer: ArmGSCompositeRenderer,
        loss: ArmGSLoss,
        optimizer: torch.optim.Optimizer,
        mean_scheduler: ExponentialMeanLRScheduler,
        sampler: StatefulShuffleSampler | None = None,
        density_controller: GsplatDensityController | None = None,
        *,
        group_lr_scheduler: ExponentialGroupLRScheduler | None = None,
        opacity_reset_interval: int | None = None,
        sh_degree_interval: int = 1_000,
        foreground_start_iteration: int = 0,
    ) -> None:
        if density_controller is not None:
            if not isinstance(optimizer, torch.optim.Adam):
                raise TypeError("density control requires torch.optim.Adam")
            expected_modules = {
                -1: renderer.scene.background,
                **{
                    actor.actor_id: actor.gaussians
                    for actor in renderer.scene.actors
                },
            }
            if set(density_controller.modules) != set(expected_modules) or any(
                density_controller.modules[group_id] is not module
                for group_id, module in expected_modules.items()
            ):
                raise ValueError(
                    "density controller modules must match renderer background/actors"
                )
        if opacity_reset_interval is not None:
            if density_controller is None:
                raise ValueError(
                    "opacity_reset_interval requires a density controller"
                )
            if (
                isinstance(opacity_reset_interval, bool)
                or not isinstance(opacity_reset_interval, int)
                or opacity_reset_interval <= 0
            ):
                raise ValueError("opacity_reset_interval must be a positive integer")
        self.renderer = renderer
        self.loss = loss
        self.optimizer = optimizer
        self.mean_scheduler = mean_scheduler
        self.group_lr_scheduler = group_lr_scheduler
        self.sampler = sampler
        self.density_controller = density_controller
        self.opacity_reset_interval = opacity_reset_interval
        if (
            isinstance(sh_degree_interval, bool)
            or not isinstance(sh_degree_interval, int)
            or sh_degree_interval <= 0
        ):
            raise ValueError("sh_degree_interval must be a positive integer")
        if (
            isinstance(foreground_start_iteration, bool)
            or not isinstance(foreground_start_iteration, int)
            or foreground_start_iteration < 0
        ):
            raise ValueError(
                "foreground_start_iteration must be a non-negative integer"
            )
        self.sh_degree_interval = sh_degree_interval
        self.foreground_start_iteration = foreground_start_iteration
        self.step = 0
        if self.group_lr_scheduler is not None:
            self.group_lr_scheduler.set_step(0)
        self.renderer.set_active_sh_degree(0)

    @classmethod
    def from_config(
        cls,
        renderer: ArmGSCompositeRenderer,
        loss: ArmGSLoss,
        config: dict[str, Any],
        *,
        sampler: StatefulShuffleSampler | None = None,
        density_controller: GsplatDensityController | None = None,
        background_extent: float = 1.0,
        actor_box_scale: float = 1.0,
    ) -> "ArmGSTrainer":
        optimizer = build_armgs_optimizer(
            renderer,
            config,
            background_extent=background_extent,
            actor_box_scale=actor_box_scale,
        )
        optimization = config["optimization"]
        rates = optimization["learning_rates"]
        scheduler = ExponentialMeanLRScheduler(
            optimizer,
            initial=float(rates["mean_initial"]),
            final=float(rates["mean_final"]),
            total_steps=int(optimization["iterations"]),
        )
        group_schedule_specs = _group_lr_schedule_specs(config)
        group_lr_scheduler = (
            ExponentialGroupLRScheduler(optimizer, group_schedule_specs)
            if group_schedule_specs
            else None
        )
        reset_interval: int | None = None
        if density_controller is not None:
            density_config = optimization.get("densification", {})
            configured = density_config.get("opacity_reset_interval")
            reset_interval = int(configured) if configured is not None else None
        return cls(
            renderer,
            loss,
            optimizer,
            scheduler,
            sampler,
            density_controller,
            group_lr_scheduler=group_lr_scheduler,
            opacity_reset_interval=reset_interval,
            sh_degree_interval=int(
                optimization.get("sh_degree_interval", 1_000)
            ),
            foreground_start_iteration=config["loss"].get(
                "foreground_start_iteration", 0
            ),
        )

    def train_step(self, batch: ArmGSTrainingBatch) -> TrainingStepOutput:
        optimization_step = self.step + 1
        self.mean_scheduler.set_step(self.step)
        if self.group_lr_scheduler is not None:
            self.group_lr_scheduler.set_step(optimization_step)
        self.renderer.set_active_sh_degree(
            min(
                self.renderer.maximum_sh_degree,
                optimization_step // self.sh_degree_interval,
            )
        )
        density_active = (
            self.density_controller is not None
            and optimization_step
            < self.density_controller.policy.schedule.end_step
        )
        self.optimizer.zero_grad(set_to_none=True)
        rendering = self.renderer(batch.view)
        losses = self.loss(
            rendering.rgb,
            batch.target_rgb.to(rendering.rgb),
            rendered_depth=(
                rendering.depth if batch.lidar_depth is not None else None
            ),
            lidar_depth=(
                batch.lidar_depth.to(rendering.depth)
                if batch.lidar_depth is not None
                else None
            ),
            depth_valid_mask=batch.depth_valid_mask,
            non_sky_accumulated_alpha=(
                rendering.non_sky_accumulated_alpha
                if batch.target_sky_mask is not None
                else None
            ),
            target_sky_mask=(
                batch.target_sky_mask.to(rendering.non_sky_accumulated_alpha)
                if batch.target_sky_mask is not None
                else None
            ),
            sky_valid_mask=(
                batch.sky_valid_mask.to(
                    device=rendering.non_sky_accumulated_alpha.device,
                    dtype=torch.bool,
                )
                if batch.sky_valid_mask is not None
                else None
            ),
            actor_alpha=rendering.actor_alpha,
            actor_bbox_mask=(
                batch.actor_bbox_mask.to(rendering.actor_alpha)
                if batch.actor_bbox_mask is not None
                and rendering.actor_alpha is not None
                else batch.actor_bbox_mask
            ),
            foreground_active=(
                optimization_step >= self.foreground_start_iteration
            ),
        )
        density_metadata = rendering.rasterization.metadata
        composite_group_ids = rendering.composite_gaussians.group_ids
        if density_active:
            if density_metadata is None:
                raise ValueError(
                    "density control requires gsplat rasterization metadata"
                )
            if composite_group_ids is None:
                raise ValueError(
                    "density control requires composite Gaussian group_ids"
                )
            self.density_controller.before_backward(density_metadata)

        losses.total.backward()
        if density_active:
            assert density_metadata is not None
            assert composite_group_ids is not None
            self.density_controller.after_backward(
                density_metadata, composite_group_ids
            )
        completed_step = self.step
        density_updates: dict[int, GaussianTopologyUpdateResult] | None = None
        if density_active:
            reset_opacity = (
                self.opacity_reset_interval is not None
                and optimization_step % self.opacity_reset_interval == 0
            )
            assert isinstance(self.optimizer, torch.optim.Adam)
            density_updates = self.density_controller.apply_scheduled_updates(
                step=optimization_step,
                optimizer=self.optimizer,
                reset_opacity=reset_opacity,
            )
        self.optimizer.step()
        self.step += 1
        return TrainingStepOutput(
            rendering, losses, completed_step, density_updates
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "renderer": self.renderer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "sampler_state": (
                self.sampler.state_dict() if self.sampler is not None else None
            ),
            "density_state": (
                self.density_controller.state_dict()
                if self.density_controller is not None
                else None
            ),
            "opacity_reset_interval": self.opacity_reset_interval,
            "sh_degree_interval": self.sh_degree_interval,
            "foreground_start_iteration": self.foreground_start_iteration,
            "group_lr_scheduler": (
                self.group_lr_scheduler.state_dict()
                if self.group_lr_scheduler is not None
                else None
            ),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        checkpoint_group_lr_scheduler = state.get("group_lr_scheduler")
        if self.group_lr_scheduler is None:
            if checkpoint_group_lr_scheduler is not None:
                raise ValueError(
                    "checkpoint has optimizer group LR schedules but trainer does not"
                )
        else:
            if not isinstance(checkpoint_group_lr_scheduler, Mapping):
                raise ValueError(
                    "checkpoint is missing optimizer group LR schedules"
                )
            self.group_lr_scheduler.load_state_dict(
                checkpoint_group_lr_scheduler
            )
        renderer_state = state["renderer"]
        if not isinstance(renderer_state, dict):
            raise ValueError("checkpoint renderer state must be a mapping")
        _resize_gaussians_for_checkpoint(
            self.renderer, self.optimizer, renderer_state
        )
        self.renderer.load_state_dict(renderer_state)
        self.optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        if step < 0:
            raise ValueError("checkpoint step cannot be negative")
        self.step = step

        checkpoint_interval = state.get("opacity_reset_interval")
        if checkpoint_interval != self.opacity_reset_interval:
            raise ValueError(
                "checkpoint opacity reset interval differs from trainer"
            )
        checkpoint_sh_interval = state.get("sh_degree_interval", 1_000)
        if checkpoint_sh_interval != self.sh_degree_interval:
            raise ValueError(
                "checkpoint SH degree interval differs from trainer"
            )
        checkpoint_foreground_start = state.get(
            "foreground_start_iteration", 0
        )
        if (
            isinstance(checkpoint_foreground_start, bool)
            or not isinstance(checkpoint_foreground_start, int)
            or checkpoint_foreground_start < 0
        ):
            raise ValueError(
                "checkpoint foreground start iteration must be a "
                "non-negative integer"
            )
        if checkpoint_foreground_start != self.foreground_start_iteration:
            raise ValueError(
                "checkpoint foreground start iteration differs from trainer"
            )
        density_state = state.get("density_state")
        if density_state is not None:
            if self.density_controller is None:
                raise ValueError(
                    "checkpoint contains density state but trainer has no controller"
                )
            self.density_controller.load_state_dict(density_state)
        elif self.density_controller is not None:
            raise ValueError(
                "trainer has density controller but checkpoint has no density state"
            )

        sampler_state = state.get("sampler_state")
        if sampler_state is not None:
            if self.sampler is None:
                raise ValueError(
                    "checkpoint contains sampler state but trainer has no sampler"
                )
            self.sampler.load_state_dict(sampler_state)
        if "torch_rng_state" in state:
            cpu_rng_state = state["torch_rng_state"]
            if (
                not isinstance(cpu_rng_state, Tensor)
                or cpu_rng_state.dtype != torch.uint8
            ):
                raise ValueError("torch_rng_state must be a uint8 tensor")
            torch.set_rng_state(cpu_rng_state.detach().cpu())
        cuda_rng_state = state.get("cuda_rng_state_all")
        if cuda_rng_state is not None and torch.cuda.is_available():
            if not isinstance(cuda_rng_state, (list, tuple)):
                raise ValueError("cuda_rng_state_all must be a sequence")
            for device_index, rng_state in enumerate(
                cuda_rng_state[: torch.cuda.device_count()]
            ):
                if (
                    not isinstance(rng_state, Tensor)
                    or rng_state.dtype != torch.uint8
                ):
                    raise ValueError("CUDA RNG states must be uint8 tensors")
                torch.cuda.set_rng_state(
                    rng_state.detach().cpu(), device=device_index
                )
        self.mean_scheduler.set_step(self.step)
        if self.group_lr_scheduler is not None:
            self.group_lr_scheduler.set_step(self.step)
        self.renderer.set_active_sh_degree(
            min(
                self.renderer.maximum_sh_degree,
                self.step // self.sh_degree_interval,
            )
        )


__all__ = [
    "ArmGSTrainer",
    "ArmGSTrainingBatch",
    "ExponentialGroupLRScheduler",
    "ExponentialGroupLRScheduleSpec",
    "ExponentialMeanLRScheduler",
    "TrainingStepOutput",
    "build_armgs_optimizer",
]
