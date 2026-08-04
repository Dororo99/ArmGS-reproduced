"""Minimal end-to-end ArmGS training step and optimizer contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

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
        add_group(
            "sky",
            list(renderer.scene.sky.parameters()),
            float(learning_rates.get("sky", learning_rates["appearance"])),
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
        opacity_reset_interval: int | None = None,
        sh_degree_interval: int = 1_000,
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
        self.sampler = sampler
        self.density_controller = density_controller
        self.opacity_reset_interval = opacity_reset_interval
        if (
            isinstance(sh_degree_interval, bool)
            or not isinstance(sh_degree_interval, int)
            or sh_degree_interval <= 0
        ):
            raise ValueError("sh_degree_interval must be a positive integer")
        self.sh_degree_interval = sh_degree_interval
        self.step = 0
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
            opacity_reset_interval=reset_interval,
            sh_degree_interval=int(
                optimization.get("sh_degree_interval", 1_000)
            ),
        )

    def train_step(self, batch: ArmGSTrainingBatch) -> TrainingStepOutput:
        self.mean_scheduler.set_step(self.step)
        optimization_step = self.step + 1
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
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
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
        self.renderer.set_active_sh_degree(
            min(
                self.renderer.maximum_sh_degree,
                self.step // self.sh_degree_interval,
            )
        )


__all__ = [
    "ArmGSTrainer",
    "ArmGSTrainingBatch",
    "ExponentialMeanLRScheduler",
    "TrainingStepOutput",
    "build_armgs_optimizer",
]
