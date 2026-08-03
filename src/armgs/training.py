"""Minimal end-to-end ArmGS training step and optimizer contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor

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


@dataclass(frozen=True)
class TrainingStepOutput:
    rendering: ArmGSRenderOutput
    losses: LossBreakdown
    step: int


def _gaussian_modules(renderer: ArmGSCompositeRenderer) -> list[LearnableGaussianSet]:
    return [
        renderer.scene.background,
        *(actor.gaussians for actor in renderer.scene.actors),
    ]


def build_armgs_optimizer(
    renderer: ArmGSCompositeRenderer, config: dict[str, Any]
) -> torch.optim.Adam:
    """Create the paper parameter groups without silently omitting parameters."""

    learning_rates = config["optimization"]["learning_rates"]
    gaussian_modules = _gaussian_modules(renderer)
    groups: list[dict[str, Any]] = []

    def add_group(name: str, parameters: list[Tensor], learning_rate: float) -> None:
        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        if trainable:
            if not math.isfinite(learning_rate) or learning_rate <= 0.0:
                raise ValueError(
                    f"learning rate for trainable group {name!r} must be finite and positive"
                )
            groups.append({"name": name, "params": trainable, "lr": learning_rate})

    add_group(
        "means",
        [module.means for module in gaussian_modules],
        float(learning_rates["mean_initial"]),
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
    return torch.optim.Adam(groups)


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
            if group.get("name") == "means":
                group["lr"] = learning_rate
                found = True
        if not found:
            raise RuntimeError("optimizer has no means parameter group")
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
    ) -> None:
        self.renderer = renderer
        self.loss = loss
        self.optimizer = optimizer
        self.mean_scheduler = mean_scheduler
        self.sampler = sampler
        self.step = 0

    @classmethod
    def from_config(
        cls,
        renderer: ArmGSCompositeRenderer,
        loss: ArmGSLoss,
        config: dict[str, Any],
        *,
        sampler: StatefulShuffleSampler | None = None,
    ) -> "ArmGSTrainer":
        optimizer = build_armgs_optimizer(renderer, config)
        optimization = config["optimization"]
        rates = optimization["learning_rates"]
        scheduler = ExponentialMeanLRScheduler(
            optimizer,
            initial=float(rates["mean_initial"]),
            final=float(rates["mean_final"]),
            total_steps=int(optimization["iterations"]),
        )
        return cls(renderer, loss, optimizer, scheduler, sampler)

    def train_step(self, batch: ArmGSTrainingBatch) -> TrainingStepOutput:
        self.mean_scheduler.set_step(self.step)
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
            actor_alpha=rendering.actor_alpha,
            actor_bbox_mask=(
                batch.actor_bbox_mask.to(rendering.actor_alpha)
                if batch.actor_bbox_mask is not None
                and rendering.actor_alpha is not None
                else batch.actor_bbox_mask
            ),
        )
        losses.total.backward()
        self.optimizer.step()
        completed_step = self.step
        self.step += 1
        return TrainingStepOutput(rendering, losses, completed_step)

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
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.renderer.load_state_dict(state["renderer"])
        self.optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        if step < 0:
            raise ValueError("checkpoint step cannot be negative")
        self.step = step
        sampler_state = state.get("sampler_state")
        if sampler_state is not None:
            if self.sampler is None:
                raise ValueError("checkpoint contains sampler state but trainer has no sampler")
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


__all__ = [
    "ArmGSTrainer",
    "ArmGSTrainingBatch",
    "ExponentialMeanLRScheduler",
    "TrainingStepOutput",
    "build_armgs_optimizer",
]
