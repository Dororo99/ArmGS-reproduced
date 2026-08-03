"""Learnable background and actor scene containers for ArmGS."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from .geometry import PoseTrajectory, normalize_quaternion, transform_actor_gaussians
from .model import ArmGSCore
from .sky import ExplicitCubemapSky
from .structures import GaussianSet
from .time import TimestampNormalizer


class LearnableGaussianSet(nn.Module):
    """Register conventional raw 3DGS parameters and expose activated values.

    The public GaussianSet contract contains positive scales and probabilities.
    Internally this module stores log-scales and opacity logits, making the
    parameterization passed to gsplat explicit and preventing accidental double
    activation in integration code.
    """

    def __init__(self, initial: GaussianSet, *, inverse_sigmoid_eps: float = 1.0e-6) -> None:
        super().__init__()
        if not 0.0 < inverse_sigmoid_eps < 0.5:
            raise ValueError("inverse_sigmoid_eps must lie in (0, 0.5)")
        floating = (
            initial.means,
            initial.quaternions,
            initial.scales,
            initial.opacities,
            initial.sh_coefficients,
        )
        if not all(torch.isfinite(value.detach()).all() for value in floating):
            raise ValueError("initial Gaussian parameters must be finite")
        if torch.any(initial.scales.detach() <= 0):
            raise ValueError("initial scales must be positive")
        if torch.any(
            torch.linalg.vector_norm(initial.quaternions.detach(), dim=-1) < 1.0e-8
        ):
            raise ValueError("initial quaternions cannot be near zero")
        if torch.any(initial.opacities.detach() < 0) or torch.any(
            initial.opacities.detach() > 1
        ):
            raise ValueError("initial opacities must lie in [0, 1]")

        self.means = nn.Parameter(initial.means.detach().clone())
        self.quaternions = nn.Parameter(initial.quaternions.detach().clone())
        self.log_scales = nn.Parameter(initial.scales.detach().clone().log())
        probabilities = initial.opacities.detach().clone().clamp(
            inverse_sigmoid_eps, 1.0 - inverse_sigmoid_eps
        )
        self.opacity_logits = nn.Parameter(torch.logit(probabilities))
        self.sh_coefficients = nn.Parameter(
            initial.sh_coefficients.detach().clone()
        )
        group_ids = (
            initial.group_ids.detach().clone().to(dtype=torch.long)
            if initial.group_ids is not None
            else torch.empty(0, dtype=torch.long, device=initial.means.device)
        )
        self.register_buffer("_group_ids", group_ids)

    @property
    def count(self) -> int:
        return self.means.shape[0]

    @property
    def sh_degree(self) -> int:
        return int(self.sh_coefficients.shape[1] ** 0.5) - 1

    def activated(self, *, group_id: int | None = None) -> GaussianSet:
        if group_id is None:
            group_ids = self._group_ids if self._group_ids.numel() else None
        else:
            group_ids = torch.full(
                (self.count,),
                int(group_id),
                dtype=torch.long,
                device=self.means.device,
            )
        return GaussianSet(
            means=self.means,
            quaternions=normalize_quaternion(self.quaternions),
            scales=self.log_scales.exp(),
            opacities=self.opacity_logits.sigmoid(),
            sh_coefficients=self.sh_coefficients,
            group_ids=group_ids,
        )


class DynamicActorModel(nn.Module):
    """Canonical actor Gaussians plus a learnable timestamped world pose."""

    def __init__(
        self,
        gaussians: LearnableGaussianSet,
        trajectory: PoseTrajectory,
        *,
        actor_id: int,
        render_outside_track: bool = False,
    ) -> None:
        super().__init__()
        if actor_id < 0:
            raise ValueError("actor_id must be non-negative")
        self.gaussians = gaussians
        self.trajectory = trajectory
        self.actor_id = int(actor_id)
        self.render_outside_track = bool(render_outside_track)

    def is_active(self, timestamp: Tensor) -> bool:
        """Return whether this actor exists at the queried dataset timestamp."""

        if timestamp.numel() != 1:
            raise ValueError("actor activity query requires one scalar timestamp")
        if self.render_outside_track:
            return True
        query = timestamp.detach().reshape(()).to(self.trajectory.timestamps)
        return bool(
            (
                (query >= self.trajectory.timestamps[0])
                & (query <= self.trajectory.timestamps[-1])
            ).item()
        )

    def world_gaussians(
        self,
        core: ArmGSCore,
        timestamp: Tensor,
        normalized_timestamp: Tensor,
    ) -> GaussianSet:
        if timestamp.numel() != 1 or normalized_timestamp.numel() != 1:
            raise ValueError("actor rendering requires one scalar timestamp")
        canonical = self.gaussians.activated(group_id=self.actor_id)
        deformed = core.deform_actor(canonical, normalized_timestamp.reshape(()))
        pose = self.trajectory.interpolate(timestamp.reshape(()))
        return transform_actor_gaussians(
            deformed, pose.quaternions[0], pose.translations[0]
        )


class CompositeGaussianScene(nn.Module):
    """Background, object-centric actors, timestamp contract, and cubemap sky."""

    def __init__(
        self,
        background: LearnableGaussianSet,
        actors: Iterable[DynamicActorModel],
        timestamp_normalizer: TimestampNormalizer,
        *,
        sky: ExplicitCubemapSky | None = None,
    ) -> None:
        super().__init__()
        actor_list = list(actors)
        actor_ids = [actor.actor_id for actor in actor_list]
        if len(actor_ids) != len(set(actor_ids)):
            raise ValueError("actor ids must be unique")
        self.background = background
        self.actors = nn.ModuleList(actor_list)
        self.timestamp_normalizer = timestamp_normalizer
        self.sky = sky

    def gaussians_at(self, core: ArmGSCore, timestamp: Tensor) -> GaussianSet:
        if timestamp.numel() != 1:
            raise ValueError("scene rendering requires one scalar timestamp")
        background = self.background.activated(group_id=-1)
        sets = [background]
        if self.actors:
            normalized_timestamp = self.timestamp_normalizer(
                timestamp, reference=background.means
            )
            sets.extend(
                actor.world_gaussians(core, timestamp, normalized_timestamp)
                for actor in self.actors
                if actor.is_active(timestamp)
            )
        return GaussianSet.concatenate(sets)


__all__ = [
    "CompositeGaussianScene",
    "DynamicActorModel",
    "LearnableGaussianSet",
]
