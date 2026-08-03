"""Composable ArmGS core independent of a particular CUDA rasterizer."""

from __future__ import annotations

from torch import Tensor, nn

from .actor import ActorDeformationRefiner
from .appearance import (
    FrameAppearanceEmbedding,
    GlobalImageAppearanceRefiner,
    LocalGaussianAppearanceRefiner,
    NearestFrameLookup,
)
from .spherical_harmonics import (
    camera_to_gaussian_directions,
    spherical_harmonics_to_rgb,
)
from .structures import GaussianSet


class ArmGSCore(nn.Module):
    """Own the three paper modules while leaving rasterization pluggable."""

    def __init__(
        self,
        frame_embeddings: FrameAppearanceEmbedding,
        local_refiner: LocalGaussianAppearanceRefiner,
        global_refiner: GlobalImageAppearanceRefiner,
        actor_refiner: ActorDeformationRefiner,
        nearest_frame_lookup: NearestFrameLookup | None = None,
    ) -> None:
        super().__init__()
        self.frame_embeddings = frame_embeddings
        self.local_refiner = local_refiner
        self.global_refiner = global_refiner
        self.actor_refiner = actor_refiner
        self.nearest_frame_lookup = nearest_frame_lookup

    def training_frame_embedding(self, training_rows: Tensor) -> Tensor:
        """Look up embeddings by contiguous training-table row."""

        return self.frame_embeddings(training_rows)

    def novel_view_embedding(
        self, query_camera_ids: Tensor, query_timestamps: Tensor
    ) -> Tensor:
        """Use the paper's nearest same-camera training frame policy."""

        if self.nearest_frame_lookup is None:
            raise RuntimeError("nearest-frame metadata was not attached to ArmGSCore")
        training_rows = self.nearest_frame_lookup(
            query_camera_ids, query_timestamps
        )
        return self.frame_embeddings(training_rows)

    def deform_actor(self, actor: GaussianSet, timestamp: Tensor) -> GaussianSet:
        if actor.sh_degree != self.actor_refiner.sh_degree:
            raise ValueError("actor SH degree does not match actor refiner")
        deformation = self.actor_refiner(
            actor.means, actor.sh_coefficients, timestamp
        )
        return actor.with_updates(
            means=deformation.means,
            sh_coefficients=deformation.sh_coefficients,
        )

    def refine_gaussian_colors(
        self,
        gaussians: GaussianSet,
        camera_center: Tensor,
        frame_embedding: Tensor,
    ) -> Tensor:
        directions = camera_to_gaussian_directions(gaussians.means, camera_center)
        colors = spherical_harmonics_to_rgb(
            gaussians.sh_coefficients, directions, gaussians.sh_degree
        )
        return self.local_refiner(gaussians.means, colors, frame_embedding)

    def refine_image(
        self,
        image: Tensor,
        frame_embedding: Tensor,
        camera_center: Tensor,
        view_direction: Tensor,
    ) -> Tensor:
        return self.global_refiner(
            image,
            frame_embedding,
            camera_center,
            view_direction,
        )

