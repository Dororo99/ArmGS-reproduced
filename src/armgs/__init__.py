"""Core modules for the ArmGS research implementation."""

from .actor import ActorDeformation, ActorDeformationRefiner
from .appearance import (
    FrameAppearanceEmbedding,
    GlobalImageAppearanceRefiner,
    LocalGaussianAppearanceRefiner,
    NearestFrameLookup,
    ViewpointEncoder,
)
from .geometry import PoseTrajectory, transform_actor_gaussians
from .losses import ArmGSLoss
from .model import ArmGSCore
from .pipeline import ArmGSCompositeRenderer, ArmGSRenderOutput, CameraView
from .scene import (
    CompositeGaussianScene,
    DynamicActorModel,
    LearnableGaussianSet,
)
from .sky import ExplicitCubemapSky
from .spherical_harmonics import spherical_harmonics_to_rgb
from .structures import GaussianSet, RasterizationInput, RasterizationOutput
from .time import TimestampNormalizer
from .training import ArmGSTrainer, ArmGSTrainingBatch

__all__ = [
    "ActorDeformation",
    "ActorDeformationRefiner",
    "ArmGSCompositeRenderer",
    "ArmGSCore",
    "ArmGSLoss",
    "ArmGSRenderOutput",
    "ArmGSTrainer",
    "ArmGSTrainingBatch",
    "CameraView",
    "CompositeGaussianScene",
    "DynamicActorModel",
    "ExplicitCubemapSky",
    "FrameAppearanceEmbedding",
    "GaussianSet",
    "GlobalImageAppearanceRefiner",
    "LearnableGaussianSet",
    "LocalGaussianAppearanceRefiner",
    "NearestFrameLookup",
    "PoseTrajectory",
    "RasterizationInput",
    "RasterizationOutput",
    "TimestampNormalizer",
    "ViewpointEncoder",
    "spherical_harmonics_to_rgb",
    "transform_actor_gaussians",
]
