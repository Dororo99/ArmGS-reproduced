"""Core modules for the ArmGS research implementation."""

from .actor import ActorDeformation, ActorDeformationRefiner
from .appearance import (
    FrameAppearanceEmbedding,
    GlobalImageAppearanceRefiner,
    LocalGaussianAppearanceRefiner,
    NearestFrameLookup,
    ViewpointEncoder,
)
from .batching import (
    canonical_frame_to_training_batch,
    pillow_image_reader,
)
from .density import (
    DensificationSchedule,
    DensityControlThresholds,
    GaussianDensityAccumulator,
    GaussianDensityPolicy,
    GaussianDensityStats,
    GaussianTopologyUpdatePlan,
    GaussianTopologyUpdateResult,
    GsplatDensityController,
    apply_gaussian_topology_update,
)
from .evaluation import (
    EvaluationAccumulator,
    ImageMetrics,
    LPIPSMetric,
    evaluate_image_pair,
    peak_signal_noise_ratio,
    project_actor_boxes_to_mask,
)
from .geometry import PoseTrajectory, transform_actor_gaussians
from .initialization import (
    GaussianInitializationConfig,
    estimate_knn_isotropic_scales,
    initialize_gaussians_from_points,
    load_colmap_points3d_text,
    merge_colored_point_clouds,
    voxel_downsample,
    world_points_to_actor_local,
)
from .losses import ArmGSLoss
from .model import ArmGSCore
from .pipeline import ArmGSCompositeRenderer, ArmGSRenderOutput, CameraView
from .sampling import StatefulShuffleSampler
from .scene import (
    CompositeGaussianScene,
    DynamicActorModel,
    LearnableGaussianSet,
)
from .scene_builder import (
    CanonicalScenePointClouds,
    ColoredPointCloud,
    build_scene_from_point_clouds,
    collect_colored_lidar_point_clouds,
    merge_sfm_background,
)
from .sky import ExplicitCubemapSky
from .spherical_harmonics import spherical_harmonics_to_rgb
from .structures import GaussianSet, RasterizationInput, RasterizationOutput
from .time import TimestampNormalizer
from .training import (
    ArmGSTrainer,
    ArmGSTrainingBatch,
    TrainingStepOutput,
)

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
    "CanonicalScenePointClouds",
    "ColoredPointCloud",
    "CompositeGaussianScene",
    "DensificationSchedule",
    "DensityControlThresholds",
    "DynamicActorModel",
    "EvaluationAccumulator",
    "ExplicitCubemapSky",
    "FrameAppearanceEmbedding",
    "GaussianDensityAccumulator",
    "GaussianDensityPolicy",
    "GaussianDensityStats",
    "GaussianInitializationConfig",
    "GaussianSet",
    "GaussianTopologyUpdatePlan",
    "GaussianTopologyUpdateResult",
    "GlobalImageAppearanceRefiner",
    "GsplatDensityController",
    "ImageMetrics",
    "LPIPSMetric",
    "LearnableGaussianSet",
    "LocalGaussianAppearanceRefiner",
    "NearestFrameLookup",
    "PoseTrajectory",
    "RasterizationInput",
    "RasterizationOutput",
    "StatefulShuffleSampler",
    "TimestampNormalizer",
    "TrainingStepOutput",
    "ViewpointEncoder",
    "apply_gaussian_topology_update",
    "build_scene_from_point_clouds",
    "canonical_frame_to_training_batch",
    "collect_colored_lidar_point_clouds",
    "evaluate_image_pair",
    "estimate_knn_isotropic_scales",
    "initialize_gaussians_from_points",
    "load_colmap_points3d_text",
    "merge_colored_point_clouds",
    "merge_sfm_background",
    "peak_signal_noise_ratio",
    "pillow_image_reader",
    "project_actor_boxes_to_mask",
    "spherical_harmonics_to_rgb",
    "transform_actor_gaussians",
    "voxel_downsample",
    "world_points_to_actor_local",
]
