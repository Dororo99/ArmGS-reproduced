"""Load the explicit ArmGS reproduction assumptions from YAML."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch import Tensor

from .actor import ActorDeformationRefiner
from .appearance import (
    FrameAppearanceEmbedding,
    GlobalImageAppearanceRefiner,
    LocalGaussianAppearanceRefiner,
    NearestFrameLookup,
    ViewpointEncoder,
)
from .density import (
    DensificationSchedule,
    DensityControlThresholds,
    GaussianDensityPolicy,
    GsplatDensityController,
)
from .encodings import HashGridEncoder
from .initialization import GaussianInitializationConfig
from .losses import ArmGSLoss
from .model import ArmGSCore
from .sampling import StatefulShuffleSampler
from .scene import CompositeGaussianScene
from .sky import ExplicitCubemapSky


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    for required in ("model", "optimization", "loss"):
        if required not in config:
            raise ValueError(f"configuration is missing '{required}'")
    return config


def build_core(
    config: dict[str, Any],
    *,
    num_training_frames: int,
    training_camera_ids: Tensor | None = None,
    training_timestamps: Tensor | None = None,
    scene_aabb_min: tuple[float, float, float] = (-1.0, -1.0, -1.0),
    scene_aabb_max: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> ArmGSCore:
    model_config = config["model"]
    frame_embedding_dim = int(model_config["frame_embedding_dim"])
    sh_degree = int(model_config["sh_degree"])

    local_config = model_config["local"]
    hash_config = dict(local_config["hash_grid"])
    hash_chunk_size = hash_config.pop("chunk_size", None)
    hash_config["aabb_min"] = scene_aabb_min
    hash_config["aabb_max"] = scene_aabb_max
    hash_encoder = HashGridEncoder(**hash_config)
    local_refiner = LocalGaussianAppearanceRefiner(
        hash_encoder,
        frame_embedding_dim,
        hidden_dim=int(local_config["hidden_dim"]),
        num_layers=int(local_config["num_layers"]),
        hash_chunk_size=(
            int(hash_chunk_size) if hash_chunk_size is not None else None
        ),
        scale_delta_limit=(
            float(local_config["scale_delta_limit"])
            if local_config.get("scale_delta_limit") is not None
            else None
        ),
        bias_limit=(
            float(local_config["bias_limit"])
            if local_config.get("bias_limit") is not None
            else None
        ),
    )

    global_config = model_config["global"]
    viewpoint_encoder = ViewpointEncoder(
        position_frequencies=int(global_config["position_frequencies"]),
        direction_frequencies=int(global_config["direction_frequencies"]),
        aabb_min=scene_aabb_min,
        aabb_max=scene_aabb_max,
    )
    global_refiner = GlobalImageAppearanceRefiner(
        viewpoint_encoder,
        frame_embedding_dim,
        hidden_dim=int(global_config["hidden_dim"]),
        num_layers=int(global_config["num_layers"]),
        matrix_delta_limit=(
            float(global_config["matrix_delta_limit"])
            if global_config.get("matrix_delta_limit") is not None
            else None
        ),
        bias_limit=(
            float(global_config["bias_limit"])
            if global_config.get("bias_limit") is not None
            else None
        ),
    )

    actor_config = model_config["actor"]
    actor_refiner = ActorDeformationRefiner(
        sh_degree,
        hidden_dim=int(actor_config["hidden_dim"]),
        position_frequencies=int(actor_config["position_frequencies"]),
        time_frequencies=int(actor_config["time_frequencies"]),
        encoder_layers=int(actor_config["encoder_layers"]),
        head_layers=int(actor_config["head_layers"]),
    )
    if (training_camera_ids is None) != (training_timestamps is None):
        raise ValueError(
            "training_camera_ids and training_timestamps must be supplied together"
        )
    nearest_lookup = (
        NearestFrameLookup(training_camera_ids, training_timestamps)
        if training_camera_ids is not None and training_timestamps is not None
        else None
    )
    return ArmGSCore(
        FrameAppearanceEmbedding(num_training_frames, frame_embedding_dim),
        local_refiner,
        global_refiner,
        actor_refiner,
        nearest_lookup,
    )


def build_loss(config: dict[str, Any]) -> ArmGSLoss:
    loss_config = config["loss"]
    return ArmGSLoss(
        lambda_ssim=float(loss_config["lambda_ssim"]),
        require_auxiliary=bool(loss_config.get("require_auxiliary", False)),
        lambda_depth=float(loss_config["lambda_depth"]),
        lambda_sky=float(loss_config["lambda_sky"]),
        lambda_foreground=float(loss_config["lambda_foreground"]),
        ssim_window_size=int(loss_config.get("ssim_window_size", 11)),
        ssim_sigma=float(loss_config.get("ssim_sigma", 1.5)),
        ssim_data_range=float(loss_config.get("ssim_data_range", 1.0)),
    )

def build_initialization_config(
    config: dict[str, Any],
) -> GaussianInitializationConfig:
    initialization = config.get("initialization")
    if not isinstance(initialization, dict):
        raise ValueError("configuration is missing 'initialization'")
    voxel_size = initialization.get("voxel_size")
    return GaussianInitializationConfig(
        sh_degree=int(config["model"]["sh_degree"]),
        initial_opacity=float(initialization["initial_opacity"]),
        initial_scale=float(initialization["initial_scale"]),
        voxel_size=float(voxel_size) if voxel_size is not None else None,
        knn_neighbors=int(initialization.get("knn_neighbors", 3)),
        knn_chunk_size=int(initialization.get("knn_chunk_size", 1024)),
        knn_backend=str(initialization.get("knn_backend", "auto")),
        minimum_squared_distance=float(
            initialization.get("minimum_squared_distance", 1.0e-7)
        ),
    )


def build_sky(config: dict[str, Any]) -> ExplicitCubemapSky:
    sky_config = config["model"].get("sky")
    if not isinstance(sky_config, dict):
        raise ValueError("configuration is missing 'model.sky'")
    return ExplicitCubemapSky(
        int(sky_config["resolution"]),
        initial_color=tuple(float(value) for value in sky_config["initial_color"]),
    )


def build_density_policy(
    config: dict[str, Any],
    *,
    scene_scale: float,
    actor_box_half_extents: tuple[float, float, float] | None = None,
) -> GaussianDensityPolicy:
    if not math.isfinite(scene_scale) or scene_scale <= 0.0:
        raise ValueError("scene_scale must be finite and positive")
    density = config["optimization"].get("densification")
    if not isinstance(density, dict):
        raise ValueError("configuration is missing 'optimization.densification'")
    maximum_radius = density.get("max_screen_radius")
    world_scale_fraction = density.get(
        "prune_world_scale_fraction_of_scene", 0.1
    )
    reset_value = density.get("opacity_reset_value")
    schedule = DensificationSchedule(
        start_step=int(density["start_iteration"]),
        end_step=int(density["end_iteration"]),
        interval=int(density["interval"]),
    )
    thresholds = DensityControlThresholds(
        position_gradient_threshold=float(
            density["position_gradient_threshold"]
        ),
        split_scale_threshold=(
            float(density["split_scale_fraction_of_scene"]) * scene_scale
        ),
        prune_opacity_threshold=float(
            density["prune_opacity_threshold"]
        ),
        split_children=int(density["split_children"]),
        split_scale_reduction=float(density["split_scale_reduction"]),
        opacity_reset_value=(
            float(reset_value) if reset_value is not None else None
        ),
        max_screen_radius=(
            float(maximum_radius) if maximum_radius is not None else None
        ),
        max_world_scale=(
            float(world_scale_fraction) * scene_scale
            if world_scale_fraction is not None
            else None
        ),
        prune_large_after_step=int(
            density.get(
                "prune_large_after_iteration",
                density.get("opacity_reset_interval", 3_000),
            )
        ),
        minimum_gaussians=int(density.get("minimum_gaussians", 0)),
    )
    return GaussianDensityPolicy(
        thresholds,
        schedule=schedule,
        actor_box_half_extents=actor_box_half_extents,
    )


def build_sampler(
    config: dict[str, Any],
    *,
    dataset_size: int,
) -> StatefulShuffleSampler:
    sampler = config.get("data", {}).get("sampler")
    if not isinstance(sampler, dict):
        raise ValueError("configuration is missing 'data.sampler'")
    workers = int(sampler.get("num_workers", 0))
    if workers != 0:
        raise ValueError(
            "exact mid-epoch checkpoint resume requires data.sampler.num_workers=0"
        )
    return StatefulShuffleSampler(
        dataset_size,
        seed=int(sampler.get("seed", 0)),
        shuffle=bool(sampler.get("shuffle", True)),
    )


def build_density_controller(
    config: dict[str, Any],
    scene: CompositeGaussianScene,
    *,
    scene_scale: float,
    actor_box_scale: float = 1.0,
    group_scene_scales: Mapping[int, float] | None = None,
) -> GsplatDensityController:
    if not math.isfinite(actor_box_scale) or actor_box_scale <= 0.0:
        raise ValueError("actor_box_scale must be finite and positive")
    modules = {
        -1: scene.background,
        **{
            actor.actor_id: actor.gaussians
            for actor in scene.actors
        },
    }
    if group_scene_scales is not None:
        if set(group_scene_scales) != set(modules):
            raise ValueError(
                "group_scene_scales must match background/actor groups"
            )
        resolved_scales = {
            group_id: float(group_scene_scales[group_id])
            for group_id in modules
        }
    else:
        resolved_scales = {
            -1: scene_scale,
            **{
                actor.actor_id: (
                    actor.density_extent(actor_box_scale=actor_box_scale)
                    if getattr(actor, "dimensions_lwh", None) is not None
                    else scene_scale
                )
                for actor in scene.actors
            },
        }
    density = config["optimization"].get("densification")
    if not isinstance(density, dict):
        raise ValueError(
            "configuration is missing 'optimization.densification'"
        )
    prune_actor_outside_box = density.get(
        "prune_actor_outside_box", False
    )
    if not isinstance(prune_actor_outside_box, bool):
        raise TypeError("prune_actor_outside_box must be a boolean")
    actor_half_extents: dict[int, tuple[float, float, float]] = {}
    if prune_actor_outside_box:
        for actor in scene.actors:
            dimensions = getattr(actor, "dimensions_lwh", None)
            if dimensions is None:
                raise ValueError(
                    "prune_actor_outside_box requires actor dimensions_lwh"
                )
            # Waymo/StreetGS expands only the ground-plane length and width.
            # Scene actors intentionally retain the raw tracker dimensions.
            effective = dimensions.detach().to(
                device="cpu", dtype=torch.float64
            ).clone()
            effective[:2] *= actor_box_scale
            actor_half_extents[actor.actor_id] = tuple(
                float(value) for value in (effective * 0.5).tolist()
            )
    policy = {
        group_id: build_density_policy(
            config,
            scene_scale=resolved_scales[group_id],
            actor_box_half_extents=actor_half_extents.get(group_id),
        )
        for group_id in modules
    }
    return GsplatDensityController(modules, policy)
