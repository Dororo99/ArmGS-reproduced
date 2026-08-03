"""Load the explicit ArmGS reproduction assumptions from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
from .encodings import HashGridEncoder
from .losses import ArmGSLoss
from .model import ArmGSCore


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
