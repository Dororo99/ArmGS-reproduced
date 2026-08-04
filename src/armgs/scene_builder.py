"""Build learnable ArmGS scenes from canonical dataset point clouds.

KITTI LiDAR does not contain RGB.  The collector projects each scan into its
camera image, samples color, and separates points using canonical actor boxes.
The symmetric, actor-centred box test is an explicit reproduction assumption;
dataset adapters with a different box origin should supply precomputed local
actor clouds instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor

from .batching import ImageTensorReader, pillow_image_reader
from .data.schema import (
    ActorTrack,
    ActorTrackSample,
    CanonicalDatasetManifest,
)
from .geometry import PoseTrajectory, quaternion_to_rotation_matrix
from .initialization import (
    GaussianInitializationConfig,
    initialize_gaussians_from_points,
    merge_colored_point_clouds,
    world_points_to_actor_local,
)
from .scene import (
    CompositeGaussianScene,
    DynamicActorModel,
    LearnableGaussianSet,
)
from .sky import ExplicitCubemapSky
from .time import TimestampNormalizer


@dataclass(frozen=True)
class ColoredPointCloud:
    """Finite XYZ/RGB point cloud with colors in [0, 1]."""

    points: Tensor
    colors: Tensor

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must have shape [N,3]")
        if self.colors.shape != self.points.shape:
            raise ValueError("colors must have shape [N,3]")
        if self.points.shape[0] == 0:
            raise ValueError("point cloud cannot be empty")
        if not self.points.is_floating_point() or not self.colors.is_floating_point():
            raise ValueError("points and colors must be floating point")
        if self.points.device != self.colors.device:
            raise ValueError("points and colors must share a device")
        if self.points.dtype != self.colors.dtype:
            raise ValueError("points and colors must share a dtype")
        if not torch.isfinite(self.points).all() or not torch.isfinite(self.colors).all():
            raise ValueError("point cloud values must be finite")
        if torch.any((self.colors < 0.0) | (self.colors > 1.0)):
            raise ValueError("colors must lie in [0,1]")


@dataclass(frozen=True)
class CanonicalScenePointClouds:
    """Background world points and actor-local point clouds."""

    background: ColoredPointCloud
    actors: Mapping[int, ColoredPointCloud]

    def __post_init__(self) -> None:
        actor_ids = list(self.actors)
        if any(isinstance(actor_id, bool) or actor_id < 0 for actor_id in actor_ids):
            raise ValueError("actor point-cloud ids must be non-negative")
        if len(actor_ids) != len(set(actor_ids)):
            raise ValueError("actor point-cloud ids must be unique")


def merge_sfm_background(
    point_clouds: CanonicalScenePointClouds,
    sfm_points: Tensor,
    sfm_colors: Tensor,
    *,
    voxel_size: float | None = None,
) -> CanonicalScenePointClouds:
    """Merge known-pose, world-aligned SfM points into the LiDAR background.

    ArmGS initializes the static model from both LiDAR and COLMAP points.  This
    function deliberately performs no implicit Sim(3) alignment: callers must
    triangulate with the dataset camera poses (the StreetGS route) or otherwise
    provide points already expressed in the canonical world frame.
    """

    reference = point_clouds.background.points
    points, colors = merge_colored_point_clouds(
        (
            (
                point_clouds.background.points,
                point_clouds.background.colors,
            ),
            (
                sfm_points.to(reference),
                sfm_colors.to(reference),
            ),
        ),
        voxel_size=voxel_size,
    )
    return CanonicalScenePointClouds(
        background=ColoredPointCloud(points, colors),
        actors=point_clouds.actors,
    )


def _normalized_rgb(image: Tensor, image_size: tuple[int, int]) -> Tensor:
    height, width = image_size
    if image.shape != (height, width, 3):
        raise ValueError(
            f"RGB reader returned {tuple(image.shape)}, expected {(height, width, 3)}"
        )
    if image.dtype == torch.uint8:
        image = image.to(torch.float32) / 255.0
    elif image.is_floating_point():
        image = image.to(torch.float32)
    else:
        raise ValueError("RGB reader must return uint8 or floating point")
    if not torch.isfinite(image).all() or torch.any((image < 0.0) | (image > 1.0)):
        raise ValueError("RGB values must be finite and lie in [0,1]")
    return image


def _actor_to_world(sample: ActorTrackSample, reference: Tensor) -> Tensor:
    transform = torch.eye(4, dtype=reference.dtype, device=reference.device)
    transform[:3, :3] = quaternion_to_rotation_matrix(
        sample.quaternion_wxyz.to(reference)
    )
    transform[:3, 3] = sample.translation.to(reference)
    return transform


def collect_colored_lidar_point_clouds(
    manifest: CanonicalDatasetManifest,
    *,
    image_reader: ImageTensorReader | None = None,
    actor_box_scale: float = 1.0,
) -> CanonicalScenePointClouds:
    """Color projected LiDAR and separate centered actor boxes.

    Only points visible in a requested camera receive RGB.  Multiple cameras or
    frames may contribute duplicate samples; the Gaussian initializer's voxel
    downsampling fuses them deterministically.
    """

    if not math.isfinite(actor_box_scale) or actor_box_scale <= 0.0:
        raise ValueError("actor_box_scale must be finite and positive")
    reader = pillow_image_reader if image_reader is None else image_reader
    samples_by_frame: dict[int, list[tuple[ActorTrack, ActorTrackSample]]] = {}
    for track in manifest.actor_tracks:
        for sample in track.samples:
            samples_by_frame.setdefault(sample.frame_index, []).append((track, sample))

    background_points: list[Tensor] = []
    background_colors: list[Tensor] = []
    actor_points: dict[int, list[Tensor]] = {}
    actor_colors: dict[int, list[Tensor]] = {}
    for frame in manifest:
        if frame.lidar is None or frame.lidar_projection is None:
            continue
        projection = frame.lidar_projection
        if projection.source_point_indices.numel() == 0:
            continue
        image = _normalized_rgb(
            reader(frame.image_path, "rgb"), frame.image_size
        )
        source_indices = projection.source_point_indices.to(
            device=frame.lidar.points.device
        )
        world = frame.lidar.world_points.index_select(0, source_indices)
        pixels = projection.pixel_indices.to(device=image.device)
        x, y = pixels.unbind(dim=-1)
        colors = image[y, x].to(device=world.device, dtype=world.dtype)
        assigned = torch.zeros(world.shape[0], dtype=torch.bool, device=world.device)

        for track, sample in samples_by_frame.get(frame.frame_index, ()):
            local, inside = world_points_to_actor_local(
                world,
                _actor_to_world(sample, world),
                box_dimensions=track.dimensions_lwh.to(world) * actor_box_scale,
            )
            take = inside & ~assigned
            if not take.any():
                continue
            # local contains only inside rows, so recompute all local coordinates
            # when overlapping boxes require a subset of those rows.
            if torch.equal(take, inside):
                selected_local = local
            else:
                all_local, _ = world_points_to_actor_local(
                    world, _actor_to_world(sample, world)
                )
                selected_local = all_local[take]
            actor_points.setdefault(track.actor_id, []).append(selected_local)
            actor_colors.setdefault(track.actor_id, []).append(colors[take])
            assigned |= take

        if (~assigned).any():
            background_points.append(world[~assigned])
            background_colors.append(colors[~assigned])

    if not background_points:
        raise ValueError("no colored background LiDAR points were collected")
    background = ColoredPointCloud(
        torch.cat(background_points, dim=0),
        torch.cat(background_colors, dim=0),
    )
    actors = {
        actor_id: ColoredPointCloud(
            torch.cat(point_sets, dim=0),
            torch.cat(actor_colors[actor_id], dim=0),
        )
        for actor_id, point_sets in actor_points.items()
    }
    return CanonicalScenePointClouds(background=background, actors=actors)


def actor_track_to_trajectory(
    track: ActorTrack,
    *,
    reference: Tensor,
) -> PoseTrajectory:
    """Convert a canonical actor track to a learnable pose trajectory."""

    timestamps = torch.stack([sample.timestamp for sample in track.samples])
    quaternions = torch.stack(
        [sample.quaternion_wxyz for sample in track.samples]
    ).to(reference)
    translations = torch.stack(
        [sample.translation for sample in track.samples]
    ).to(reference)
    return PoseTrajectory(timestamps, quaternions, translations)


def build_scene_from_point_clouds(
    manifest: CanonicalDatasetManifest,
    point_clouds: CanonicalScenePointClouds,
    *,
    initialization: GaussianInitializationConfig | None = None,
    sky: ExplicitCubemapSky | None = None,
    require_all_actor_points: bool = True,
) -> CompositeGaussianScene:
    """Initialize background and tracked actor modules from colored points."""

    initialization = initialization or GaussianInitializationConfig()
    background = LearnableGaussianSet(
        initialize_gaussians_from_points(
            point_clouds.background.points,
            point_clouds.background.colors,
            config=initialization,
        )
    )
    tracks = {track.actor_id: track for track in manifest.actor_tracks}
    unknown = set(point_clouds.actors) - set(tracks)
    if unknown:
        raise ValueError(f"point clouds reference unknown actor ids: {sorted(unknown)}")
    missing = set(tracks) - set(point_clouds.actors)
    if require_all_actor_points and missing:
        raise ValueError(f"missing point clouds for actor ids: {sorted(missing)}")

    actors: list[DynamicActorModel] = []
    reference = background.means
    for actor_id in sorted(set(tracks) & set(point_clouds.actors)):
        cloud = point_clouds.actors[actor_id]
        actor_gaussians = LearnableGaussianSet(
            initialize_gaussians_from_points(
                cloud.points.to(reference),
                cloud.colors.to(reference),
                config=initialization,
            )
        )
        actors.append(
            DynamicActorModel(
                actor_gaussians,
                actor_track_to_trajectory(tracks[actor_id], reference=reference),
                actor_id=actor_id,
                lifecycle_timestamps=tracks[actor_id].lifecycle_timestamps,
                dimensions_lwh=tracks[actor_id].dimensions_lwh,
            )
        )

    timestamps = torch.stack([frame.timestamp for frame in manifest])
    normalizer = TimestampNormalizer.from_timestamps(timestamps)
    return CompositeGaussianScene(
        background,
        actors,
        normalizer,
        sky=sky,
    )


__all__ = [
    "CanonicalScenePointClouds",
    "ColoredPointCloud",
    "actor_track_to_trajectory",
    "build_scene_from_point_clouds",
    "collect_colored_lidar_point_clouds",
    "merge_sfm_background",
]
