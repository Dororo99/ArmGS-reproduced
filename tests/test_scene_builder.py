from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from armgs.data.schema import (
    ActorTrack,
    ActorTrackSample,
    CanonicalDatasetManifest,
    CanonicalFrame,
    LidarFrame,
    LidarProjection,
)
from armgs.initialization import GaussianInitializationConfig
from armgs.scene_builder import (
    CanonicalScenePointClouds,
    ColoredPointCloud,
    build_scene_from_point_clouds,
    collect_colored_lidar_point_clouds,
    merge_sfm_background,
)
from armgs.data.split import split_manifest_by_frame_indices


def make_manifest(tmp_path: Path) -> CanonicalDatasetManifest:
    image_path = tmp_path / "000000.png"
    scan_path = tmp_path / "000000.bin"
    image_path.write_bytes(b"image")
    scan_path.write_bytes(b"scan")
    lidar = LidarFrame(
        points=torch.tensor([[0.25, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        reflectance=torch.tensor([0.5, 0.8]),
        sensor_to_world=torch.eye(4),
        source_path=scan_path,
    )
    projection = LidarProjection(
        camera_id=2,
        source_point_indices=torch.tensor([0, 1]),
        image_coordinates=torch.tensor([[0.1, 0.1], [1.1, 0.1]]),
        pixel_indices=torch.tensor([[0, 0], [1, 0]]),
        depths=torch.tensor([2.0, 3.0]),
        image_size=(1, 2),
    )
    frame = CanonicalFrame(
        timestamp=torch.tensor(1_000_000_000, dtype=torch.int64),
        camera_id=2,
        camera_convention="opencv",
        camera_to_world=torch.eye(4),
        intrinsics=torch.eye(3),
        image_path=image_path,
        image_size=(1, 2),
        frame_index=0,
        lidar=lidar,
        lidar_projection=projection,
    )
    sample = ActorTrackSample(
        timestamp=frame.timestamp.clone(),
        translation=torch.zeros(3),
        quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        frame_index=0,
    )
    track = ActorTrack(
        actor_id=7,
        class_name="Car",
        dimensions_lwh=torch.tensor([2.0, 2.0, 2.0]),
        samples=(sample,),
    )
    return CanonicalDatasetManifest(frames=(frame,), actor_tracks=(track,))


def test_projected_lidar_is_colored_and_split_into_actor_and_background(
    tmp_path: Path,
) -> None:
    manifest = make_manifest(tmp_path)
    image = torch.tensor([[[255, 0, 0], [0, 255, 0]]], dtype=torch.uint8)

    def reader(path: Path, mode: str) -> torch.Tensor:
        assert path == manifest[0].image_path
        assert mode == "rgb"
        return image

    clouds = collect_colored_lidar_point_clouds(
        manifest, image_reader=reader
    )

    torch.testing.assert_close(
        clouds.background.points, torch.tensor([[3.0, 0.0, 0.0]])
    )
    torch.testing.assert_close(
        clouds.background.colors, torch.tensor([[0.0, 1.0, 0.0]])
    )
    assert set(clouds.actors) == {7}
    torch.testing.assert_close(
        clouds.actors[7].points, torch.tensor([[0.25, 0.0, 0.0]])
    )
    torch.testing.assert_close(
        clouds.actors[7].colors, torch.tensor([[1.0, 0.0, 0.0]])
    )


def test_lidar_actor_box_scale_expands_planar_axes_but_not_height(
    tmp_path: Path,
) -> None:
    source = make_manifest(tmp_path)
    source_frame = source[0]
    assert source_frame.lidar is not None
    scan_path = source_frame.lidar.source_path
    points = torch.tensor(
        [
            [1.5, 0.0, 0.0],  # admitted only by planar x expansion
            [0.0, 0.0, 1.5],  # remains outside the unscaled height
            [3.0, 0.0, 0.0],  # remains background
        ]
    )
    lidar = LidarFrame(
        points=points,
        reflectance=torch.ones(3),
        sensor_to_world=torch.eye(4),
        source_path=scan_path,
    )
    projection = LidarProjection(
        camera_id=source_frame.camera_id,
        source_point_indices=torch.arange(3),
        image_coordinates=torch.tensor(
            [[0.1, 0.1], [1.1, 0.1], [2.1, 0.1]]
        ),
        pixel_indices=torch.tensor([[0, 0], [1, 0], [2, 0]]),
        depths=torch.ones(3),
        image_size=(1, 3),
    )
    frame = replace(
        source_frame,
        image_size=(1, 3),
        lidar=lidar,
        lidar_projection=projection,
    )
    manifest = CanonicalDatasetManifest(
        frames=(frame,), actor_tracks=source.actor_tracks
    )
    image = torch.tensor(
        [[[255, 0, 0], [0, 255, 0], [0, 0, 255]]],
        dtype=torch.uint8,
    )

    clouds = collect_colored_lidar_point_clouds(
        manifest,
        image_reader=lambda *_: image,
        actor_box_scale=2.0,
    )

    torch.testing.assert_close(
        clouds.actors[7].points, torch.tensor([[1.5, 0.0, 0.0]])
    )
    torch.testing.assert_close(
        clouds.background.points,
        torch.tensor([[0.0, 0.0, 1.5], [3.0, 0.0, 0.0]]),
    )


def test_scene_builder_creates_background_actor_trajectory_and_time_contract(
    tmp_path: Path,
) -> None:
    manifest = make_manifest(tmp_path)
    clouds = CanonicalScenePointClouds(
        background=ColoredPointCloud(
            torch.tensor([[0.0, 0.0, 3.0]]),
            torch.tensor([[0.2, 0.3, 0.4]]),
        ),
        actors={
            7: ColoredPointCloud(
                torch.tensor([[0.0, 0.0, 0.0]]),
                torch.tensor([[0.8, 0.1, 0.2]]),
            )
        },
    )
    scene = build_scene_from_point_clouds(
        manifest,
        clouds,
        initialization=GaussianInitializationConfig(
            sh_degree=1,
            initial_opacity=0.1,
            initial_scale=0.05,
            voxel_size=None,
        ),
    )

    assert scene.background.count == 1
    assert len(scene.actors) == 1
    assert scene.actors[0].actor_id == 7
    assert scene.actors[0].gaussians.count == 1
    assert scene.actors[0].trajectory.timestamps.dtype == torch.float64
    assert scene.actors[0].is_active(manifest[0].timestamp)
    normalized = scene.timestamp_normalizer(
        manifest[0].timestamp, reference=scene.background.means
    )
    assert normalized.dtype == scene.background.means.dtype
    assert torch.isfinite(normalized)
    assert scene.actors[0].dimensions_lwh is not None
    assert scene.actors[0].density_extent() == pytest.approx(1.5)
    assert scene.actors[0].density_extent(actor_box_scale=2.0) == pytest.approx(
        1.5
    )


def test_training_track_retains_raw_lifecycle_and_extrapolates_boundary_pose(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image")
    frames = tuple(
        CanonicalFrame(
            timestamp=torch.tensor(timestamp, dtype=torch.int64),
            camera_id=0,
            camera_convention="opencv",
            camera_to_world=torch.eye(4),
            intrinsics=torch.eye(3),
            image_path=image_path,
            image_size=(1, 1),
            frame_index=frame_index,
        )
        for frame_index, timestamp in enumerate((0, 100, 200))
    )
    samples = tuple(
        ActorTrackSample(
            timestamp=frame.timestamp.clone(),
            translation=torch.tensor([float(frame.frame_index), 0.0, 0.0]),
            quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            frame_index=frame.frame_index,
        )
        for frame in frames
    )
    source = CanonicalDatasetManifest(
        frames=frames,
        actor_tracks=(
            ActorTrack(
                actor_id=3,
                class_name="Car",
                dimensions_lwh=torch.tensor([4.0, 2.0, 1.5]),
                samples=samples,
            ),
        ),
    )
    split = split_manifest_by_frame_indices(source, [0])
    train_track = split.train_manifest.actor_tracks[0]
    assert [sample.frame_index for sample in train_track.samples] == [1, 2]
    assert tuple(value.item() for value in train_track.lifecycle_timestamps) == (0, 200)

    scene = build_scene_from_point_clouds(
        split.train_manifest,
        CanonicalScenePointClouds(
            background=ColoredPointCloud(
                torch.tensor([[0.0, 0.0, 3.0]]),
                torch.tensor([[0.2, 0.3, 0.4]]),
            ),
            actors={
                3: ColoredPointCloud(
                    torch.tensor([[0.0, 0.0, 0.0]]),
                    torch.tensor([[0.8, 0.1, 0.2]]),
                )
            },
        ),
    )

    actor = scene.actors[0]
    assert actor.is_active(frames[0].timestamp)
    boundary_pose = actor.trajectory.interpolate(
        frames[0].timestamp, extrapolate=True
    )
    torch.testing.assert_close(boundary_pose.translations[0], torch.zeros(3))


def test_scene_builder_rejects_missing_or_unknown_actor_clouds(
    tmp_path: Path,
) -> None:
    manifest = make_manifest(tmp_path)
    background = ColoredPointCloud(
        torch.tensor([[0.0, 0.0, 3.0]]),
        torch.tensor([[0.5, 0.5, 0.5]]),
    )
    with pytest.raises(ValueError, match="missing point clouds"):
        build_scene_from_point_clouds(
            manifest, CanonicalScenePointClouds(background, {})
        )

    unknown = ColoredPointCloud(
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
    )
    with pytest.raises(ValueError, match="unknown actor"):
        build_scene_from_point_clouds(
            manifest,
            CanonicalScenePointClouds(background, {99: unknown}),
            require_all_actor_points=False,
        )


def test_collector_requires_background_and_valid_box_scale(
    tmp_path: Path,
) -> None:
    manifest = make_manifest(tmp_path)
    image = torch.tensor([[[255, 0, 0], [0, 255, 0]]], dtype=torch.uint8)

    with pytest.raises(ValueError, match="actor_box_scale"):
        collect_colored_lidar_point_clouds(
            manifest, image_reader=lambda *_: image, actor_box_scale=0.0
        )
    with pytest.raises(ValueError, match="no colored background"):
        collect_colored_lidar_point_clouds(
            manifest,
            image_reader=lambda *_: image,
            actor_box_scale=10.0,
        )


def test_sfm_background_merge_preserves_actor_local_clouds() -> None:
    actor_cloud = ColoredPointCloud(
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
    )
    clouds = CanonicalScenePointClouds(
        background=ColoredPointCloud(
            torch.tensor([[0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 1.0, 0.0]]),
        ),
        actors={7: actor_cloud},
    )

    merged = merge_sfm_background(
        clouds,
        torch.tensor([[2.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
    )

    assert merged.actors[7] is actor_cloud
    torch.testing.assert_close(
        merged.background.points,
        torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )


def test_scene_builder_supports_actor_specific_no_voxel_initialization(
    tmp_path: Path,
) -> None:
    manifest = make_manifest(tmp_path)
    clouds = CanonicalScenePointClouds(
        background=ColoredPointCloud(
            torch.tensor([[0.0, 0.0, 3.0], [0.1, 0.0, 3.0]]),
            torch.tensor([[0.2, 0.3, 0.4], [0.4, 0.3, 0.2]]),
        ),
        actors={
            7: ColoredPointCloud(
                torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
                torch.tensor([[0.8, 0.1, 0.2], [0.2, 0.1, 0.8]]),
            )
        },
    )
    shared = GaussianInitializationConfig(voxel_size=1.0)
    inherited = build_scene_from_point_clouds(
        manifest, clouds, initialization=shared
    )
    separate = build_scene_from_point_clouds(
        manifest,
        clouds,
        initialization=shared,
        actor_initialization=GaussianInitializationConfig(voxel_size=None),
    )

    assert inherited.background.count == 1
    assert inherited.actors[0].gaussians.count == 1
    assert separate.background.count == 1
    assert separate.actors[0].gaussians.count == 2
