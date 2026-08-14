from __future__ import annotations

from pathlib import Path

import pytest
import torch

from armgs.appearance import NearestFrameLookup
from armgs.data.schema import (
    ActorTrack,
    ActorTrackSample,
    CanonicalDatasetManifest,
    CanonicalFrame,
)
from armgs.data.split import (
    linspace_train_eval_split,
    periodic_train_eval_split,
    split_manifest_by_frame_indices,
)


def _frame(
    image_path: Path,
    *,
    frame_index: int,
    timestamp: int,
    camera_id: int,
    capture_timestamp: int | None = None,
) -> CanonicalFrame:
    return CanonicalFrame(
        timestamp=torch.tensor(timestamp, dtype=torch.int64),
        camera_id=camera_id,
        camera_convention="opencv",
        camera_to_world=torch.eye(4, dtype=torch.float64),
        intrinsics=torch.eye(3, dtype=torch.float64),
        image_path=image_path,
        image_size=(8, 12),
        frame_index=frame_index,
        capture_timestamp=(
            torch.tensor(capture_timestamp, dtype=torch.int64)
            if capture_timestamp is not None
            else None
        ),
    )


def _track() -> ActorTrack:
    return ActorTrack(
        actor_id=7,
        class_name="Car",
        dimensions_lwh=torch.tensor([4.0, 2.0, 1.5]),
        samples=(
            ActorTrackSample(
                timestamp=torch.tensor(100, dtype=torch.int64),
                translation=torch.zeros(3),
                quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                frame_index=10,
            ),
            ActorTrackSample(
                timestamp=torch.tensor(400, dtype=torch.int64),
                translation=torch.ones(3),
                quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                frame_index=40,
            ),
        ),
    )


def _manifest(tmp_path: Path) -> CanonicalDatasetManifest:
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    frames = tuple(
        _frame(
            image,
            frame_index=frame_index,
            timestamp=timestamp,
            camera_id=camera_id,
        )
        for frame_index, timestamp in ((10, 100), (20, 200), (30, 300), (40, 400))
        for camera_id in (0, 1)
    )
    return CanonicalDatasetManifest(frames=frames, actor_tracks=(_track(),))


def test_periodic_split_is_capture_atomic_and_builds_embedding_metadata(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)

    split = periodic_train_eval_split(manifest, every=2, offset=1)

    assert [frame.frame_index for frame in split.train_manifest] == [10, 10, 30, 30]
    assert [frame.frame_index for frame in split.eval_manifest] == [20, 20, 40, 40]
    train_track = split.train_manifest.actor_tracks[0]
    eval_track = split.eval_manifest.actor_tracks[0]
    assert train_track is not manifest.actor_tracks[0]
    assert eval_track is not manifest.actor_tracks[0]
    assert [sample.frame_index for sample in train_track.samples] == [10]
    assert [sample.frame_index for sample in eval_track.samples] == [40]
    assert tuple(value.item() for value in train_track.lifecycle_timestamps) == (
        100,
        400,
    )
    assert tuple(value.item() for value in eval_track.lifecycle_timestamps) == (
        100,
        400,
    )
    assert split.train_source_indices == (0, 1, 4, 5)
    assert split.eval_source_indices == (2, 3, 6, 7)
    assert dict(split.source_index_to_training_row) == {0: 0, 1: 1, 4: 2, 5: 3}
    assert split.training_row(4) == 2
    with pytest.raises(ValueError, match="not in the training split"):
        split.training_row(2)
    assert split.training_camera_ids.tolist() == [0, 1, 0, 1]
    assert split.training_timestamps.tolist() == [100, 100, 300, 300]
    assert split.training_camera_ids.is_contiguous()
    assert split.training_timestamps.is_contiguous()

    lookup = NearestFrameLookup(
        split.training_camera_ids, split.training_timestamps
    )
    assert lookup(
        torch.tensor([0, 1]), torch.tensor([200, 400], dtype=torch.int64)
    ).tolist() == [0, 3]


def test_linspace_split_matches_splatad_50_percent_and_is_capture_atomic(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    manifest = CanonicalDatasetManifest(
        frames=tuple(
            _frame(
                image,
                frame_index=frame_index,
                timestamp=100 + frame_index,
                camera_id=camera_id,
            )
            for frame_index in range(8)
            for camera_id in (0, 1)
        )
    )

    split = linspace_train_eval_split(manifest, train_fraction=0.5)

    # np.linspace(0, 7, ceil(8 * 0.5), dtype=int64) == [0, 2, 4, 7].
    assert {frame.frame_index for frame in split.train_manifest} == {0, 2, 4, 7}
    assert {frame.frame_index for frame in split.eval_manifest} == {1, 3, 5, 6}
    assert split.train_source_indices == (0, 1, 4, 5, 8, 9, 14, 15)
    assert split.eval_source_indices == (2, 3, 6, 7, 10, 11, 12, 13)


@pytest.mark.parametrize("fraction", [0.0, 1.0, float("nan"), float("inf")])
def test_linspace_split_rejects_non_holdout_fractions(
    tmp_path: Path,
    fraction: float,
) -> None:
    with pytest.raises(ValueError, match="0 < value < 1"):
        linspace_train_eval_split(_manifest(tmp_path), train_fraction=fraction)


def test_split_drops_eval_only_actor_and_held_out_pose_from_training(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    eval_only_actor = ActorTrack(
        actor_id=8,
        class_name="Pedestrian",
        dimensions_lwh=torch.tensor([0.8, 0.6, 1.7]),
        samples=(
            ActorTrackSample(
                timestamp=torch.tensor(200, dtype=torch.int64),
                translation=torch.tensor([1000.0, 0.0, 0.0]),
                quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                frame_index=20,
            ),
        ),
    )
    source = CanonicalDatasetManifest(
        frames=manifest.frames,
        actor_tracks=manifest.actor_tracks + (eval_only_actor,),
    )

    split = split_manifest_by_frame_indices(source, [20, 40])

    assert [track.actor_id for track in split.train_manifest.actor_tracks] == [7]
    assert [track.actor_id for track in split.eval_manifest.actor_tracks] == [7, 8]
    training_samples = split.train_manifest.actor_tracks[0].samples
    assert [sample.frame_index for sample in training_samples] == [10]
    # Trajectory and AABB consumers receive only this filtered training track.
    assert torch.stack([sample.translation for sample in training_samples]).amax() == 0


def test_periodic_split_uses_timestamp_order_but_preserves_source_row_order(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    # Source rows need not be chronological. Capture ordering remains deterministic.
    manifest = CanonicalDatasetManifest(
        frames=(
            _frame(image, frame_index=30, timestamp=300, camera_id=0),
            _frame(image, frame_index=10, timestamp=100, camera_id=0),
            _frame(image, frame_index=20, timestamp=200, camera_id=0),
        )
    )

    split = periodic_train_eval_split(manifest, every=2, offset=0)

    assert [frame.frame_index for frame in split.eval_manifest] == [30, 10]
    assert [frame.frame_index for frame in split.train_manifest] == [20]
    assert split.eval_source_indices == (0, 1)
    assert split.train_source_indices == (2,)


def test_periodic_split_can_keep_initial_waymo_capture_in_training(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    manifest = CanonicalDatasetManifest(
        frames=tuple(
            _frame(
                image,
                frame_index=frame_index,
                timestamp=100 + frame_index,
                camera_id=camera_id,
            )
            for frame_index in range(10)
            for camera_id in (0, 1)
        )
    )

    split = periodic_train_eval_split(
        manifest,
        every=4,
        offset=0,
        start_position=4,
    )

    assert {frame.frame_index for frame in split.eval_manifest} == {4, 8}
    assert 0 in {frame.frame_index for frame in split.train_manifest}
    assert len(split.eval_manifest) == 4
    assert len(split.train_manifest) == 16


def test_scene_61_style_split_keeps_boundary_and_holds_out_8_to_32(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    manifest = CanonicalDatasetManifest(
        frames=tuple(
            _frame(
                image,
                frame_index=frame_index,
                timestamp=1_000_000 + frame_index * 100 + camera_id,
                capture_timestamp=1_000_000 + frame_index * 100,
                camera_id=camera_id,
            )
            for frame_index in range(40)
            for camera_id in (0, 1)
        )
    )

    split = periodic_train_eval_split(
        manifest,
        every=8,
        offset=0,
        start_position=1,
    )

    assert {frame.frame_index for frame in split.eval_manifest} == {8, 16, 24, 32}
    training_indices = {frame.frame_index for frame in split.train_manifest}
    assert {0, 39} <= training_indices


def test_explicit_split_validates_indices(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    split = split_manifest_by_frame_indices(manifest, [20, 40])
    assert {frame.frame_index for frame in split.eval_manifest} == {20, 40}

    with pytest.raises(ValueError, match="must be unique"):
        split_manifest_by_frame_indices(manifest, [20, 20])
    with pytest.raises(ValueError, match="unknown"):
        split_manifest_by_frame_indices(manifest, [999])
    with pytest.raises(ValueError, match="evaluation split"):
        split_manifest_by_frame_indices(manifest, [])
    with pytest.raises(ValueError, match="training split"):
        split_manifest_by_frame_indices(manifest, [10, 20, 30, 40])
    with pytest.raises(TypeError, match="integers"):
        split_manifest_by_frame_indices(manifest, [True])


@pytest.mark.parametrize(
    ("every", "offset", "start_position"),
    (
        (0, 0, 0),
        (-1, 0, 0),
        (2, -1, 0),
        (2, 2, 0),
        (True, 0, 0),
        (2, True, 0),
        (2, 0, -1),
        (2, 0, True),
    ),
)
def test_periodic_split_validates_schedule(
    tmp_path: Path,
    every: int,
    offset: int,
    start_position: int,
) -> None:
    with pytest.raises(ValueError):
        periodic_train_eval_split(
            _manifest(tmp_path),
            every=every,
            offset=offset,
            start_position=start_position,
        )


def test_periodic_split_rejects_empty_sides(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    with pytest.raises(ValueError, match="training split"):
        periodic_train_eval_split(manifest, every=1)
    with pytest.raises(ValueError, match="evaluation split"):
        periodic_train_eval_split(manifest, every=8, offset=7)


def test_duplicate_camera_capture_row_is_rejected(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    manifest = CanonicalDatasetManifest(
        frames=(
            _frame(image, frame_index=0, timestamp=100, camera_id=0),
            _frame(image, frame_index=0, timestamp=100, camera_id=0),
            _frame(image, frame_index=1, timestamp=200, camera_id=0),
        )
    )

    with pytest.raises(ValueError, match="frame_index/camera_id"):
        periodic_train_eval_split(manifest, every=2)


def test_asynchronous_camera_timestamps_share_one_atomic_capture(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    manifest = CanonicalDatasetManifest(
        frames=(
            _frame(
                image,
                frame_index=0,
                timestamp=90,
                capture_timestamp=100,
                camera_id=0,
            ),
            _frame(
                image,
                frame_index=0,
                timestamp=110,
                capture_timestamp=100,
                camera_id=1,
            ),
            _frame(
                image,
                frame_index=1,
                timestamp=190,
                capture_timestamp=200,
                camera_id=0,
            ),
            _frame(
                image,
                frame_index=1,
                timestamp=210,
                capture_timestamp=200,
                camera_id=1,
            ),
        )
    )

    split = periodic_train_eval_split(manifest, every=2, offset=1)

    assert [frame.frame_index for frame in split.train_manifest] == [0, 0]
    assert [frame.frame_index for frame in split.eval_manifest] == [1, 1]
    assert split.training_timestamps.tolist() == [90, 110]


@pytest.mark.parametrize(
    "frames",
    (
        ((0, 100, 0), (0, 101, 1), (1, 200, 0)),
        ((0, 100, 0), (1, 100, 1), (2, 200, 0)),
    ),
)
def test_inconsistent_capture_identity_is_rejected(
    tmp_path: Path, frames: tuple[tuple[int, int, int], ...]
) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    manifest = CanonicalDatasetManifest(
        frames=tuple(
            _frame(
                image,
                frame_index=frame_index,
                timestamp=timestamp,
                camera_id=camera_id,
            )
            for frame_index, timestamp, camera_id in frames
        )
    )

    with pytest.raises(ValueError, match="multiple"):
        periodic_train_eval_split(manifest, every=2)


def test_eval_camera_without_training_row_is_rejected(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    manifest = CanonicalDatasetManifest(
        frames=(
            _frame(image, frame_index=0, timestamp=100, camera_id=0),
            _frame(image, frame_index=0, timestamp=100, camera_id=1),
            _frame(image, frame_index=1, timestamp=200, camera_id=1),
        )
    )

    with pytest.raises(ValueError, match="no training frames"):
        periodic_train_eval_split(manifest, every=2, offset=0)
