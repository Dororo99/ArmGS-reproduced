from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from armgs import materialize_actor_bbox_masks
from armgs.batching import canonical_frame_to_training_batch
from armgs.data.schema import (
    ActorTrack,
    ActorTrackSample,
    CanonicalDatasetManifest,
    CanonicalFrame,
)


_IMAGE_SIZE = (16, 20)


def _frame(tmp_path: Path, *, frame_index: int, camera_id: int = 0) -> CanonicalFrame:
    image_path = tmp_path / f"rgb_{camera_id}_{frame_index}.png"
    Image.new("RGB", (_IMAGE_SIZE[1], _IMAGE_SIZE[0]), color=(20, 40, 60)).save(
        image_path
    )
    return CanonicalFrame(
        timestamp=torch.tensor(1_000_000_000 + frame_index, dtype=torch.int64),
        camera_id=camera_id,
        camera_convention="opencv",
        camera_to_world=torch.eye(4, dtype=torch.float64),
        intrinsics=torch.tensor(
            ((20.0, 0.0, 10.0), (0.0, 20.0, 8.0), (0.0, 0.0, 1.0)),
            dtype=torch.float64,
        ),
        image_path=image_path,
        image_size=_IMAGE_SIZE,
        frame_index=frame_index,
    )


def _manifest(tmp_path: Path) -> CanonicalDatasetManifest:
    frames = (_frame(tmp_path, frame_index=0), _frame(tmp_path, frame_index=1))
    track = ActorTrack(
        actor_id=0,
        class_name="vehicle",
        dimensions_lwh=torch.tensor((2.0, 2.0, 2.0), dtype=torch.float64),
        samples=(
            ActorTrackSample(
                timestamp=frames[0].timestamp.clone(),
                translation=torch.tensor((0.0, 0.0, 10.0), dtype=torch.float64),
                quaternion_wxyz=torch.tensor(
                    (1.0, 0.0, 0.0, 0.0), dtype=torch.float64
                ),
                frame_index=0,
            ),
        ),
    )
    return CanonicalDatasetManifest(frames=frames, actor_tracks=(track,))


def test_materializer_attaches_visible_and_fully_empty_l_mode_masks(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    output_root = tmp_path / "actor_masks"
    # A stale, wrong-mode file must be atomically replaced rather than attached.
    stale = output_root / "camera_000" / "frame_00000000.png"
    stale.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2), color=(255, 255, 255)).save(stale)

    result = materialize_actor_bbox_masks(manifest, output_root, box_scale=1.0)

    assert result.actor_tracks is manifest.actor_tracks
    assert [frame.actor_mask_path for frame in result.frames] == [
        output_root.resolve() / "camera_000" / "frame_00000000.png",
        output_root.resolve() / "camera_000" / "frame_00000001.png",
    ]
    arrays: list[np.ndarray] = []
    for frame in result.frames:
        assert frame.actor_mask_path is not None
        with Image.open(frame.actor_mask_path) as image:
            image.load()
            assert image.mode == "L"
            assert image.size == (_IMAGE_SIZE[1], _IMAGE_SIZE[0])
            arrays.append(np.asarray(image).copy())
    assert set(np.unique(arrays[0]).tolist()).issubset({0, 255})
    assert np.any(arrays[0] == 255)
    assert not np.any(arrays[1])


def test_empty_mask_stays_attached_and_batches_as_false_bool_supervision(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    output_root = tmp_path / "actor_masks"
    result = materialize_actor_bbox_masks(manifest, output_root)
    empty_frame = result.frames[1]
    assert empty_frame.actor_mask_path is not None

    # Even a correctly sized L-mode file is stale if its pixels disagree. The
    # next materialization must restore the all-background target.
    Image.new(
        "L", (_IMAGE_SIZE[1], _IMAGE_SIZE[0]), color=255
    ).save(empty_frame.actor_mask_path)
    rematerialized = materialize_actor_bbox_masks(result, output_root)
    empty_frame = rematerialized.frames[1]

    batch = canonical_frame_to_training_batch(empty_frame, training_row=1)

    assert batch.actor_bbox_mask is not None
    assert batch.actor_bbox_mask.dtype == torch.bool
    assert batch.actor_bbox_mask.shape == (1, *_IMAGE_SIZE, 1)
    assert not batch.actor_bbox_mask.any()


def test_materializer_rejects_colliding_camera_frame_keys(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    duplicate = replace(
        manifest.frames[1],
        frame_index=manifest.frames[0].frame_index,
    )
    colliding = CanonicalDatasetManifest(
        frames=(manifest.frames[0], duplicate),
        actor_tracks=manifest.actor_tracks,
    )

    with pytest.raises(ValueError, match="duplicate camera_id/frame_index"):
        materialize_actor_bbox_masks(colliding, tmp_path / "actor_masks")
