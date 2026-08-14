from __future__ import annotations

import builtins
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from armgs.batching import (
    canonical_frame_to_training_batch,
    pillow_image_reader,
)
from armgs.data.schema import CanonicalFrame, LidarFrame, LidarProjection
from armgs.training import ArmGSTrainingBatch


class TensorReader:
    def __init__(self, values: dict[Path, Any]) -> None:
        self.values = values
        self.calls: list[tuple[Path, str]] = []

    def __call__(self, path: Path, mode: str) -> torch.Tensor:
        self.calls.append((path, mode))
        return self.values[path]


def _make_frame(
    tmp_path: Path,
    *,
    with_lidar: bool = True,
    with_masks: bool = True,
) -> CanonicalFrame:
    image_path = tmp_path / "image.png"
    sky_path = tmp_path / "sky.png"
    actor_path = tmp_path / "actor.png"
    scan_path = tmp_path / "scan.bin"
    for path in (image_path, sky_path, actor_path, scan_path):
        path.write_bytes(b"fixture")

    lidar = None
    projection = None
    if with_lidar:
        lidar = LidarFrame(
            points=torch.tensor(
                [
                    [0.0, 0.0, 4.0],
                    [0.0, 0.0, 2.0],
                    [0.0, 0.0, 3.0],
                    [0.0, 0.0, 6.0],
                ]
            ),
            reflectance=torch.tensor([0.1, 0.2, 0.3, 0.4]),
            sensor_to_world=torch.eye(4),
            source_path=scan_path,
        )
        projection = LidarProjection(
            camera_id=2,
            source_point_indices=torch.tensor([0, 1, 2, 3]),
            image_coordinates=torch.tensor(
                [[1.1, 0.1], [1.2, 0.2], [0.1, 1.1], [2.1, 1.1]]
            ),
            pixel_indices=torch.tensor([[1, 0], [1, 0], [0, 1], [2, 1]]),
            depths=torch.tensor([4.0, 2.0, 3.0, 6.0]),
            image_size=(2, 3),
        )

    return CanonicalFrame(
        timestamp=torch.tensor(123_456_789, dtype=torch.int64),
        camera_id=2,
        camera_convention="opengl",
        camera_to_world=torch.eye(4),
        intrinsics=torch.tensor(
            [[10.0, 0.0, 1.5], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]]
        ),
        image_path=image_path,
        image_size=(2, 3),
        frame_index=5,
        lidar=lidar,
        lidar_projection=projection,
        sky_mask_path=sky_path if with_masks else None,
        actor_mask_path=actor_path if with_masks else None,
    )


def test_full_frame_becomes_singleton_training_batch(tmp_path: Path) -> None:
    frame = _make_frame(tmp_path)
    rgb = torch.tensor(
        [
            [[0, 127, 255], [10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120], [130, 140, 150]],
        ],
        dtype=torch.uint8,
    )
    sky = torch.tensor([[0, 255, 0], [1, 0, 0]], dtype=torch.uint8)
    actor = torch.tensor(
        [[[0.0], [1.0], [0.0]], [[0.0], [0.0], [0.25]]]
    )
    reader = TensorReader(
        {
            frame.image_path: rgb,
            frame.sky_mask_path: sky,
            frame.actor_mask_path: actor,
        }
    )

    batch = canonical_frame_to_training_batch(
        frame, training_row=7, image_reader=reader
    )

    assert isinstance(batch, ArmGSTrainingBatch)
    assert batch.target_rgb.shape == (1, 2, 3, 3)
    assert batch.target_rgb.dtype == torch.float32
    torch.testing.assert_close(
        batch.target_rgb[0, 0, 0],
        torch.tensor([0.0, 127.0 / 255.0, 1.0]),
    )
    assert batch.target_sky_mask is not None
    assert batch.target_sky_mask.shape == (1, 2, 3, 1)
    assert batch.target_sky_mask.dtype == torch.bool
    assert batch.sky_valid_mask is not None
    assert batch.sky_valid_mask.shape == (1, 1, 1, 1)
    assert batch.sky_valid_mask.dtype == torch.bool
    assert batch.sky_valid_mask.item() is True
    torch.testing.assert_close(
        batch.target_sky_mask[0, ..., 0],
        torch.tensor([[False, True, False], [True, False, False]]),
    )
    assert batch.actor_bbox_mask is not None
    torch.testing.assert_close(
        batch.actor_bbox_mask[0, ..., 0],
        torch.tensor([[False, True, False], [False, False, True]]),
    )
    assert reader.calls == [
        (frame.image_path, "rgb"),
        (frame.sky_mask_path, "mask"),
        (frame.actor_mask_path, "mask"),
    ]

    assert batch.view.image_size == (2, 3)
    assert batch.view.camera_convention == "opengl"
    assert batch.view.timestamp.shape == ()
    assert batch.view.timestamp.dtype == torch.int64
    assert batch.view.timestamp.item() == 123_456_789
    assert batch.view.camera_id.shape == ()
    assert batch.view.camera_id.dtype == torch.long
    assert batch.view.camera_id.item() == 2
    assert batch.view.training_row is not None
    assert batch.view.training_row.shape == ()
    assert batch.view.training_row.dtype == torch.long
    assert batch.view.training_row.item() == 7
    torch.testing.assert_close(batch.view.camera_to_world, frame.camera_to_world)
    torch.testing.assert_close(batch.view.intrinsics, frame.intrinsics)

    assert batch.lidar_depth is not None
    assert batch.depth_valid_mask is not None
    assert batch.lidar_depth.shape == (1, 2, 3, 1)
    assert batch.depth_valid_mask.dtype == torch.bool
    torch.testing.assert_close(
        batch.lidar_depth[0, ..., 0],
        torch.tensor([[0.0, 2.0, 0.0], [3.0, 0.0, 6.0]]),
    )
    torch.testing.assert_close(
        batch.depth_valid_mask[0, ..., 0],
        torch.tensor([[False, True, False], [True, False, True]]),
    )


def test_held_out_frame_preserves_none_training_row(tmp_path: Path) -> None:
    frame = _make_frame(tmp_path, with_lidar=False, with_masks=False)
    reader = TensorReader(
        {frame.image_path: torch.zeros((2, 3, 3), dtype=torch.uint8)}
    )

    batch = canonical_frame_to_training_batch(
        frame,
        training_row=None,
        image_reader=reader,
    )

    assert batch.view.training_row is None



def test_duplicate_depth_resolution_is_independent_of_input_order(
    tmp_path: Path,
) -> None:
    frame = _make_frame(tmp_path)
    assert frame.lidar_projection is not None
    projection = frame.lidar_projection
    order = torch.tensor([3, 1, 0, 2])
    reordered = replace(
        projection,
        source_point_indices=projection.source_point_indices[order],
        image_coordinates=projection.image_coordinates[order],
        pixel_indices=projection.pixel_indices[order],
        depths=projection.depths[order],
    )
    reordered_frame = replace(frame, lidar_projection=reordered)
    reader = TensorReader(
        {
            frame.image_path: torch.zeros(2, 3, 3),
            frame.sky_mask_path: torch.zeros(2, 3),
            frame.actor_mask_path: torch.zeros(2, 3),
        }
    )

    original = canonical_frame_to_training_batch(
        frame, 0, image_reader=reader
    )
    permuted = canonical_frame_to_training_batch(
        reordered_frame, 0, image_reader=reader
    )

    torch.testing.assert_close(original.lidar_depth, permuted.lidar_depth)
    torch.testing.assert_close(
        original.depth_valid_mask, permuted.depth_valid_mask
    )


def test_rejected_sky_frame_still_reads_raw_target_with_false_validity(
    tmp_path: Path,
) -> None:
    frame = replace(_make_frame(tmp_path), sky_supervision_valid=False)
    reader = TensorReader(
        {
            frame.image_path: torch.zeros(2, 3, 3),
            frame.sky_mask_path: torch.ones(2, 3),
            frame.actor_mask_path: torch.zeros(2, 3),
        }
    )

    batch = canonical_frame_to_training_batch(frame, 0, image_reader=reader)

    assert batch.target_sky_mask is not None
    assert batch.target_sky_mask.all()
    assert batch.sky_valid_mask is not None
    assert batch.sky_valid_mask.item() is False
    assert (frame.sky_mask_path, "mask") in reader.calls


def test_missing_optional_supervision_stays_none(tmp_path: Path) -> None:
    frame = _make_frame(tmp_path, with_lidar=False, with_masks=False)
    target = torch.full((2, 3, 3), 0.25, dtype=torch.float64)
    reader = TensorReader({frame.image_path: target})

    batch = canonical_frame_to_training_batch(
        frame,
        torch.tensor(4, dtype=torch.int32),
        image_reader=reader,
        device="cpu",
    )

    assert batch.target_rgb.dtype == torch.float32
    torch.testing.assert_close(batch.target_rgb, target.float().unsqueeze(0))
    assert batch.lidar_depth is None
    assert batch.depth_valid_mask is None
    assert batch.target_sky_mask is None
    assert batch.sky_valid_mask is None
    assert batch.actor_bbox_mask is None
    assert reader.calls == [(frame.image_path, "rgb")]


def test_empty_projection_produces_empty_sparse_depth_maps(
    tmp_path: Path,
) -> None:
    frame = _make_frame(tmp_path, with_masks=False)
    assert frame.lidar_projection is not None
    empty = replace(
        frame.lidar_projection,
        source_point_indices=torch.empty(0, dtype=torch.long),
        image_coordinates=torch.empty(0, 2),
        pixel_indices=torch.empty(0, 2, dtype=torch.long),
        depths=torch.empty(0),
    )
    frame = replace(frame, lidar_projection=empty)
    reader = TensorReader({frame.image_path: torch.zeros(2, 3, 3)})

    batch = canonical_frame_to_training_batch(frame, 0, image_reader=reader)

    assert batch.lidar_depth is not None
    assert batch.depth_valid_mask is not None
    assert not batch.depth_valid_mask.any()
    assert torch.count_nonzero(batch.lidar_depth) == 0


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (torch.zeros(2, 3), "RGB image size mismatch"),
        (torch.zeros(3, 2, 3), "RGB image size mismatch"),
        (torch.zeros(2, 3, 4), "RGB image size mismatch"),
        (torch.full((2, 3, 3), 1.1), r"values must lie in \[0,1\]"),
        (
            torch.full((2, 3, 3), torch.nan),
            "contain only finite values",
        ),
        (torch.zeros(2, 3, 3, dtype=torch.int16), "uint8 or floating point"),
    ],
)
def test_invalid_rgb_is_rejected(
    tmp_path: Path,
    value: torch.Tensor,
    message: str,
) -> None:
    frame = _make_frame(tmp_path, with_lidar=False, with_masks=False)
    reader = TensorReader({frame.image_path: value})

    with pytest.raises(ValueError, match=message):
        canonical_frame_to_training_batch(frame, 0, image_reader=reader)


@pytest.mark.parametrize(
    "mask",
    [
        torch.zeros(3, 3),
        torch.zeros(2, 3, 3),
        torch.zeros(2, 3, 1, 1),
    ],
)
def test_invalid_mask_dimensions_are_rejected(
    tmp_path: Path,
    mask: torch.Tensor,
) -> None:
    frame = _make_frame(tmp_path, with_lidar=False)
    reader = TensorReader(
        {
            frame.image_path: torch.zeros(2, 3, 3),
            frame.sky_mask_path: mask,
            frame.actor_mask_path: torch.zeros(2, 3),
        }
    )

    with pytest.raises(ValueError, match="sky mask size mismatch"):
        canonical_frame_to_training_batch(frame, 0, image_reader=reader)


@pytest.mark.parametrize(
    "training_row",
    [
        True,
        -1,
        1.0,
        torch.tensor([1, 2]),
    ],
)
def test_training_row_must_be_non_negative_integer_scalar(
    tmp_path: Path,
    training_row: Any,
) -> None:
    frame = _make_frame(tmp_path, with_lidar=False, with_masks=False)
    reader = TensorReader({frame.image_path: torch.zeros(2, 3, 3)})

    with pytest.raises(ValueError, match="training_row"):
        canonical_frame_to_training_batch(
            frame, training_row, image_reader=reader
        )


def test_reader_must_return_tensor(tmp_path: Path) -> None:
    frame = _make_frame(tmp_path, with_lidar=False, with_masks=False)
    reader = TensorReader({frame.image_path: [[[0, 0, 0]]]})

    with pytest.raises(TypeError, match="must return a Tensor"):
        canonical_frame_to_training_batch(frame, 0, image_reader=reader)


def test_missing_pillow_reports_optional_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"fixture")
    real_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "PIL":
            raise ModuleNotFoundError("Pillow intentionally hidden")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match="optional Pillow"):
        pillow_image_reader(path, "rgb")
