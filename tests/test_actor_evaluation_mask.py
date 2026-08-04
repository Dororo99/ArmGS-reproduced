from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from armgs import project_actor_boxes_to_mask
from armgs.data.schema import ActorTrack, ActorTrackSample, CanonicalFrame


def _frame(
    tmp_path: Path,
    *,
    frame_index: int = 3,
    convention: str = "opencv",
    image_size: tuple[int, int] = (100, 120),
) -> CanonicalFrame:
    image_path = tmp_path / f"frame-{frame_index}-{convention}.png"
    image_path.write_bytes(b"image")
    height, width = image_size
    intrinsics = torch.tensor(
        [
            [100.0, 0.0, width / 2],
            [0.0, 100.0, height / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    return CanonicalFrame(
        timestamp=torch.tensor(frame_index * 1_000_000, dtype=torch.int64),
        camera_id=0,
        camera_convention=convention,  # type: ignore[arg-type]
        camera_to_world=torch.eye(4, dtype=torch.float64),
        intrinsics=intrinsics,
        image_path=image_path,
        image_size=image_size,
        frame_index=frame_index,
    )


def _track(
    *,
    actor_id: int,
    frame_index: int,
    translation: tuple[float, float, float],
    dimensions: tuple[float, float, float] = (2.0, 2.0, 2.0),
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> ActorTrack:
    return ActorTrack(
        actor_id=actor_id,
        class_name="vehicle.car",
        dimensions_lwh=torch.tensor(dimensions, dtype=torch.float64),
        samples=(
            ActorTrackSample(
                timestamp=torch.tensor(frame_index * 1_000_000, dtype=torch.int64),
                translation=torch.tensor(translation, dtype=torch.float64),
                quaternion_wxyz=torch.tensor(quaternion, dtype=torch.float64),
                frame_index=frame_index,
            ),
        ),
    )


def test_centered_actor_projects_to_expected_silhouette(tmp_path: Path) -> None:
    frame = _frame(tmp_path)
    actor = _track(actor_id=0, frame_index=3, translation=(0.0, 0.0, 10.0))

    mask = project_actor_boxes_to_mask(frame, (actor,))

    expected = torch.zeros((100, 120), dtype=torch.bool)
    expected[39:62, 49:72] = True
    assert mask.dtype == torch.bool
    assert mask.shape == (100, 120)
    torch.testing.assert_close(mask, expected)


def test_oblique_actor_excludes_enclosing_rectangle_background(
    tmp_path: Path,
) -> None:
    frame = _frame(tmp_path)
    angle = math.pi / 4.0
    actor = _track(
        actor_id=0,
        frame_index=3,
        translation=(0.0, 0.0, 10.0),
        dimensions=(4.0, 2.0, 2.0),
        quaternion=(math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0),
    )

    mask = project_actor_boxes_to_mask(frame, (actor,))
    coordinates = torch.nonzero(mask)
    y_min, x_min = coordinates.amin(dim=0).tolist()
    y_max, x_max = coordinates.amax(dim=0).tolist()

    assert mask.any()
    assert not mask[y_min, x_min]
    assert not mask[y_min, x_max]
    assert not mask[y_max, x_min]
    assert not mask[y_max, x_max]
    enclosing_rectangle_pixels = (y_max - y_min + 1) * (x_max - x_min + 1)
    assert int(mask.sum()) < enclosing_rectangle_pixels


def test_actor_entirely_behind_camera_is_empty(tmp_path: Path) -> None:
    frame = _frame(tmp_path)
    actor = _track(actor_id=0, frame_index=3, translation=(0.0, 0.0, -5.0))

    assert not project_actor_boxes_to_mask(frame, (actor,)).any()


def test_near_plane_crossing_is_clipped_and_finite(tmp_path: Path) -> None:
    frame = _frame(tmp_path)
    actor = _track(
        actor_id=0,
        frame_index=3,
        translation=(0.0, 0.0, 0.5),
        dimensions=(1.0, 1.0, 2.0),
    )

    mask = project_actor_boxes_to_mask(frame, (actor,), near_plane=0.1)

    assert mask.all()


def test_partially_offscreen_actor_is_clamped_to_image(tmp_path: Path) -> None:
    frame = _frame(tmp_path, image_size=(100, 100))
    actor = _track(actor_id=0, frame_index=3, translation=(3.0, 0.0, 5.0))

    mask = project_actor_boxes_to_mask(frame, (actor,))

    assert mask.any()
    assert mask[:, :83].sum() == 0
    assert mask[:, 83:].any()
    assert mask.shape == (100, 100)


def test_union_uses_only_samples_matching_exact_frame_index(tmp_path: Path) -> None:
    frame = _frame(tmp_path, frame_index=3)
    left = _track(actor_id=0, frame_index=3, translation=(-2.5, 0.0, 10.0))
    right = _track(actor_id=1, frame_index=3, translation=(2.5, 0.0, 10.0))
    unmatched = _track(actor_id=2, frame_index=2, translation=(0.0, 0.0, 5.0))

    mask = project_actor_boxes_to_mask(frame, (left, unmatched, right))

    assert mask[50, 25]
    assert mask[50, 95]
    assert not mask[50, 60]
    torch.testing.assert_close(
        mask,
        project_actor_boxes_to_mask(frame, (right, left)),
    )


def test_opengl_axis_conversion_and_wxyz_rotation(tmp_path: Path) -> None:
    opencv = _frame(tmp_path, convention="opencv")
    opengl = _frame(tmp_path, convention="opengl")
    identity_actor = _track(
        actor_id=0,
        frame_index=3,
        translation=(0.0, 0.0, 10.0),
        dimensions=(2.0, 4.0, 2.0),
    )
    half_sqrt = math.sqrt(0.5)
    rotated_actor = _track(
        actor_id=1,
        frame_index=3,
        translation=(0.0, 0.0, 10.0),
        dimensions=(4.0, 2.0, 2.0),
        quaternion=(half_sqrt, 0.0, 0.0, half_sqrt),
    )
    gl_actor = _track(
        actor_id=2,
        frame_index=3,
        translation=(0.0, 0.0, -10.0),
        dimensions=(2.0, 4.0, 2.0),
    )

    torch.testing.assert_close(
        project_actor_boxes_to_mask(opencv, (identity_actor,)),
        project_actor_boxes_to_mask(opencv, (rotated_actor,)),
    )
    torch.testing.assert_close(
        project_actor_boxes_to_mask(opencv, (identity_actor,)),
        project_actor_boxes_to_mask(opengl, (gl_actor,)),
    )


def test_empty_tracks_and_fully_offscreen_box_are_empty(tmp_path: Path) -> None:
    frame = _frame(tmp_path)
    offscreen = _track(
        actor_id=0,
        frame_index=3,
        translation=(100.0, 0.0, 5.0),
    )

    assert not project_actor_boxes_to_mask(frame, ()).any()
    assert not project_actor_boxes_to_mask(frame, (offscreen,)).any()


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"box_scale": 0.0}, ValueError, "box_scale"),
        ({"box_scale": float("nan")}, ValueError, "box_scale"),
        ({"near_plane": 0.0}, ValueError, "near_plane"),
        ({"near_plane": float("inf")}, ValueError, "near_plane"),
    ],
)
def test_invalid_projection_arguments_are_rejected(
    tmp_path: Path,
    kwargs: dict[str, float],
    error: type[Exception],
    message: str,
) -> None:
    frame = _frame(tmp_path)
    with pytest.raises(error, match=message):
        project_actor_boxes_to_mask(frame, (), **kwargs)

    with pytest.raises(TypeError, match="CanonicalFrame"):
        project_actor_boxes_to_mask(object(), ())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sequence"):
        project_actor_boxes_to_mask(frame, None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="only ActorTrack"):
        project_actor_boxes_to_mask(frame, (object(),))  # type: ignore[arg-type]
