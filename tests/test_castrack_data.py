from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from armgs.data.castrack import (
    CASTRACK_ACTOR_SOURCE,
    extract_castrack_scene_json,
    load_castrack_actor_tracks,
)
from armgs.data.schema import CanonicalFrame


_SEQUENCE = "synthetic_context"


def _transform(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> torch.Tensor:
    value = torch.eye(4, dtype=torch.float64)
    value[:3, 3] = torch.tensor((tx, ty, tz), dtype=torch.float64)
    return value


def _front_frame(
    tmp_path: Path,
    *,
    frame_index: int,
    capture_timestamp_micros: int,
    world_x: float = -50.0,
    observation_offset_ns: int = 0,
) -> CanonicalFrame:
    image_path = tmp_path / f"front_{frame_index}.png"
    image_path.write_bytes(b"canonical frame only validates file existence")
    return CanonicalFrame(
        timestamp=torch.tensor(
            capture_timestamp_micros * 1_000 + observation_offset_ns,
            dtype=torch.int64,
        ),
        camera_id=0,
        camera_convention="opencv",
        camera_to_world=_transform(world_x),
        intrinsics=torch.tensor(
            ((10.0, 0.0, 6.0), (0.0, 10.0, 6.0), (0.0, 0.0, 1.0)),
            dtype=torch.float64,
        ),
        image_path=image_path,
        image_size=(12, 12),
        frame_index=frame_index,
        capture_timestamp=torch.tensor(
            capture_timestamp_micros * 1_000, dtype=torch.int64
        ),
    )


def _box(
    x: float,
    *,
    y: float = 0.0,
    z: float = 10.0,
    dimensions: tuple[float, float, float] = (4.0, 2.0, 1.5),
    heading: float = 0.0,
) -> list[float]:
    return [x, y, z, *dimensions, heading]


def _record(
    entries: list[tuple[int, str, list[float]]], *, frame_id: int
) -> dict[str, object]:
    return {
        "obj_ids": [entry[0] for entry in entries],
        "name": [entry[1] for entry in entries],
        "boxes_lidar": [entry[2] for entry in entries],
        "frame_id": str(frame_id),
        "seq_id": f"segment-{_SEQUENCE}_with_camera_labels",
    }


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _inputs(
    tmp_path: Path,
    timestamps: tuple[int, ...],
    *,
    first_source_frame: int = 10,
    observation_offsets_ns: tuple[int, ...] | None = None,
) -> dict[str, object]:
    if observation_offsets_ns is None:
        observation_offsets_ns = (0,) * len(timestamps)
    relative_indices = {timestamp: index for index, timestamp in enumerate(timestamps)}
    transforms = {timestamp: _transform(-50.0) for timestamp in timestamps}
    frames = tuple(
        _front_frame(
            tmp_path,
            frame_index=index,
            capture_timestamp_micros=timestamp,
            observation_offset_ns=observation_offsets_ns[index],
        )
        for index, timestamp in enumerate(timestamps)
    )
    return {
        "source_frame_indices": tuple(
            range(first_source_frame, first_source_frame + len(timestamps))
        ),
        "selected_timestamps_micros": timestamps,
        "relative_indices": relative_indices,
        "vehicle_transforms": transforms,
        "front_frames": frames,
    }


def test_direct_scene_load_filters_visibility_static_and_classes(
    tmp_path: Path,
) -> None:
    timestamps = (1_000_000, 1_100_000, 1_200_000)
    payload = {
        "10": _record(
            [
                (99, "Vehicle", _box(0.0)),
                (7, "Vehicle", _box(0.0, dimensions=(4.0, 2.0, 1.5))),
                (8, "Sign", _box(0.0)),
                (9, "Pedestrian", _box(100.0)),
            ],
            frame_id=10,
        ),
        "11": _record(
            [
                (99, "Vehicle", _box(0.1)),
                (7, "Vehicle", _box(1.0, dimensions=(4.5, 2.2, 1.6))),
            ],
            frame_id=11,
        ),
        "12": _record(
            [
                (99, "Vehicle", _box(0.2)),
                (7, "Vehicle", _box(3.0, dimensions=(4.2, 2.5, 1.8))),
            ],
            frame_id=12,
        ),
    }
    path = _write_json(tmp_path / "scene.json", payload)
    inputs = _inputs(
        tmp_path,
        timestamps,
        observation_offsets_ns=(-20_000_000, 0, 30_000_000),
    )

    tracks = load_castrack_actor_tracks(
        path,
        sequence=_SEQUENCE,
        **inputs,
    )

    assert CASTRACK_ACTOR_SOURCE == "castrack"
    assert len(tracks) == 1
    track = tracks[0]
    assert track.actor_id == 0
    assert track.class_name == "vehicle"
    torch.testing.assert_close(
        track.dimensions_lwh,
        torch.tensor((4.5, 2.5, 1.8), dtype=torch.float64),
    )
    assert [sample.frame_index for sample in track.samples] == [0, 1, 2]
    assert [int(sample.timestamp.item()) for sample in track.samples] == [
        1_000_000_000,
        1_100_000_000,
        1_200_000_000,
    ]
    torch.testing.assert_close(
        torch.stack([sample.translation for sample in track.samples]),
        torch.tensor(
            ((-50.0, 0.0, 10.0), (-49.0, 0.0, 10.0), (-47.0, 0.0, 10.0)),
            dtype=torch.float64,
        ),
    )
    assert int(track.lifecycle_start_timestamp.item()) == 980_000_000
    assert int(track.lifecycle_end_timestamp.item()) == 1_230_000_000


def test_classes_endpoint_motion_and_heading_are_canonicalized(tmp_path: Path) -> None:
    timestamps = (2_000_000, 2_100_000)
    payload = {
        "10": _record(
            [
                (30, "Cyclist", _box(0.0, heading=0.0)),
                (10, "Vehicle", _box(0.0, heading=0.0)),
                (20, "Pedestrian", _box(0.0, heading=0.0)),
            ],
            frame_id=10,
        ),
        "11": _record(
            [
                (30, "Cyclist", _box(3.0, heading=math.pi / 2.0)),
                (10, "Vehicle", _box(3.0, heading=math.pi / 2.0)),
                (20, "Pedestrian", _box(3.0, heading=math.pi / 2.0)),
            ],
            frame_id=11,
        ),
    }
    path = _write_json(tmp_path / "scene.json", payload)

    tracks = load_castrack_actor_tracks(
        path,
        sequence=f"segment-{_SEQUENCE}_with_camera_labels",
        static_std_threshold=100.0,
        static_displacement_threshold=2.0,
        **_inputs(tmp_path, timestamps),
    )

    assert [track.actor_id for track in tracks] == [0, 1, 2]
    assert [track.class_name for track in tracks] == [
        "cyclist",
        "vehicle",
        "pedestrian",
    ]
    expected = torch.tensor(
        (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)), dtype=torch.float64
    )
    for track in tracks:
        actual = track.samples[-1].quaternion_wxyz
        torch.testing.assert_close(actual.abs(), expected.abs(), atol=1.0e-7, rtol=1.0e-7)


def test_visibility_requires_an_in_frame_positive_depth_corner(tmp_path: Path) -> None:
    timestamps = (3_000_000,)
    payload = {
        "10": _record(
            [
                # The image center lies inside this enormous projected box, but
                # none of its actual corners lies inside the image.
                (1, "Vehicle", _box(0.0, dimensions=(100.0, 100.0, 1.0))),
                (2, "Pedestrian", _box(0.0, z=-10.0)),
                (3, "Cyclist", _box(0.0)),
            ],
            frame_id=10,
        )
    }
    path = _write_json(tmp_path / "scene.json", payload)

    tracks = load_castrack_actor_tracks(
        path,
        sequence=_SEQUENCE,
        filter_static_actors=False,
        **_inputs(tmp_path, timestamps),
    )

    assert len(tracks) == 1
    assert tracks[0].class_name == "cyclist"


def test_extract_keyed_scene_and_reload_without_the_full_payload(tmp_path: Path) -> None:
    timestamps = (4_000_000,)
    target_key = f"segment-{_SEQUENCE}_with_camera_labels"
    other_key = "segment-another_context_with_camera_labels"
    target_scene = {
        "10": _record([(17, "Vehicle", _box(0.0))], frame_id=10)
    }
    source = _write_json(
        tmp_path / "full.json",
        {
            other_key: {
                "0": {
                    "obj_ids": [],
                    "name": [],
                    "boxes_lidar": [],
                }
            },
            target_key: target_scene,
        },
    )
    destination = tmp_path / "extracted" / "scene.json"

    result = extract_castrack_scene_json(
        source, destination, sequence=_SEQUENCE
    )

    assert result == destination.resolve()
    assert set(json.loads(destination.read_text(encoding="utf-8"))) == {target_key}
    tracks = load_castrack_actor_tracks(
        destination,
        sequence=_SEQUENCE,
        filter_static_actors=False,
        **_inputs(tmp_path, timestamps),
    )
    assert len(tracks) == 1
    with pytest.raises(FileExistsError, match="already exists"):
        extract_castrack_scene_json(source, destination, sequence=_SEQUENCE)
    with pytest.raises(KeyError, match="another_context"):
        load_castrack_actor_tracks(
            destination,
            sequence="another_context",
            filter_static_actors=False,
            **_inputs(tmp_path, timestamps),
        )


def test_alignment_contract_rejects_wrong_capture_timestamp(tmp_path: Path) -> None:
    timestamp = 5_000_000
    path = _write_json(
        tmp_path / "scene.json",
        {"10": _record([(1, "Vehicle", _box(0.0))], frame_id=10)},
    )
    inputs = _inputs(tmp_path, (timestamp,))
    wrong_frame = _front_frame(
        tmp_path,
        frame_index=0,
        capture_timestamp_micros=timestamp + 1,
    )
    inputs["front_frames"] = (wrong_frame,)

    with pytest.raises(ValueError, match="capture timestamp does not match"):
        load_castrack_actor_tracks(
            path,
            sequence=_SEQUENCE,
            filter_static_actors=False,
            **inputs,
        )
