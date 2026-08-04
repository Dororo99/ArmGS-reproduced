from __future__ import annotations

import json
from pathlib import Path
import struct

import pytest
import torch

import armgs.data.nuscenes as nuscenes_data
from armgs.data import (
    CanonicalFrameDataset,
    load_nuscenes_manifest,
    normalize_nuscenes_scene_name,
    parse_nuscenes_sky_mask_reject_list,
    read_nuscenes_lidar_bin,
)


_CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT")


def _write_json(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def _write_lidar(path: Path) -> None:
    rows = (
        (0.0, 0.0, 2.0, 127.5, 3.0),
        (2.0, 0.0, 1.0, 255.0, 4.0),
        (0.0, 0.0, -1.0, 0.0, 5.0),
    )
    path.write_bytes(b"".join(struct.pack("<fffff", *row) for row in rows))


def _make_nuscenes(root: Path) -> Path:
    metadata = root / "v1.0-trainval"
    metadata.mkdir(parents=True)
    (root / "samples" / "CAM_FRONT").mkdir(parents=True)
    (root / "samples" / "CAM_FRONT_LEFT").mkdir(parents=True)
    (root / "samples" / "LIDAR_TOP").mkdir(parents=True)

    samples = [
        {
            "token": "sample-0",
            "timestamp": 1_000_000,
            "prev": "",
            "next": "sample-1",
            "scene_token": "scene-token",
        },
        {
            "token": "sample-1",
            "timestamp": 1_500_000,
            "prev": "sample-0",
            "next": "",
            "scene_token": "scene-token",
        },
    ]
    _write_json(
        metadata / "scene.json",
        [
            {
                "token": "scene-token",
                "name": "scene-0061",
                "nbr_samples": 2,
                "first_sample_token": "sample-0",
                "last_sample_token": "sample-1",
            }
        ],
    )
    _write_json(metadata / "sample.json", samples)

    sensor_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    for channel in (*_CAMERAS, "LIDAR_TOP"):
        slug = channel.lower()
        sensor_rows.append(
            {
                "token": f"sensor-{slug}",
                "channel": channel,
                "modality": "camera" if channel.startswith("CAM") else "lidar",
            }
        )
        calibration_rows.append(
            {
                "token": f"calib-{slug}",
                "sensor_token": f"sensor-{slug}",
                "translation": [0.0, 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "camera_intrinsic": (
                    [[10.0, 0.0, 2.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]]
                    if channel.startswith("CAM")
                    else []
                ),
            }
        )
    _write_json(metadata / "sensor.json", sensor_rows)
    _write_json(metadata / "calibrated_sensor.json", calibration_rows)

    sample_data: list[dict[str, object]] = []
    ego_poses: list[dict[str, object]] = []
    for frame_index, sample in enumerate(samples):
        base_x = 10.0 * (frame_index + 1)
        for channel_index, channel in enumerate((*_CAMERAS, "LIDAR_TOP")):
            slug = channel.lower()
            extension = "jpg" if channel.startswith("CAM") else "pcd.bin"
            relative = Path("samples") / channel / f"{frame_index}.{extension}"
            source = root / relative
            if channel.startswith("CAM"):
                source.write_bytes(b"synthetic jpeg placeholder")
            else:
                _write_lidar(source)
            pose_token = f"pose-{frame_index}-{slug}"
            ego_poses.append(
                {
                    "token": pose_token,
                    # Each camera uses its own asynchronous ego pose.
                    "translation": [base_x + 0.1 * channel_index, 0.0, 0.0],
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "timestamp": int(sample["timestamp"]) + channel_index,
                }
            )
            sample_data.append(
                {
                    "token": f"data-{frame_index}-{slug}",
                    "sample_token": sample["token"],
                    "ego_pose_token": pose_token,
                    "calibrated_sensor_token": f"calib-{slug}",
                    "timestamp": int(sample["timestamp"]) + channel_index,
                    "is_key_frame": True,
                    "height": 4 if channel.startswith("CAM") else 0,
                    "width": 6 if channel.startswith("CAM") else 0,
                    "filename": relative.as_posix(),
                }
            )
    # A sweep assigned to sample-0 must not be mistaken for key sample data.
    sample_data.append(
        {
            **sample_data[0],
            "token": "front-sweep",
            "is_key_frame": False,
            "filename": sample_data[0]["filename"],
        }
    )
    _write_json(metadata / "sample_data.json", sample_data)
    _write_json(metadata / "ego_pose.json", ego_poses)

    annotations: list[dict[str, object]] = []
    for frame_index, sample in enumerate(samples):
        annotations.extend(
            [
                {
                    "token": f"moving-{frame_index}",
                    "sample_token": sample["token"],
                    "instance_token": "instance-moving",
                    "translation": [float(frame_index), 2.0, 0.0],
                    "size": [2.0, 4.0, 1.5],
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                },
                {
                    "token": f"stationary-{frame_index}",
                    "sample_token": sample["token"],
                    "instance_token": "instance-stationary",
                    "translation": [5.0 + 0.1 * frame_index, 0.0, 0.0],
                    "size": [2.0, 4.0, 1.5],
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                },
                {
                    "token": f"animal-{frame_index}",
                    "sample_token": sample["token"],
                    "instance_token": "instance-animal",
                    "translation": [10.0 * frame_index, 0.0, 0.0],
                    "size": [1.0, 1.0, 1.0],
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                },
            ]
        )
    _write_json(metadata / "sample_annotation.json", annotations)
    _write_json(
        metadata / "instance.json",
        [
            {"token": "instance-moving", "category_token": "category-car"},
            {"token": "instance-stationary", "category_token": "category-car"},
            {"token": "instance-animal", "category_token": "category-animal"},
        ],
    )
    _write_json(
        metadata / "category.json",
        [
            {"token": "category-car", "name": "vehicle.car"},
            {"token": "category-animal", "name": "animal"},
        ],
    )
    return root


@pytest.mark.parametrize("selector", [61, "0061", "scene-0061"])
def test_scene_selector_normalization(selector: int | str) -> None:
    assert normalize_nuscenes_scene_name(selector) == "scene-0061"


def test_sky_mask_reject_list_parser_allows_comments_and_normalizes_hex(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reject.txt"
    path.write_text(
        "# reviewed masks\n\n" + "A" * 32 + "  # facade\n" + "b" * 32 + "\n",
        encoding="utf-8",
    )

    assert parse_nuscenes_sky_mask_reject_list(path) == {
        "a" * 32,
        "b" * 32,
    }


@pytest.mark.parametrize(
    "contents",
    [
        "not-a-token\n",
        "g" * 32 + "\n",
        "a" * 31 + "\n",
        "a" * 32 + "\n" + "A" * 32 + "\n",
    ],
)
def test_sky_mask_reject_list_parser_rejects_malformed_or_duplicate_tokens(
    tmp_path: Path,
    contents: str,
) -> None:
    path = tmp_path / "reject.txt"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="malformed|duplicate"):
        parse_nuscenes_sky_mask_reject_list(path)


def test_nuscenes_lidar_reader_uses_five_floats_and_normalizes_intensity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scan.pcd.bin"
    _write_lidar(path)

    points, reflectance = read_nuscenes_lidar_bin(path)

    assert points.shape == (3, 3)
    assert torch.allclose(reflectance, torch.tensor([0.5, 1.0, 0.0]))


def test_load_manifest_aligns_cameras_lidar_timestamps_and_dynamic_actors(
    tmp_path: Path,
) -> None:
    root = _make_nuscenes(tmp_path / "nuscenes")

    manifest = load_nuscenes_manifest(
        root,
        scene="61",
        camera_channels=_CAMERAS,
    )
    dataset = CanonicalFrameDataset(manifest)

    assert len(dataset) == 4
    front, left, next_front, next_left = manifest.frames
    assert front.timestamp.item() == 1_000_000_000
    assert left.timestamp.item() == 1_000_001_000
    assert next_front.timestamp.item() == 1_500_000_000
    assert next_left.timestamp.item() == 1_500_001_000
    assert front.observation_timestamp is front.timestamp
    assert front.capture_timestamp is not None
    assert left.capture_timestamp is not None
    assert front.capture_timestamp.item() == left.capture_timestamp.item() == 1_000_000_000
    assert next_front.capture_timestamp is not None
    assert next_left.capture_timestamp is not None
    assert (
        next_front.capture_timestamp.item()
        == next_left.capture_timestamp.item()
        == 1_500_000_000
    )
    assert (front.camera_id, left.camera_id) == (0, 1)
    assert front.camera_convention == left.camera_convention == "opencv"
    assert front.frame_index == left.frame_index == 0
    # Per-camera ego poses and asynchronous sensor timestamps are both preserved.
    assert front.camera_to_world[0, 3].item() == pytest.approx(10.0)
    assert left.camera_to_world[0, 3].item() == pytest.approx(10.1)
    assert front.lidar is left.lidar
    assert front.lidar is not None and front.lidar_projection is not None
    assert front.lidar.points.shape == (1, 3)
    assert torch.allclose(front.lidar.world_points[0], torch.tensor([10.2, 0.0, 2.0]))
    assert front.lidar_projection.source_point_indices.tolist() == [0]
    assert front.lidar_projection.depths.tolist() == [2.0]

    assert len(manifest.actor_tracks) == 1
    actor = manifest.actor_tracks[0]
    assert actor.class_name == "vehicle.car"
    assert torch.allclose(
        actor.dimensions_lwh,
        torch.tensor([4.0, 2.0, 1.5], dtype=torch.float64),
    )
    assert [sample.frame_index for sample in actor.samples] == [0, 1]
    assert [sample.timestamp.item() for sample in actor.samples] == [
        1_000_000_000,
        1_500_000_000,
    ]
    assert tuple(value.item() for value in actor.lifecycle_timestamps) == (
        1_000_000_000,
        1_500_001_000,
    )
    assert torch.allclose(
        actor.samples[0].quaternion_wxyz,
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
    )


def test_manifest_can_retain_complete_nuscenes_scan_and_stationary_tracks(
    tmp_path: Path,
) -> None:
    root = _make_nuscenes(tmp_path / "nuscenes")

    manifest = load_nuscenes_manifest(
        root,
        camera_channels=("CAM_FRONT",),
        retain_unprojected_lidar=True,
        include_stationary_actors=True,
    )

    assert manifest.frames[0].lidar is not None
    assert manifest.frames[0].lidar.points.shape == (3, 3)
    assert len(manifest.actor_tracks) == 2


def test_loader_streams_official_pretty_json_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_nuscenes(tmp_path / "nuscenes")
    # Exercise the same record-at-a-time path used for the multi-GB trainval
    # tables without making the unit-test fixture large.
    monkeypatch.setattr(nuscenes_data, "_SMALL_JSON_LIMIT", 0)

    manifest = load_nuscenes_manifest(
        root,
        camera_channels=("CAM_FRONT",),
    )

    assert len(manifest.frames) == 2
    assert len(manifest.actor_tracks) == 1


def test_manifest_requires_every_selected_camera_keyframe(tmp_path: Path) -> None:
    root = _make_nuscenes(tmp_path / "nuscenes")
    path = root / "v1.0-trainval" / "sample_data.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    records = [record for record in records if record["token"] != "data-1-cam_front_left"]
    _write_json(path, records)

    with pytest.raises(ValueError, match="CAM_FRONT_LEFT"):
        load_nuscenes_manifest(root, camera_channels=_CAMERAS)


def test_manifest_maps_sky_masks_by_camera_and_sample_data_token(
    tmp_path: Path,
) -> None:
    root = _make_nuscenes(tmp_path / "nuscenes")
    mask_root = tmp_path / "sky_masks"

    expected_paths: list[Path] = []
    for frame_index in range(2):
        for channel in _CAMERAS:
            token = f"data-{frame_index}-{channel.lower()}"
            path = mask_root / channel / f"{token}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"binary mask placeholder")
            expected_paths.append(path)

    manifest = load_nuscenes_manifest(
        root,
        camera_channels=_CAMERAS,
        sky_mask_root=mask_root,
    )

    assert [frame.sky_mask_path for frame in manifest.frames] == expected_paths


def test_manifest_keeps_rejected_raw_mask_and_maps_sky_validity(
    tmp_path: Path,
) -> None:
    root = _make_nuscenes(tmp_path / "nuscenes")
    sample_data_path = root / "v1.0-trainval" / "sample_data.json"
    records = json.loads(sample_data_path.read_text(encoding="utf-8"))
    rejected_token = "a" * 32
    for record in records:
        if record["token"] == "data-0-cam_front":
            record["token"] = rejected_token
            break
    _write_json(sample_data_path, records)

    mask_root = tmp_path / "sky_masks"
    for record in records:
        if not record["is_key_frame"]:
            continue
        parts = Path(str(record["filename"])).parts
        if len(parts) < 2 or parts[1] not in _CAMERAS:
            continue
        path = mask_root / parts[1] / f"{record['token']}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"raw mask remains present")

    manifest = load_nuscenes_manifest(
        root,
        camera_channels=_CAMERAS,
        sky_mask_root=mask_root,
        sky_mask_reject_tokens={rejected_token},
    )

    assert manifest.frames[0].sky_mask_path == (
        mask_root / "CAM_FRONT" / f"{rejected_token}.png"
    )
    assert manifest.frames[0].sky_supervision_valid is False
    assert all(frame.sky_supervision_valid for frame in manifest.frames[1:])

    with pytest.raises(ValueError, match="absent from the selected scene/cameras"):
        load_nuscenes_manifest(
            root,
            camera_channels=_CAMERAS,
            sky_mask_root=mask_root,
            sky_mask_reject_tokens={"b" * 32},
        )


def test_manifest_sky_mask_root_is_strict_about_missing_inputs(
    tmp_path: Path,
) -> None:
    root = _make_nuscenes(tmp_path / "nuscenes")

    with pytest.raises(FileNotFoundError, match="sky mask root does not exist"):
        load_nuscenes_manifest(
            root,
            camera_channels=_CAMERAS,
            sky_mask_root=tmp_path / "missing",
        )

    mask_root = tmp_path / "incomplete_masks"
    for channel in _CAMERAS:
        (mask_root / channel).mkdir(parents=True)
    present = mask_root / "CAM_FRONT" / "data-0-cam_front.png"
    present.write_bytes(b"binary mask placeholder")

    with pytest.raises(
        FileNotFoundError,
        match="CAM_FRONT_LEFT.*data-0-cam_front_left",
    ):
        load_nuscenes_manifest(
            root,
            camera_channels=_CAMERAS,
            sky_mask_root=mask_root,
        )
