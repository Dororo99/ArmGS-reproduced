from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("waymo_open_dataset.v2")
Image = pytest.importorskip("PIL.Image")

import armgs.data.waymo as waymo_data
from armgs.data import (
    WAYMO_ACTOR_SOURCE,
    WAYMO_OPENCV_TO_NATIVE,
    load_waymo_world_center,
    load_waymo_v2_manifest,
)


_SEQUENCE = "synthetic_waymo_context"
_TIMESTAMPS = (1_000_000, 1_100_000)
_SOURCE_SIZE = (7, 12)
_TARGET_SIZE = (3, 6)


def test_tensorflow_guard_rejects_a_late_visible_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LateConfig:
        @staticmethod
        def set_visible_devices(devices: object, device_type: str) -> None:
            assert devices == []
            assert device_type == "GPU"
            raise RuntimeError("runtime already initialized")

        @staticmethod
        def get_visible_devices(device_type: str) -> list[str]:
            assert device_type == "GPU"
            return ["GPU:0"]

    monkeypatch.setitem(sys.modules, "tensorflow", SimpleNamespace(config=LateConfig()))
    with pytest.raises(RuntimeError, match="before ArmGS could disable"):
        waymo_data._configure_tensorflow_cpu_only()


def _transform(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> list[float]:
    return [
        1.0,
        0.0,
        0.0,
        tx,
        0.0,
        1.0,
        0.0,
        ty,
        0.0,
        0.0,
        1.0,
        tz,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def _jpeg(width: int = 12, height: int = 7) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), color=(20, 40, 60)).save(
        buffer, format="JPEG"
    )
    return buffer.getvalue()


def _write_component(
    root: Path,
    component: str,
    rows: list[dict[str, object]],
) -> None:
    path = root / "training" / component / f"{_SEQUENCE}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _camera_calibration_row() -> dict[str, object]:
    return {
        "key.segment_context_name": _SEQUENCE,
        "key.camera_name": 1,
        "[CameraCalibrationComponent].intrinsic.f_u": 6.0,
        "[CameraCalibrationComponent].intrinsic.f_v": 8.0,
        "[CameraCalibrationComponent].intrinsic.c_u": 6.0,
        "[CameraCalibrationComponent].intrinsic.c_v": 4.0,
        "[CameraCalibrationComponent].intrinsic.k1": 0.0,
        "[CameraCalibrationComponent].intrinsic.k2": 0.0,
        "[CameraCalibrationComponent].intrinsic.p1": 0.0,
        "[CameraCalibrationComponent].intrinsic.p2": 0.0,
        "[CameraCalibrationComponent].intrinsic.k3": 0.0,
        "[CameraCalibrationComponent].extrinsic.transform": _transform(1.0, 2.0, 3.0),
        "[CameraCalibrationComponent].width": _SOURCE_SIZE[1],
        "[CameraCalibrationComponent].height": _SOURCE_SIZE[0],
        "[CameraCalibrationComponent].rolling_shutter_direction": 1,
    }


def _camera_image_row(
    timestamp: int,
    *,
    pose_x: float,
    pose_timestamp: float,
) -> dict[str, object]:
    return {
        "key.segment_context_name": _SEQUENCE,
        "key.frame_timestamp_micros": timestamp,
        "key.camera_name": 1,
        "[CameraImageComponent].image": _jpeg(),
        "[CameraImageComponent].pose.transform": _transform(pose_x),
        "[CameraImageComponent].velocity.linear_velocity.x": 0.0,
        "[CameraImageComponent].velocity.linear_velocity.y": 0.0,
        "[CameraImageComponent].velocity.linear_velocity.z": 0.0,
        "[CameraImageComponent].velocity.angular_velocity.x": 0.0,
        "[CameraImageComponent].velocity.angular_velocity.y": 0.0,
        "[CameraImageComponent].velocity.angular_velocity.z": 0.0,
        "[CameraImageComponent].pose_timestamp": pose_timestamp,
        "[CameraImageComponent].rolling_shutter_params.shutter": 0.001,
        "[CameraImageComponent].rolling_shutter_params.camera_trigger_time": (
            pose_timestamp - 0.01
        ),
        "[CameraImageComponent].rolling_shutter_params.camera_readout_done_time": (
            pose_timestamp + 0.01
        ),
    }


def _vehicle_pose_row(timestamp: int, tx: float) -> dict[str, object]:
    return {
        "key.segment_context_name": _SEQUENCE,
        "key.frame_timestamp_micros": timestamp,
        "[VehiclePoseComponent].world_from_vehicle.transform": _transform(tx),
    }


def _box_row(
    timestamp: int,
    object_id: str,
    *,
    box_type: int,
    center_x: float,
    size: tuple[float, float, float] = (4.0, 2.0, 1.5),
) -> dict[str, object]:
    return {
        "key.segment_context_name": _SEQUENCE,
        "key.frame_timestamp_micros": timestamp,
        "key.laser_object_id": object_id,
        "[LiDARBoxComponent].box.center.x": center_x,
        "[LiDARBoxComponent].box.center.y": 0.0,
        "[LiDARBoxComponent].box.center.z": 0.0,
        "[LiDARBoxComponent].box.size.x": size[0],
        "[LiDARBoxComponent].box.size.y": size[1],
        "[LiDARBoxComponent].box.size.z": size[2],
        "[LiDARBoxComponent].box.heading": 0.0,
        "[LiDARBoxComponent].type": box_type,
        "[LiDARBoxComponent].num_lidar_points_in_box": 5,
        "[LiDARBoxComponent].num_top_lidar_points_in_box": 3,
        "[LiDARBoxComponent].speed.x": 0.0,
        "[LiDARBoxComponent].speed.y": 0.0,
        "[LiDARBoxComponent].speed.z": 0.0,
        "[LiDARBoxComponent].acceleration.x": 0.0,
        "[LiDARBoxComponent].acceleration.y": 0.0,
        "[LiDARBoxComponent].acceleration.z": 0.0,
        "[LiDARBoxComponent].difficulty_level.detection": 1,
        "[LiDARBoxComponent].difficulty_level.tracking": 1,
    }


def _range_values(
    ranges: tuple[float, float],
    intensities: tuple[float, float],
) -> list[float]:
    values: list[float] = []
    for distance, intensity in zip(ranges, intensities):
        values.extend((distance, intensity, 0.0, 0.0))
    return values


def _lidar_row(timestamp: int, laser_id: int) -> dict[str, object]:
    return {
        "key.segment_context_name": _SEQUENCE,
        "key.frame_timestamp_micros": timestamp,
        "key.laser_name": laser_id,
        "[LiDARComponent].range_image_return1.values": _range_values(
            (1.0, 2.0), (0.1, 0.2)
        ),
        "[LiDARComponent].range_image_return1.shape": [1, 2, 4],
        "[LiDARComponent].range_image_return2.values": _range_values(
            (3.0, 4.0), (0.3, 0.4)
        ),
        "[LiDARComponent].range_image_return2.shape": [1, 2, 4],
    }


def _lidar_calibration_row(laser_id: int) -> dict[str, object]:
    return {
        "key.segment_context_name": _SEQUENCE,
        "key.laser_name": laser_id,
        "[LiDARCalibrationComponent].extrinsic.transform": _transform(),
        "[LiDARCalibrationComponent].beam_inclination.min": 0.0,
        "[LiDARCalibrationComponent].beam_inclination.max": 0.0,
        "[LiDARCalibrationComponent].beam_inclination.values": [0.0],
    }


def _lidar_pose_row(timestamp: int, world_x: float) -> dict[str, object]:
    values = [0.0, 0.0, 0.0, world_x, 0.0, 0.0] * 2
    return {
        "key.segment_context_name": _SEQUENCE,
        "key.frame_timestamp_micros": timestamp,
        "key.laser_name": 1,
        "[LiDARPoseComponent].range_image_return1.values": values,
        "[LiDARPoseComponent].range_image_return1.shape": [1, 2, 6],
    }


def _make_waymo(root: Path, *, with_lidar: bool = False) -> Path:
    root.mkdir()
    _write_component(root, "camera_calibration", [_camera_calibration_row()])
    _write_component(
        root,
        "camera_image",
        [
            _camera_image_row(_TIMESTAMPS[0], pose_x=10.0, pose_timestamp=1.00025),
            _camera_image_row(_TIMESTAMPS[1], pose_x=13.0, pose_timestamp=1.10025),
        ],
    )
    _write_component(
        root,
        "vehicle_pose",
        [
            _vehicle_pose_row(_TIMESTAMPS[0], 0.0),
            _vehicle_pose_row(_TIMESTAMPS[1], 3.0),
        ],
    )
    _write_component(
        root,
        "lidar_box",
        [
            _box_row(_TIMESTAMPS[0], "moving", box_type=1, center_x=5.0),
            _box_row(
                _TIMESTAMPS[1],
                "moving",
                box_type=1,
                center_x=5.0,
                size=(4.2, 2.2, 1.6),
            ),
            _box_row(_TIMESTAMPS[0], "stationary", box_type=2, center_x=2.0),
            _box_row(_TIMESTAMPS[1], "stationary", box_type=2, center_x=-1.0),
            _box_row(_TIMESTAMPS[0], "sign", box_type=3, center_x=1.0),
            _box_row(_TIMESTAMPS[1], "sign", box_type=3, center_x=-2.0),
        ],
    )
    if with_lidar:
        _write_component(
            root,
            "lidar_calibration",
            [_lidar_calibration_row(1), _lidar_calibration_row(2)],
        )
        _write_component(
            root,
            "lidar",
            [
                _lidar_row(timestamp, laser_id)
                for timestamp in _TIMESTAMPS
                for laser_id in (1, 2)
            ],
        )
        _write_component(
            root,
            "lidar_pose",
            [
                _lidar_pose_row(_TIMESTAMPS[0], 0.0),
                _lidar_pose_row(_TIMESTAMPS[1], 3.0),
            ],
        )
    return root


def _write_sky_masks(root: Path) -> None:
    for source_index in range(2):
        path = root / _SEQUENCE / "FRONT" / f"{source_index:08d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (_TARGET_SIZE[1], _TARGET_SIZE[0]), color=255).save(path)


def _write_castrack(path: Path) -> Path:
    scene_key = f"segment-{_SEQUENCE}_with_camera_labels"
    payload = {
        scene_key: {
            str(source_index): {
                "obj_ids": [701],
                "name": ["Cyclist"],
                # The synthetic camera looks along vehicle/world +x.  Its
                # embedded camera pose is intentionally offset from the
                # capture vehicle pose, so these boxes exercise projection
                # through the final CanonicalFrame rather than a pose shortcut.
                "boxes_lidar": [
                    [16.0, 2.0, 3.0, 4.0, 1.0, 1.0, 0.0]
                ],
            }
            for source_index in range(2)
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_camera_manifest_uses_waymo_axis_uniform_k_exact_timestamps_and_cache(
    tmp_path: Path,
) -> None:
    root = _make_waymo(tmp_path / "waymo")
    sky_root = tmp_path / "sky"
    _write_sky_masks(sky_root)

    manifest = load_waymo_v2_manifest(
        root,
        sequence=_SEQUENCE,
        target_size=_TARGET_SIZE,
        cache_dir=tmp_path / "cache",
        sky_mask_root=sky_root,
        require_lidar=False,
    )

    assert len(manifest) == 2
    frame = manifest[0]
    expected_pose = torch.eye(4, dtype=torch.float64)
    expected_pose[:3, 3] = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)
    expected_extrinsic = torch.eye(4, dtype=torch.float64)
    expected_extrinsic[:3, 3] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    expected_c2w = expected_pose @ expected_extrinsic @ WAYMO_OPENCV_TO_NATIVE

    assert frame.camera_id == 0
    assert frame.camera_convention == "opencv"
    assert torch.allclose(frame.camera_to_world, expected_c2w)
    assert torch.allclose(
        frame.intrinsics,
        torch.tensor(
            [[3.0, 0.0, 3.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ),
    )
    assert int(frame.timestamp.item()) == 1_000_250_000
    assert int(frame.capture_timestamp.item()) == 1_000_000_000
    assert frame.image_size == _TARGET_SIZE
    assert frame.image_path == (
        tmp_path
        / "cache"
        / _SEQUENCE
        / "images"
        / "FRONT"
        / "00000000.png"
    )
    with Image.open(frame.image_path) as image:
        assert image.size == (_TARGET_SIZE[1], _TARGET_SIZE[0])
    assert frame.sky_mask_path == (
        sky_root / _SEQUENCE / "FRONT" / "00000000.png"
    )

    assert WAYMO_ACTOR_SOURCE == "waymo_gt"
    assert len(manifest.actor_tracks) == 1
    actor = manifest.actor_tracks[0]
    assert actor.actor_id == 0
    assert actor.class_name == "vehicle"
    assert torch.allclose(
        actor.dimensions_lwh,
        torch.tensor([4.2, 2.2, 1.6], dtype=torch.float64),
    )
    assert [sample.frame_index for sample in actor.samples] == [0, 1]
    assert [sample.translation[0].item() for sample in actor.samples] == [5.0, 8.0]
    lifecycle_start, lifecycle_end = actor.lifecycle_timestamps
    assert int(lifecycle_start.item()) <= int(actor.samples[0].timestamp.item())
    assert int(lifecycle_end.item()) >= int(actor.samples[-1].timestamp.item())

    unfiltered = load_waymo_v2_manifest(
        root,
        sequence=_SEQUENCE,
        target_size=_TARGET_SIZE,
        cache_dir=tmp_path / "cache",
        require_lidar=False,
        filter_static_actors=False,
    )
    assert [(track.actor_id, track.class_name) for track in unfiltered.actor_tracks] == [
        (0, "vehicle"),
        (1, "pedestrian"),
    ]


def test_uniform_intrinsic_scaling_rejects_incompatible_target_geometry(
    tmp_path: Path,
) -> None:
    root = _make_waymo(tmp_path / "waymo")

    with pytest.raises(ValueError, match="incompatible with uniform Waymo"):
        load_waymo_v2_manifest(
            root,
            sequence=_SEQUENCE,
            target_size=(4, 6),
            cache_dir=tmp_path / "cache",
            require_lidar=False,
        )


def test_lidar_decodes_all_sensors_both_returns_and_top_motion_pose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _make_waymo(tmp_path / "waymo", with_lidar=True)
    from waymo_open_dataset.v2.perception.utils import lidar_utils

    original = lidar_utils.convert_range_image_to_point_cloud
    calls: list[tuple[bool, bool]] = []

    def wrapped(*args: object, **kwargs: object) -> object:
        calls.append(
            (kwargs.get("pixel_pose") is not None, kwargs.get("frame_pose") is not None)
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(lidar_utils, "convert_range_image_to_point_cloud", wrapped)
    manifest = load_waymo_v2_manifest(
        root,
        sequence=_SEQUENCE,
        start_frame=0,
        end_frame=0,
        target_size=_TARGET_SIZE,
        cache_dir=tmp_path / "cache",
        require_lidar=True,
        retain_unprojected_lidar=True,
    )

    lidar = manifest[0].lidar
    assert lidar is not None
    assert lidar.points.shape == (8, 3)
    assert torch.allclose(
        torch.sort(lidar.reflectance).values,
        torch.tensor([0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4]),
        atol=1.0e-6,
    )
    assert torch.allclose(lidar.sensor_to_world, torch.eye(4, dtype=torch.float64))
    assert manifest[0].lidar_projection is not None
    # Two sensors x two returns: TOP receives one shared per-pixel motion pose
    # for both returns, while the non-TOP sensor is converted without it.
    assert calls == [(True, True), (True, True), (False, False), (False, False)]

    calls.clear()
    first_return_manifest = load_waymo_v2_manifest(
        root,
        sequence=_SEQUENCE,
        start_frame=0,
        end_frame=0,
        target_size=_TARGET_SIZE,
        cache_dir=tmp_path / "cache",
        require_lidar=True,
        lidar_returns="first",
        retain_unprojected_lidar=True,
    )
    first_return_lidar = first_return_manifest[0].lidar
    assert first_return_lidar is not None
    assert first_return_lidar.points.shape == (4, 3)
    assert torch.allclose(
        torch.sort(first_return_lidar.reflectance).values,
        torch.tensor([0.1, 0.1, 0.2, 0.2]),
        atol=1.0e-6,
    )
    # First return only: TOP uses its motion pose and the non-TOP sensor does not.
    assert calls == [(True, True), (False, False)]
    import tensorflow as tf

    assert tf.config.get_visible_devices("GPU") == []


def test_camera_only_mode_never_invokes_lidar_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _make_waymo(tmp_path / "waymo")

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("LiDAR decode must not run")

    monkeypatch.setattr(waymo_data, "_decode_lidar_frames", forbidden)
    manifest = load_waymo_v2_manifest(
        root,
        sequence=_SEQUENCE,
        camera_ids=(1,),
        start_frame=1,
        end_frame=1,
        target_size=_TARGET_SIZE,
        cache_dir=tmp_path / "cache",
        require_lidar=False,
        filter_static_actors=False,
    )

    assert len(manifest) == 1
    assert manifest[0].frame_index == 0
    assert manifest[0].image_path.name == "00000001.png"
    assert manifest[0].lidar is None


def test_castrack_branch_skips_gt_boxes_and_preserves_centered_world(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _make_waymo(tmp_path / "waymo")
    castrack_path = _write_castrack(tmp_path / "castrack_scene.json")

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("Waymo GT actor boxes must not be parsed for CAStrack")

    monkeypatch.setattr(waymo_data, "_actor_tracks", forbidden)
    manifest = load_waymo_v2_manifest(
        root,
        sequence=_SEQUENCE,
        target_size=_TARGET_SIZE,
        cache_dir=tmp_path / "cache",
        require_lidar=False,
        center_world=True,
        castrack_path=castrack_path,
    )

    assert len(manifest.actor_tracks) == 1
    actor = manifest.actor_tracks[0]
    assert actor.class_name == "cyclist"
    assert [sample.frame_index for sample in actor.samples] == [0, 1]
    torch.testing.assert_close(
        torch.stack([sample.translation for sample in actor.samples]),
        torch.tensor(
            ((14.5, 2.0, 3.0), (17.5, 2.0, 3.0)), dtype=torch.float64
        ),
    )
    torch.testing.assert_close(
        actor.dimensions_lwh,
        torch.tensor((4.0, 1.0, 1.0), dtype=torch.float64),
    )


def test_castrack_uses_absolute_source_index_and_requires_front(
    tmp_path: Path,
) -> None:
    root = _make_waymo(tmp_path / "waymo")
    castrack_path = _write_castrack(tmp_path / "castrack_scene.json")

    manifest = load_waymo_v2_manifest(
        root,
        sequence=_SEQUENCE,
        start_frame=1,
        end_frame=1,
        target_size=_TARGET_SIZE,
        cache_dir=tmp_path / "cache",
        require_lidar=False,
        filter_static_actors=False,
        castrack_path=castrack_path,
    )

    assert len(manifest.actor_tracks) == 1
    sample = manifest.actor_tracks[0].samples[0]
    assert sample.frame_index == 0
    assert int(sample.timestamp.item()) == _TIMESTAMPS[1] * 1_000
    torch.testing.assert_close(
        sample.translation,
        torch.tensor((19.0, 2.0, 3.0), dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="requires the FRONT camera"):
        load_waymo_v2_manifest(
            root,
            sequence=_SEQUENCE,
            camera_channels=("FRONT_LEFT",),
            target_size=_TARGET_SIZE,
            cache_dir=tmp_path / "cache",
            require_lidar=False,
            castrack_path=castrack_path,
        )


def test_full_context_world_center_is_shared_by_camera_lidar_and_actor(
    tmp_path: Path,
) -> None:
    root = _make_waymo(tmp_path / "waymo", with_lidar=True)

    center = load_waymo_world_center(root, sequence=_SEQUENCE)
    assert torch.allclose(
        center,
        torch.tensor([1.5, 0.0, 0.0], dtype=torch.float64),
    )

    manifest = load_waymo_v2_manifest(
        root,
        sequence=_SEQUENCE,
        start_frame=1,
        end_frame=1,
        target_size=_TARGET_SIZE,
        cache_dir=tmp_path / "cache",
        require_lidar=True,
        retain_unprojected_lidar=True,
        filter_static_actors=False,
        center_world=True,
    )

    # Only the second capture is selected, but the center still includes both
    # source poses (0 and 3), rather than collapsing to the selected pose at 3.
    frame = manifest[0]
    expected_camera = torch.eye(4, dtype=torch.float64)
    expected_camera[:3, 3] = torch.tensor([12.5, 2.0, 3.0], dtype=torch.float64)
    expected_camera = expected_camera @ WAYMO_OPENCV_TO_NATIVE
    assert torch.allclose(frame.camera_to_world, expected_camera)

    assert frame.lidar is not None
    expected_lidar_pose = torch.eye(4, dtype=torch.float64)
    expected_lidar_pose[0, 3] = 1.5
    assert torch.allclose(frame.lidar.sensor_to_world, expected_lidar_pose)

    moving_actor = next(
        track for track in manifest.actor_tracks if track.class_name == "vehicle"
    )
    assert len(moving_actor.samples) == 1
    assert torch.allclose(
        moving_actor.samples[0].translation,
        torch.tensor([6.5, 0.0, 0.0], dtype=torch.float64),
    )
