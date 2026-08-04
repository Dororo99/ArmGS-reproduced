from __future__ import annotations

import importlib.util
from io import BytesIO
import json
from pathlib import Path
import sys

from PIL import Image
import pytest


pyarrow = pytest.importorskip("pyarrow")
parquet = pytest.importorskip("pyarrow.parquet")

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "prepare_waymo_v2_evaluation.py"
)
MODULE_NAME = "_armgs_prepare_waymo_v2_evaluation_for_tests"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
waymo_cli = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = waymo_cli
SPEC.loader.exec_module(waymo_cli)


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (6, 4), color).save(buffer, format="JPEG", quality=100)
    return buffer.getvalue()


def _write_table(path: Path, payload: dict[str, list[object]]) -> None:
    path.parent.mkdir(parents=True)
    parquet.write_table(pyarrow.table(payload), path)


def _make_waymo_v2(
    root: Path,
    *,
    sequence: str = "segment",
    missing_image_keys: frozenset[tuple[int, int]] = frozenset(),
) -> None:
    base = root / "training"
    timestamps = [1_000_000 + index * 100_000 for index in range(10)]
    image_rows = [
        (timestamp, camera_id)
        for frame_index, timestamp in enumerate(timestamps)
        for camera_id in (1, 2)
        if (frame_index, camera_id) not in missing_image_keys
    ]
    _write_table(
        base / "camera_image" / f"{sequence}.parquet",
        {
            "key.frame_timestamp_micros": [
                timestamp for timestamp, _ in image_rows
            ],
            "key.camera_name": [
                camera_id for _, camera_id in image_rows
            ],
            "[CameraImageComponent].image": [
                _jpeg_bytes((frame_index * 10, camera_id * 20, 30))
                for frame_index, (_, camera_id) in enumerate(image_rows)
            ],
        },
    )
    _write_table(
        base / "vehicle_pose" / f"{sequence}.parquet",
        {"key.frame_timestamp_micros": timestamps},
    )
    _write_table(
        base / "camera_calibration" / f"{sequence}.parquet",
        {
            "key.camera_name": [1, 2],
            "[CameraCalibrationComponent].intrinsic.f_u": [12.0, 10.0],
            "[CameraCalibrationComponent].intrinsic.f_v": [13.0, 11.0],
            "[CameraCalibrationComponent].intrinsic.c_u": [3.0, 3.0],
            "[CameraCalibrationComponent].intrinsic.c_v": [2.0, 2.0],
            "[CameraCalibrationComponent].width": [6, 6],
            "[CameraCalibrationComponent].height": [4, 4],
        },
    )
    component_keys = {
        "lidar": {
            "key.frame_timestamp_micros": [timestamps[0]],
            "key.laser_name": [1],
        },
        "lidar_pose": {
            "key.frame_timestamp_micros": [timestamps[0]],
            "key.laser_name": [1],
        },
        "lidar_box": {
            "key.frame_timestamp_micros": [timestamps[0]],
            "key.laser_object_id": ["actor"],
        },
        "lidar_calibration": {"key.laser_name": [1]},
    }
    for component, payload in component_keys.items():
        _write_table(base / component / f"{sequence}.parquet", payload)


def test_prepare_waymo_uses_streetgs_first_test_position_and_writes_manifests(
    tmp_path: Path,
) -> None:
    root = tmp_path / "waymo"
    output = tmp_path / "prepared"
    _make_waymo_v2(root)

    setup = waymo_cli.prepare_waymo_v2_evaluation(
        waymo_root=root,
        parquet_dir="training",
        sequence="segment",
        cameras=("FRONT",),
        start_frame=0,
        end_frame=None,
        test_every=4,
        first_test_position=4,
        target_size=(4, 6),
        output_directory=output,
        extract_images=True,
    )

    assert setup["counts"] == {
        "captures": 10,
        "camera_images": 10,
        "reconstruction_captures": 8,
        "reconstruction_images": 8,
        "novel_view_captures": 2,
        "novel_view_images": 2,
    }
    novel_indices = {
        frame["frame_index"]
        for frame in setup["frames"]
        if frame["split"] == "novel_view"
    }
    assert novel_indices == {4, 8}
    assert setup["frames"][0]["split"] == "reconstruction"
    with Image.open(output / "targets" / "FRONT" / "000000.png") as image:
        assert image.size == (6, 4)

    manifest = json.loads(
        (output / "novel_view_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["pairs"]) == 2
    assert manifest["metric_protocols"]["lpips_net"] == "alex"
    assert manifest["pairs"][0]["prediction"].startswith(
        "renders/novel_view/FRONT/"
    )
    assert manifest["pairs"][0]["target"].endswith(".png")


def test_prepare_waymo_is_capture_atomic_across_requested_cameras(
    tmp_path: Path,
) -> None:
    root = tmp_path / "waymo"
    _make_waymo_v2(root)

    setup = waymo_cli.prepare_waymo_v2_evaluation(
        waymo_root=root,
        parquet_dir="training",
        sequence="segment",
        cameras=("FRONT", "FRONT_LEFT"),
        start_frame=2,
        end_frame=7,
        test_every=4,
        first_test_position=4,
        target_size=(4, 6),
        output_directory=tmp_path / "index",
        extract_images=False,
    )

    assert setup["counts"]["captures"] == 6
    assert setup["counts"]["camera_images"] == 12
    assert setup["counts"]["novel_view_captures"] == 1
    held_out = [
        frame for frame in setup["frames"] if frame["split"] == "novel_view"
    ]
    assert {frame["camera"] for frame in held_out} == {
        "FRONT",
        "FRONT_LEFT",
    }
    assert {frame["frame_index"] for frame in held_out} == {4}


def test_prepare_waymo_requires_all_training_components(tmp_path: Path) -> None:
    root = tmp_path / "waymo"
    _make_waymo_v2(root)
    (
        root
        / "training"
        / "lidar_pose"
        / "segment.parquet"
    ).unlink()

    with pytest.raises(FileNotFoundError, match="lidar_pose"):
        waymo_cli.prepare_waymo_v2_evaluation(
            waymo_root=root,
            parquet_dir="training",
            sequence="segment",
            cameras=("FRONT",),
            start_frame=0,
            end_frame=None,
            test_every=4,
            first_test_position=4,
            target_size=(4, 6),
            output_directory=tmp_path / "index",
            extract_images=False,
        )


def test_prepare_waymo_uses_vehicle_pose_capture_positions_before_range_slice(
    tmp_path: Path,
) -> None:
    root = tmp_path / "waymo"
    _make_waymo_v2(root, missing_image_keys=frozenset({(0, 1)}))

    setup = waymo_cli.prepare_waymo_v2_evaluation(
        waymo_root=root,
        parquet_dir="training",
        sequence="segment",
        cameras=("FRONT",),
        start_frame=2,
        end_frame=7,
        test_every=4,
        first_test_position=4,
        target_size=(4, 6),
        output_directory=tmp_path / "index",
        extract_images=False,
    )

    assert setup["frames"][0]["source_frame_index"] == 2
    assert setup["frames"][-1]["source_frame_index"] == 7


def test_no_extract_invalidates_stale_metric_manifests(tmp_path: Path) -> None:
    root = tmp_path / "waymo"
    output = tmp_path / "prepared"
    _make_waymo_v2(root)
    output.mkdir()
    stale_paths = (
        output / "waymo_evaluation_setup.json",
        output / "reconstruction_manifest.json",
        output / "novel_view_manifest.json",
    )
    for path in stale_paths:
        path.write_text("stale", encoding="utf-8")

    waymo_cli.prepare_waymo_v2_evaluation(
        waymo_root=root,
        parquet_dir="training",
        sequence="segment",
        cameras=("FRONT",),
        start_frame=0,
        end_frame=None,
        test_every=4,
        first_test_position=4,
        target_size=(4, 6),
        output_directory=output,
        extract_images=False,
    )

    assert (output / "waymo_evaluation_setup.json").is_file()
    assert not (output / "reconstruction_manifest.json").exists()
    assert not (output / "novel_view_manifest.json").exists()


def test_prepare_waymo_matches_streetgs_uniform_bilinear_resize(
    tmp_path: Path,
) -> None:
    root = tmp_path / "waymo"
    output = tmp_path / "prepared"
    _make_waymo_v2(root)

    setup = waymo_cli.prepare_waymo_v2_evaluation(
        waymo_root=root,
        parquet_dir="training",
        sequence="segment",
        cameras=("FRONT",),
        start_frame=0,
        end_frame=9,
        test_every=4,
        first_test_position=4,
        target_size=(3, 5),
        output_directory=output,
        extract_images=True,
    )

    calibration = setup["calibration"]["FRONT"]
    assert calibration["uniform_scale"] == pytest.approx(5.0 / 6.0)
    assert calibration["target_size"] == [3, 5]
    assert calibration["intrinsics"][0][0] == pytest.approx(10.0)
    assert calibration["intrinsics"][1][1] == pytest.approx(13.0 * 5.0 / 6.0)
    assert setup["selected_source_range"] == {
        "start": 0,
        "end": 9,
        "end_inclusive": True,
    }
    with Image.open(output / "targets" / "FRONT" / "000000.png") as image:
        assert image.size == (5, 3)


def test_prepare_waymo_rejects_empty_reconstruction_split(tmp_path: Path) -> None:
    root = tmp_path / "waymo"
    _make_waymo_v2(root)

    with pytest.raises(ValueError, match="no reconstruction"):
        waymo_cli.prepare_waymo_v2_evaluation(
            waymo_root=root,
            parquet_dir="training",
            sequence="segment",
            cameras=("FRONT",),
            start_frame=0,
            end_frame=9,
            test_every=1,
            first_test_position=0,
            target_size=(4, 6),
            output_directory=tmp_path / "invalid",
            extract_images=False,
        )


def test_prepare_waymo_rejects_inclusive_end_past_sequence(tmp_path: Path) -> None:
    root = tmp_path / "waymo"
    _make_waymo_v2(root)

    with pytest.raises(ValueError, match=r"end_frame 10 is outside 10"):
        waymo_cli.prepare_waymo_v2_evaluation(
            waymo_root=root,
            parquet_dir="training",
            sequence="segment",
            cameras=("FRONT",),
            start_frame=0,
            end_frame=10,
            test_every=4,
            first_test_position=4,
            target_size=(4, 6),
            output_directory=tmp_path / "invalid_range",
            extract_images=False,
        )
