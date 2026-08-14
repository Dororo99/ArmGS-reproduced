from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sqlite3
import struct
import sys
from typing import Any

from PIL import Image
import pytest
import torch

from armgs.data.schema import (
    ActorTrack,
    ActorTrackSample,
    CanonicalDatasetManifest,
    CanonicalFrame,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_nuscenes_colmap.py"
)
MODULE_NAME = "_armgs_prepare_nuscenes_colmap_for_tests"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
colmap_cli = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = colmap_cli
SPEC.loader.exec_module(colmap_cli)


def _image(path: Path, value: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), color=(value, value, value)).save(path)
    return path


def _frame(path: Path, frame_index: int) -> CanonicalFrame:
    camera_to_world = torch.eye(4, dtype=torch.float64)
    camera_to_world[0, 3] = 0.25 * frame_index
    timestamp = torch.tensor((frame_index + 1) * 1_000_000_000, dtype=torch.int64)
    return CanonicalFrame(
        timestamp=timestamp,
        capture_timestamp=timestamp.clone(),
        camera_id=0,
        camera_convention="opencv",
        camera_to_world=camera_to_world,
        intrinsics=torch.tensor(
            [[50.0, 0.0, 50.0], [0.0, 50.0, 40.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ),
        image_path=path,
        image_size=(80, 100),
        frame_index=frame_index,
    )


def _track() -> ActorTrack:
    samples = tuple(
        ActorTrackSample(
            timestamp=torch.tensor((index + 1) * 1_000_000_000, dtype=torch.int64),
            translation=torch.tensor([0.2 * index, 0.0, 6.0], dtype=torch.float64),
            quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            frame_index=index,
        )
        for index in range(3)
    )
    return ActorTrack(
        actor_id=7,
        class_name="vehicle.car",
        dimensions_lwh=torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64),
        samples=samples,
    )


def _manifest(tmp_path: Path) -> CanonicalDatasetManifest:
    frames = tuple(
        _frame(_image(tmp_path / "raw" / f"frame-{index}.jpg", 30 + index), index)
        for index in range(3)
    )
    return CanonicalDatasetManifest(frames=frames, actor_tracks=(_track(),))


def _args(tmp_path: Path, *extra: str) -> Any:
    return colmap_cli.parse_args(
        [
            "--nuscenes-root",
            str(tmp_path / "nuscenes"),
            "--scene",
            "61",
            "--cameras",
            "CAM_FRONT",
            "--output-dir",
            str(tmp_path / "colmap"),
            "--split-every",
            "2",
            "--split-offset",
            "0",
            "--split-start-position",
            "1",
            *extra,
        ]
    )


def _selected(manifest: CanonicalDatasetManifest) -> tuple[Any, ...]:
    _split, selected = colmap_cli.select_training_frames(
        manifest, every=2, offset=0, start_position=1
    )
    return selected


def test_parser_supports_all_or_selected_cameras_and_safe_execution_modes(
    tmp_path: Path,
) -> None:
    all_args = colmap_cli.parse_args(
        [
            "--nuscenes-root",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "out"),
            "--cameras",
            "all",
            "--dry-run",
        ]
    )
    assert all_args.cameras == colmap_cli.NUSCENES_CAMERA_CHANNELS
    assert all_args.split_every == 8
    assert all_args.split_start_position == 1
    assert all_args.dry_run is True

    selected = _args(tmp_path, "--skip-execution")
    assert selected.cameras == ("CAM_FRONT",)
    assert selected.skip_execution is True

    with pytest.raises(argparse.ArgumentTypeError if False else SystemExit):
        _args(tmp_path, "--dry-run", "--skip-execution")
    with pytest.raises(argparse.ArgumentTypeError, match="unknown"):
        colmap_cli.parse_camera_channels("CAM_FRONT,CAM_UNKNOWN")
    with pytest.raises(argparse.ArgumentTypeError, match="unique"):
        colmap_cli.parse_camera_channels("CAM_FRONT,CAM_FRONT")


def test_periodic_selection_stages_only_training_rows_with_colmap_mask_names(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)

    split, selected = colmap_cli.select_training_frames(
        manifest, every=2, offset=0, start_position=1
    )

    assert split.train_source_indices == (0, 1)
    assert split.eval_source_indices == (2,)
    assert [record.frame.frame_index for record in selected] == [0, 1]
    assert [record.image_name for record in selected] == [
        "CAM_FRONT/frame-0.jpg",
        "CAM_FRONT/frame-1.jpg",
    ]
    assert selected[0].mask_name == "CAM_FRONT/frame-0.jpg.png"


def test_projected_actor_cuboid_masks_dynamic_pixels_black(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)

    mask, actor_ids, dynamic_pixels = colmap_cli.render_static_feature_mask(
        manifest.frames[0], manifest.actor_tracks
    )

    assert mask.mode == "L"
    assert mask.size == (100, 80)
    assert actor_ids == (7,)
    assert dynamic_pixels > 0
    assert mask.getpixel((50, 40)) == 0
    assert mask.getpixel((0, 0)) == 255


def test_skip_execution_stages_train_images_masks_and_mapping_without_colmap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(colmap_cli, "load_nuscenes_manifest", lambda *args, **kwargs: manifest)
    args = _args(tmp_path, "--skip-execution")

    payload = colmap_cli.prepare_nuscenes_colmap(args)

    output = args.output_dir
    assert payload["status"] == "staged"
    assert payload["split"]["train_source_indices"] == [0, 1]
    assert payload["split"]["eval_source_indices"] == [2]
    assert (output / "images" / "CAM_FRONT" / "frame-0.jpg").is_file()
    assert (output / "images" / "CAM_FRONT" / "frame-1.jpg").is_file()
    assert not (output / "images" / "CAM_FRONT" / "frame-2.jpg").exists()
    mask_path = output / "masks" / "CAM_FRONT" / "frame-0.jpg.png"
    with Image.open(mask_path) as mask:
        assert mask.mode == "L"
        assert mask.getpixel((50, 40)) == 0
        assert mask.getpixel((0, 0)) == 255
    mapping = json.loads((output / "mapping.json").read_text(encoding="utf-8"))
    assert mapping["known_pose_contract"] == {
        "camera_model": "PINHOLE",
        "camera_convention": "opencv",
        "world_frame": "nuscenes_global",
        "pose_refinement": False,
        "intrinsics_refinement": False,
    }
    assert len(mapping["frames"]) == 2
    assert mapping["frames"][0]["visible_dynamic_actor_ids"] == [7]

    with pytest.raises(FileExistsError, match="refusing to overwrite non-empty"):
        colmap_cli.prepare_nuscenes_colmap(args)


def test_dry_run_does_not_write_and_reports_train_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(colmap_cli, "load_nuscenes_manifest", lambda *args, **kwargs: manifest)
    args = _args(tmp_path, "--dry-run")

    payload = colmap_cli.prepare_nuscenes_colmap(args)

    assert payload["status"] == "dry_run"
    assert payload["summary"]["train_frame_count"] == 2
    assert payload["summary"]["eval_frame_count"] == 1
    assert not args.output_dir.exists()


def _make_database(path: Path, selected: tuple[Any, ...]) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE cameras ("
            "camera_id INTEGER PRIMARY KEY, model INTEGER NOT NULL, "
            "width INTEGER NOT NULL, height INTEGER NOT NULL, "
            "params BLOB, prior_focal_length INTEGER NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE images ("
            "image_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, "
            "camera_id INTEGER NOT NULL)"
        )
        connection.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (7, 0, 100, 80, struct.pack("<3d", 1.0, 2.0, 3.0), 0),
        )
        for image_id, record in enumerate(selected, start=11):
            connection.execute(
                "INSERT INTO images VALUES (?, ?, ?)",
                (image_id, record.image_name, 7),
            )
        connection.commit()


def test_database_intrinsics_and_known_pose_text_model_use_colmap_ids(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    selected = _selected(manifest)
    database = tmp_path / "database.db"
    _make_database(database, selected)

    database_images = colmap_cli.update_database_known_intrinsics(database, selected)

    assert [(image.image_id, image.camera_id) for image in database_images] == [
        (11, 7),
        (12, 7),
    ]
    with sqlite3.connect(database) as connection:
        model, width, height, params, prior = connection.execute(
            "SELECT model, width, height, params, prior_focal_length "
            "FROM cameras WHERE camera_id = 7"
        ).fetchone()
    assert (model, width, height, prior) == (1, 100, 80, 1)
    assert struct.unpack("<4d", params) == pytest.approx((50.0, 50.0, 50.0, 40.0))

    model_dir = tmp_path / "known"
    colmap_cli.write_known_pose_model(model_dir, selected, database_images)

    camera_data = [
        line
        for line in (model_dir / "cameras.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert camera_data == ["7 PINHOLE 100 80 50 50 50 40"]
    image_data = [
        line
        for line in (model_dir / "images.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    first = image_data[0].split()
    assert first[:9] == ["11", "1", "0", "0", "0", "0", "0", "0", "7"]
    assert first[9] == "CAM_FRONT/frame-0.jpg"
    second = image_data[1].split()
    assert second[:5] == ["12", "1", "0", "0", "0"]
    assert float(second[5]) == pytest.approx(-0.25)
    assert second[9] == "CAM_FRONT/frame-1.jpg"
    assert "Number of points: 0" in (model_dir / "points3D.txt").read_text(
        encoding="utf-8"
    )


def test_commands_are_argument_arrays_lock_calibration_and_run_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = colmap_cli.build_colmap_commands(tmp_path / "colmap", "my-colmap")
    feature = commands["feature_extractor"]
    assert feature[:2] == ["my-colmap", "feature_extractor"]
    assert feature[feature.index("--ImageReader.single_camera_per_folder") + 1] == "1"
    assert feature[feature.index("--ImageReader.mask_path") + 1].endswith("/masks")
    assert feature[feature.index("--ImageReader.camera_model") + 1] == "PINHOLE"
    triangulator = commands["point_triangulator"]
    for option in (
        "--Mapper.ba_refine_focal_length",
        "--Mapper.ba_refine_principal_point",
        "--Mapper.ba_refine_extra_params",
    ):
        assert triangulator[triangulator.index(option) + 1] == "0"
    converter = commands["model_converter"]
    assert converter[converter.index("--output_type") + 1] == "TXT"

    calls: list[tuple[list[str], bool]] = []

    def fake_run(command: list[str], *, check: bool) -> None:
        calls.append((command, check))

    monkeypatch.setattr(colmap_cli.subprocess, "run", fake_run)
    colmap_cli.run_checked(("my-colmap", "exhaustive_matcher"))
    assert calls == [(["my-colmap", "exhaustive_matcher"], True)]


def test_nonempty_points3d_contract_rejects_empty_triangulation(
    tmp_path: Path,
) -> None:
    points = tmp_path / "points3D.txt"
    points.write_text(
        "# 3D point list\n# Number of points: 0\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="zero points"):
        colmap_cli._require_nonempty_points3d(points)

    points.write_text(
        "# 3D point list\n1 0 0 0 255 255 255 0.1 1 2\n",
        encoding="utf-8",
    )
    assert colmap_cli._require_nonempty_points3d(points) == 1
