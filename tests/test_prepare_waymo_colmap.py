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


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_waymo_colmap.py"
MODULE_NAME = "_armgs_prepare_waymo_colmap_for_tests"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
waymo_cli = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = waymo_cli
SPEC.loader.exec_module(waymo_cli)


@pytest.fixture(autouse=True)
def _stub_full_context_world_center(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        waymo_cli,
        "_load_waymo_world_center",
        lambda *args, **kwargs: torch.tensor(
            [10.0, 20.0, 30.0], dtype=torch.float64
        ),
    )


def _write_image(path: Path, value: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), color=(value, value, value)).save(path)
    return path


def _frame(path: Path, frame_index: int, *, camera_id: int = 0) -> CanonicalFrame:
    camera_to_world = torch.eye(4, dtype=torch.float64)
    camera_to_world[0, 3] = 0.2 * frame_index
    timestamp = torch.tensor((frame_index + 1) * 100_000_000, dtype=torch.int64)
    return CanonicalFrame(
        timestamp=timestamp,
        capture_timestamp=timestamp.clone(),
        camera_id=camera_id,
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


def _actor_track() -> ActorTrack:
    samples = tuple(
        ActorTrackSample(
            timestamp=torch.tensor((index + 1) * 100_000_000, dtype=torch.int64),
            translation=torch.tensor([0.1 * index, 0.0, 6.0], dtype=torch.float64),
            quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            frame_index=index,
        )
        for index in range(6)
    )
    return ActorTrack(
        actor_id=3,
        class_name="TYPE_VEHICLE",
        dimensions_lwh=torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64),
        samples=samples,
    )


def _manifest(tmp_path: Path) -> CanonicalDatasetManifest:
    frames = tuple(
        _frame(_write_image(tmp_path / "cache" / f"{index:06d}.png", 20 + index), index)
        for index in range(6)
    )
    return CanonicalDatasetManifest(frames=frames, actor_tracks=(_actor_track(),))


def _args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    return waymo_cli.parse_args(
        [
            "--waymo-root",
            str(tmp_path / "waymo"),
            "--sequence",
            "segment-test",
            "--cameras",
            "FRONT",
            "--target-height",
            "80",
            "--target-width",
            "100",
            "--cache-dir",
            str(tmp_path / "decoded"),
            "--output-dir",
            str(tmp_path / "colmap"),
            *extra,
        ]
    )


def _selected(manifest: CanonicalDatasetManifest) -> tuple[Any, ...]:
    _split, selected = waymo_cli.select_training_frames(
        manifest, target_size=(80, 100)
    )
    return selected


def test_parser_defaults_to_paper_front_resolution_split_and_cpu(
    tmp_path: Path,
) -> None:
    args = waymo_cli.parse_args(
        [
            "--waymo-root",
            str(tmp_path / "waymo"),
            "--sequence",
            "segment",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert args.cameras == ("FRONT",)
    assert (args.target_height, args.target_width) == (1066, 1600)
    assert (args.split_every, args.split_offset, args.split_start_position) == (
        4,
        0,
        4,
    )
    assert args.use_gpu is False
    assert args.castrack_path is None
    assert waymo_cli.parse_camera_channels("all") == waymo_cli.WAYMO_CAMERA_CHANNELS
    with pytest.raises(argparse.ArgumentTypeError, match="unknown"):
        waymo_cli.parse_camera_channels("FRONT,BACK")
    with pytest.raises(SystemExit):
        _args(tmp_path, "--dry-run", "--skip-execution")


def test_selection_holds_out_streetgs_position_four_and_requires_target_size(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)

    split, selected = waymo_cli.select_training_frames(
        manifest, target_size=(80, 100)
    )

    assert split.train_source_indices == (0, 1, 2, 3, 5)
    assert split.eval_source_indices == (4,)
    assert [record.frame.frame_index for record in selected] == [0, 1, 2, 3, 5]
    assert selected[0].image_name == "FRONT/000000.png"
    assert selected[0].mask_name == "FRONT/000000.png.png"

    bad_frame = _frame(
        _write_image(tmp_path / "bad.png", 0),
        0,
    )
    bad_manifest = CanonicalDatasetManifest(frames=(bad_frame, manifest.frames[1]))
    with pytest.raises(ValueError, match="paper target"):
        waymo_cli.select_training_frames(
            bad_manifest,
            every=2,
            offset=1,
            start_position=1,
            target_size=(1066, 1600),
        )


def test_skip_execution_calls_waymo_loader_stages_train_rgb_and_actor_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    loader_calls: list[dict[str, Any]] = []

    def fake_loader(root: Path, **kwargs: Any) -> CanonicalDatasetManifest:
        loader_calls.append({"root": root, **kwargs})
        return manifest

    monkeypatch.setattr(waymo_cli, "_load_waymo_v2_manifest", fake_loader)
    args = _args(tmp_path, "--skip-execution")

    payload = waymo_cli.prepare_waymo_colmap(args)

    assert loader_calls == [
        {
            "root": args.waymo_root,
            "sequence": "segment-test",
            "parquet_dir": "training",
            "camera_channels": ("FRONT",),
            "start_frame": 0,
            "end_frame": None,
            "target_size": (80, 100),
            "cache_dir": args.cache_dir.resolve(),
            "castrack_path": None,
        }
    ]
    assert payload["status"] == "staged"
    assert payload["split"]["eval_source_indices"] == [4]
    output = args.output_dir
    assert (output / "images" / "FRONT" / "000000.png").is_file()
    assert not (output / "images" / "FRONT" / "000004.png").exists()
    mask_path = output / "masks" / "FRONT" / "000000.png.png"
    with Image.open(mask_path) as mask:
        assert mask.mode == "L"
        assert mask.getpixel((50, 40)) == 0
        assert mask.getpixel((0, 0)) == 255
    mapping = json.loads((output / "mapping.json").read_text(encoding="utf-8"))
    assert mapping["paper_resolution"] == [80, 100]
    assert mapping["frames"][0]["visible_dynamic_actor_ids"] == [3]
    assert mapping["known_pose_contract"]["sift_gpu"] is False
    assert mapping["known_pose_contract"]["world_frame"] == (
        "waymo_world_centered"
    )
    assert mapping["coordinate_frame"] == {
        "name": "waymo_world_centered",
        "centered": True,
        "centering_method": "full_context_mean_vehicle_translation",
        "world_center_m": [10.0, 20.0, 30.0],
    }

    with pytest.raises(FileExistsError, match="refusing to overwrite non-empty"):
        waymo_cli.prepare_waymo_colmap(args)


def test_colmap_loader_forwards_centering_and_optional_castrack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    sentinel = object()

    def fake_loader(root: Path, **kwargs: Any) -> object:
        captured["root"] = root
        captured.update(kwargs)
        return sentinel

    monkeypatch.setitem(
        sys.modules,
        "armgs.data.waymo",
        type("FakeWaymo", (), {"load_waymo_v2_manifest": staticmethod(fake_loader)}),
    )
    castrack = tmp_path / "castrack.json"
    castrack.write_text("{}\n", encoding="utf-8")
    result = waymo_cli._load_waymo_v2_manifest(
        tmp_path / "waymo",
        sequence="segment-test",
        parquet_dir="validation",
        camera_channels=("FRONT",),
        start_frame=0,
        end_frame=85,
        target_size=(1066, 1600),
        cache_dir=tmp_path / "cache",
        castrack_path=castrack,
    )

    assert result is sentinel
    assert captured["center_world"] is True
    assert captured["castrack_path"] == castrack
    assert captured["require_lidar"] is False


def test_dry_run_records_castrack_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(
        waymo_cli, "_load_waymo_v2_manifest", lambda *args, **kwargs: manifest
    )
    castrack = tmp_path / "tracking" / "castrack.json"
    castrack.parent.mkdir()
    castrack.write_text('{"scene": {}}\n', encoding="utf-8")
    args = _args(
        tmp_path,
        "--castrack-path",
        str(castrack),
        "--actor-box-scale",
        "2.0",
        "--dry-run",
    )

    payload = waymo_cli.prepare_waymo_colmap(args)

    assert payload["actor_mask"]["tracker_source"] == "castrack"
    assert payload["actor_mask"]["castrack_path"] == str(castrack.resolve())
    assert payload["actor_mask"]["box_scale"] == 2.0


def test_dry_run_leaves_colmap_output_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(
        waymo_cli, "_load_waymo_v2_manifest", lambda *args, **kwargs: manifest
    )
    args = _args(tmp_path, "--dry-run")

    payload = waymo_cli.prepare_waymo_colmap(args)

    assert payload["status"] == "dry_run"
    assert payload["summary"]["train_image_count"] == 5
    assert payload["summary"]["eval_image_count"] == 1
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
            (9, 0, 100, 80, struct.pack("<3d", 1.0, 2.0, 3.0), 0),
        )
        for image_id, record in enumerate(selected, start=20):
            connection.execute(
                "INSERT INTO images VALUES (?, ?, ?)",
                (image_id, record.image_name, 9),
            )
        connection.commit()


def test_waymo_wrappers_update_pinhole_db_and_write_known_world_poses(
    tmp_path: Path,
) -> None:
    selected = _selected(_manifest(tmp_path))
    database = tmp_path / "database.db"
    _make_database(database, selected)

    database_images = waymo_cli.update_database_known_intrinsics(database, selected)
    with sqlite3.connect(database) as connection:
        model, params = connection.execute(
            "SELECT model, params FROM cameras WHERE camera_id = 9"
        ).fetchone()
    assert model == 1
    assert struct.unpack("<4d", params) == pytest.approx((50.0, 50.0, 50.0, 40.0))

    model_dir = tmp_path / "known"
    waymo_cli.write_known_pose_model(model_dir, selected, database_images)
    camera_lines = [
        line
        for line in (model_dir / "cameras.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert camera_lines == ["9 PINHOLE 100 80 50 50 50 40"]
    image_lines = [
        line
        for line in (model_dir / "images.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert image_lines[0].split()[:10] == [
        "20",
        "1",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "9",
        "FRONT/000000.png",
    ]
    assert float(image_lines[1].split()[5]) == pytest.approx(-0.2)


def test_colmap_391_commands_default_to_cpu_and_lock_intrinsics(tmp_path: Path) -> None:
    commands = waymo_cli.build_colmap_commands(
        tmp_path / "out", "/usr/bin/colmap"
    )
    feature = commands["feature_extractor"]
    matcher = commands["exhaustive_matcher"]
    triangulator = commands["point_triangulator"]

    assert feature[feature.index("--SiftExtraction.use_gpu") + 1] == "0"
    assert matcher[matcher.index("--SiftMatching.use_gpu") + 1] == "0"
    assert feature[feature.index("--ImageReader.single_camera_per_folder") + 1] == "1"
    for option in (
        "--Mapper.ba_refine_focal_length",
        "--Mapper.ba_refine_principal_point",
        "--Mapper.ba_refine_extra_params",
    ):
        assert triangulator[triangulator.index(option) + 1] == "0"
    assert triangulator[triangulator.index("--Mapper.fix_existing_images") + 1] == "1"
    assert commands["model_converter"][-2:] == ["--output_type", "TXT"]

    gpu = waymo_cli.build_colmap_commands(
        tmp_path / "gpu", "colmap", use_gpu=True
    )
    assert gpu["feature_extractor"][-1] == "1"
    assert gpu["exhaustive_matcher"][-1] == "1"


def test_complete_pipeline_fails_and_records_empty_points3d(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(
        waymo_cli, "_load_waymo_v2_manifest", lambda *args, **kwargs: manifest
    )

    def fake_database(
        _database_path: Path, staged: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return tuple(
            waymo_cli._known_pose.DatabaseImage(
                image_id=index + 1,
                name=record.image_name,
                camera_id=1,
            )
            for index, record in enumerate(staged)
        )

    def fake_model(
        model_dir: Path, _staged: tuple[Any, ...], _images: tuple[Any, ...]
    ) -> None:
        model_dir.mkdir()

    def fake_run(command: list[str]) -> None:
        if command[1] == "model_converter":
            output_path = Path(command[command.index("--output_path") + 1])
            (output_path / "points3D.txt").write_text(
                "# Number of points: 0\n", encoding="utf-8"
            )

    monkeypatch.setattr(waymo_cli, "update_database_known_intrinsics", fake_database)
    monkeypatch.setattr(waymo_cli, "write_known_pose_model", fake_model)
    monkeypatch.setattr(waymo_cli, "run_checked", fake_run)
    args = _args(tmp_path)

    with pytest.raises(RuntimeError, match="empty points3D"):
        waymo_cli.prepare_waymo_colmap(args)

    mapping = json.loads(
        (args.output_dir / "mapping.json").read_text(encoding="utf-8")
    )
    assert mapping["status"] == "failed_empty_points3D"
    assert mapping["summary"]["sfm_point_count"] == 0
