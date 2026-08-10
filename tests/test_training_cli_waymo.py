from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys
from typing import Any

import pytest
import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_armgs_waymo.py"
MODULE_NAME = "_armgs_waymo_train_cli_for_tests"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
waymo_training_cli = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = waymo_training_cli
SPEC.loader.exec_module(waymo_training_cli)


OFFICIAL_COMPONENTS = (
    "camera_image",
    "camera_calibration",
    "lidar",
    "lidar_pose",
    "lidar_box",
    "lidar_calibration",
    "vehicle_pose",
)


def _official_launcher_environment(
    tmp_path: Path,
    *,
    sequence: str,
    start_frame: int,
    end_frame: int,
    with_castrack: bool = True,
) -> dict[str, str]:
    root = tmp_path / "waymo"
    prepared = tmp_path / "prepared"
    for component in OFFICIAL_COMPONENTS:
        path = root / "validation" / component / f"{sequence}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(component.encode("utf-8"))
    colmap = prepared / "colmap" / sequence
    points = colmap / "triangulated_text" / "points3D.txt"
    points.parent.mkdir(parents=True, exist_ok=True)
    points.write_text("1 0 0 0 255 255 255 0\n", encoding="utf-8")
    (colmap / "mapping.json").write_text("{}\n", encoding="utf-8")
    masks = prepared / "sky_masks" / sequence / "FRONT"
    masks.mkdir(parents=True, exist_ok=True)
    for source_index in range(start_frame, end_frame + 1):
        (masks / f"{source_index:08d}.png").write_bytes(b"mask")
    if with_castrack:
        castrack = prepared / "tracking" / "castrack" / f"{sequence}.json"
        castrack.parent.mkdir(parents=True, exist_ok=True)
        castrack.write_text("{}\n", encoding="utf-8")

    environment = os.environ.copy()
    environment.pop("ACTOR_BOX_SCALE", None)
    environment.pop("CAS_TRACK_PATH", None)
    environment.update(
        {
            "WAYMO_ROOT": str(root),
            "PREPARED_ROOT": str(prepared),
            "OUTPUT_DIR": str(tmp_path / "output"),
            "WANDB_DIR": str(tmp_path / "wandb"),
            "ARMGS_PYTHON": "/bin/true",
            "WANDB_ENABLED": "0",
            "DRY_RUN": "1",
        }
    )
    return environment


def _base_cli(tmp_path: Path) -> list[str]:
    return [
        "--waymo-root",
        str(tmp_path / "waymo"),
        "--sequence",
        "context-001",
        "--output-dir",
        str(tmp_path / "output"),
    ]


def _paper_protocol_config(*, iterations: int = 30_000) -> dict[str, Any]:
    return {
        "model": {"sh_degree": 1},
        "initialization": {
            "voxel_size": None,
            "streetgs_waymo": {
                "actor_min_lidar_points": 2_000,
                "actor_max_lidar_points": None,
                "actor_fallback_grid_resolution": 20,
                "actor_fallback_random_seed": 0,
            },
        },
        "data": {"waymo": {"scene_extent": 20.0}},
        "optimization": {
            "iterations": iterations,
            "densification": {
                "prune_actor_outside_box": True,
                "max_screen_radius": None,
            },
        },
    }


def test_parser_exposes_waymo_paper_initialization_eval_and_wandb_options(
    tmp_path: Path,
) -> None:
    args = waymo_training_cli.parse_args(
        [
            "--root",
            str(tmp_path / "waymo"),
            "--parquet-dir",
            "training",
            "--sequence",
            "context-001",
            "--start-frame",
            "20",
            "--end-frame",
            "100",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--sky-mask-root",
            str(tmp_path / "sky"),
            "--colmap-points3d",
            str(tmp_path / "points3D.txt"),
            "--castrack-path",
            str(tmp_path / "castrack.json"),
            "--actor-box-scale",
            "2.0",
            "--camera",
            "FRONT",
            "--lidar-initialization-frames",
            "train-only",
            "--paper-mode",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "output"),
            "--resume",
            str(tmp_path / "checkpoint.pt"),
            "--iterations",
            "30000",
            "--checkpoint-interval",
            "2000",
            "--log-interval",
            "50",
            "--image-log-interval",
            "500",
            "--eval-interval",
            "2000",
            "--no-eval-at-end",
            "--no-eval-reconstruction-at-end",
            "--eval-lpips",
            "--eval-lpips-net",
            "vgg",
            "--eval-only",
            "--wandb",
            "--wandb-entity",
            "CamoSplat_ICLR_2027",
            "--wandb-project",
            "Ours-ArmGS-Waymo",
            "--wandb-run-name",
            "waymo-001",
            "--wandb-run-id",
            "resume-123",
            "--wandb-mode",
            "offline",
            "--wandb-dir",
            str(tmp_path / "wandb"),
            "--wandb-fail-fast",
            "--wandb-log-checkpoint-artifact",
        ]
    )

    assert args.waymo_root == tmp_path / "waymo"
    assert args.parquet_dir == "training"
    assert args.sequence == "context-001"
    assert (args.start_frame, args.end_frame) == (20, 100)
    assert args.cache_dir == tmp_path / "cache"
    assert args.sky_mask_root == tmp_path / "sky"
    assert args.colmap_points3d == tmp_path / "points3D.txt"
    assert args.castrack_path == tmp_path / "castrack.json"
    assert args.actor_box_scale == 2.0
    assert args.camera == "FRONT"
    assert args.lidar_initialization_frames == "train-only"
    assert args.lidar_returns == "first"
    assert args.paper_mode is True
    assert args.iterations == 30_000
    assert args.image_log_interval == 500
    assert args.eval_interval == 2000
    assert args.eval_at_end is False
    assert args.eval_reconstruction_at_end is False
    assert args.eval_lpips is True
    assert args.eval_lpips_net == "vgg"
    assert args.eval_only is True
    assert args.wandb_entity == "CamoSplat_ICLR_2027"
    assert args.wandb_project == "Ours-ArmGS-Waymo"
    assert args.wandb_run_id == "resume-123"
    assert args.wandb_fail_fast is True
    assert args.wandb_log_checkpoint_artifact is True


def test_parser_paper_defaults_and_guards(tmp_path: Path) -> None:
    args = waymo_training_cli.parse_args(_base_cli(tmp_path))

    assert args.parquet_dir == "validation"
    assert args.start_frame == 0
    assert args.end_frame is None
    assert args.lidar_returns == "first"
    assert args.camera == "FRONT"
    assert (args.target_height, args.target_width) == (1066, 1600)
    assert args.lidar_initialization_frames == "all-selected"
    assert args.castrack_path is None
    assert args.actor_box_scale is None
    assert args.image_log_interval == 500
    assert args.eval_interval == 1000
    assert args.eval_at_end is True
    assert args.eval_reconstruction_at_end is True
    assert args.eval_lpips is False
    assert args.eval_lpips_net == "alex"
    assert args.wandb_entity == "CamoSplat_ICLR_2027"
    assert args.wandb_project == "Ours-ArmGS-Waymo"

    with pytest.raises(SystemExit):
        waymo_training_cli.parse_args([*_base_cli(tmp_path), "--eval-only"])
    with pytest.raises(SystemExit):
        waymo_training_cli.parse_args(
            [*_base_cli(tmp_path), "--start-frame", "5", "--end-frame", "4"]
        )
    with pytest.raises(SystemExit):
        waymo_training_cli.parse_args(
            [*_base_cli(tmp_path), "--image-log-interval", "-1"]
        )
    with pytest.raises(SystemExit):
        waymo_training_cli.parse_args(
            [*_base_cli(tmp_path), "--camera", "FRONT_LEFT"]
        )
    with pytest.raises(SystemExit):
        waymo_training_cli.parse_args(
            [*_base_cli(tmp_path), "--actor-box-scale", "0"]
        )


def test_paper_mode_requires_colmap_sky_end_frame_and_30k(tmp_path: Path) -> None:
    args = waymo_training_cli.parse_args([*_base_cli(tmp_path), "--paper-mode"])
    with pytest.raises(ValueError, match="colmap-points3d"):
        waymo_training_cli.validate_paper_protocol(
            args, _paper_protocol_config()
        )

    complete = waymo_training_cli.parse_args(
        [
            *_base_cli(tmp_path),
            "--paper-mode",
            "--end-frame",
            "85",
            "--sky-mask-root",
            str(tmp_path / "sky"),
            "--colmap-points3d",
            str(tmp_path / "points3D.txt"),
            "--castrack-path",
            str(tmp_path / "castrack.json"),
            "--eval-lpips",
        ]
    )
    waymo_training_cli.validate_paper_protocol(
        complete, _paper_protocol_config()
    )
    with pytest.raises(ValueError, match="30000"):
        waymo_training_cli.validate_paper_protocol(
            complete, _paper_protocol_config(iterations=29_999)
        )
    without_castrack = argparse.Namespace(**vars(complete))
    without_castrack.castrack_path = None
    with pytest.raises(ValueError, match="castrack-path"):
        waymo_training_cli.validate_paper_protocol(
            without_castrack, _paper_protocol_config()
        )

    config_deviations = (
        (("model", "sh_degree"), 3, "sh_degree must be 1"),
        (("initialization", "voxel_size"), 0.15, "voxel_size must be null"),
        (("data", "waymo", "scene_extent"), 19.0, "scene_extent must be 20"),
        (
            ("optimization", "densification", "prune_actor_outside_box"),
            False,
            "prune_actor_outside_box must be true",
        ),
        (
            ("optimization", "densification", "max_screen_radius"),
            20,
            "max_screen_radius must be null",
        ),
        (
            (
                "initialization",
                "streetgs_waymo",
                "actor_min_lidar_points",
            ),
            1_999,
            "actor_min_lidar_points must be 2000",
        ),
        (
            (
                "initialization",
                "streetgs_waymo",
                "actor_max_lidar_points",
            ),
            20_000,
            "actor_max_lidar_points must be null",
        ),
        (
            (
                "initialization",
                "streetgs_waymo",
                "actor_fallback_grid_resolution",
            ),
            19,
            "actor_fallback_grid_resolution must be 20",
        ),
    )
    for path, value, message in config_deviations:
        deviating_config = json.loads(json.dumps(_paper_protocol_config()))
        target = deviating_config
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError, match=message):
            waymo_training_cli.validate_paper_protocol(
                complete, deviating_config
            )

    missing_screen_policy = _paper_protocol_config()
    del missing_screen_policy["optimization"]["densification"][
        "max_screen_radius"
    ]
    with pytest.raises(ValueError, match="max_screen_radius must be null"):
        waymo_training_cli.validate_paper_protocol(
            complete, missing_screen_policy
        )

    deviations = (
        ("lidar_initialization_frames", "train-only", "all-selected"),
        ("lidar_returns", "both", "lidar-returns must be first"),
        ("eval_at_end", False, "eval-at-end"),
        ("eval_reconstruction_at_end", False, "reconstruction-at-end"),
        ("eval_lpips", False, "eval-lpips is required"),
        ("eval_lpips_net", "vgg", "must be alex"),
    )
    for field, value, message in deviations:
        deviating = argparse.Namespace(**vars(complete))
        setattr(deviating, field, value)
        with pytest.raises(ValueError, match=message):
            waymo_training_cli.validate_paper_protocol(
                deviating,
                _paper_protocol_config(),
            )


def test_waymo_profile_explicitly_disables_dense_actor_lidar_cap() -> None:
    config = waymo_training_cli.load_config(
        SCRIPT_PATH.parents[1] / "configs" / "armgs_waymo_streetgs.yaml"
    )
    settings = (
        waymo_training_cli.streetgs_waymo_actor_initialization_settings(
            config
        )
    )

    assert settings == {
        "minimum_lidar_points": 2_000,
        "maximum_lidar_points": None,
        "fallback_grid_resolution": 20,
        "random_seed": 0,
    }

    malformed = json.loads(json.dumps(config))
    malformed["initialization"]["streetgs_waymo"][
        "actor_min_lidar_points"
    ] = True
    with pytest.raises(ValueError, match="must be an integer"):
        waymo_training_cli.streetgs_waymo_actor_initialization_settings(
            malformed
        )

    missing = json.loads(json.dumps(config))
    del missing["initialization"]["streetgs_waymo"][
        "actor_max_lidar_points"
    ]
    with pytest.raises(ValueError, match="missing actor point setting"):
        waymo_training_cli.streetgs_waymo_actor_initialization_settings(
            missing
        )


def test_loader_adapter_forwards_front_resolution_cache_masks_and_lidar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
        SimpleNamespace(load_waymo_v2_manifest=fake_loader),
    )
    result = waymo_training_cli.load_waymo_manifest(
        tmp_path / "waymo",
        sequence="context-001",
        parquet_dir="validation",
        camera="FRONT",
        start_frame=0,
        end_frame=85,
        target_size=(1066, 1600),
        lidar_returns="first",
        cache_dir=tmp_path / "cache",
        sky_mask_root=tmp_path / "sky",
        castrack_path=tmp_path / "castrack.json",
    )

    assert result is sentinel
    assert captured == {
        "root": tmp_path / "waymo",
        "sequence": "context-001",
        "parquet_dir": "validation",
        "camera_channels": ("FRONT",),
        "start_frame": 0,
        "end_frame": 85,
        "lidar_returns": "first",
        "target_size": (1066, 1600),
        "cache_dir": tmp_path / "cache",
        "sky_mask_root": tmp_path / "sky",
        "require_lidar": True,
        "center_world": True,
        "castrack_path": tmp_path / "castrack.json",
    }


def test_context_center_adapter_uses_the_complete_sequence_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def fake_center(root: Path, **kwargs: Any) -> torch.Tensor:
        captured["root"] = root
        captured.update(kwargs)
        return torch.tensor([10.0, -4.0, 2.0], dtype=torch.float64)

    monkeypatch.setitem(
        sys.modules,
        "armgs.data.waymo",
        SimpleNamespace(load_waymo_world_center=fake_center),
    )
    center = waymo_training_cli.load_waymo_context_center(
        tmp_path / "waymo",
        sequence="context-001",
        parquet_dir="validation",
    )

    torch.testing.assert_close(center, torch.tensor([10.0, -4.0, 2.0]))
    assert captured == {
        "root": tmp_path / "waymo",
        "sequence": "context-001",
        "parquet_dir": "validation",
    }


def test_split_is_fixed_to_streetgs_relative_positions_four_eight_and_onward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    sentinel = object()

    def fake_split(manifest: Any, **kwargs: Any) -> object:
        captured["manifest"] = manifest
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        waymo_training_cli, "periodic_train_eval_split", fake_split
    )
    manifest = object()

    assert waymo_training_cli.split_waymo_manifest(manifest) is sentinel
    assert captured == {
        "manifest": manifest,
        "every": 4,
        "offset": 0,
        "start_position": 4,
    }


def test_lidar_initialization_policy_defaults_to_all_selected_but_can_be_strict(
) -> None:
    selected = object()
    training = object()
    assert (
        waymo_training_cli.lidar_initialization_manifest(
            selected, training, "all-selected"
        )
        is selected
    )
    assert (
        waymo_training_cli.lidar_initialization_manifest(
            selected, training, "train-only"
        )
        is training
    )
    with pytest.raises(ValueError, match="unknown"):
        waymo_training_cli.lidar_initialization_manifest(
            selected, training, "test-only"
        )


def test_all_selected_actor_clouds_are_limited_to_trainable_tracks() -> None:
    background = waymo_training_cli.CanonicalScenePointClouds(
        background=waymo_training_cli.collect_colored_lidar_point_clouds.__globals__[
            "ColoredPointCloud"
        ](
            torch.tensor([[0.0, 0.0, 0.0]]),
            torch.tensor([[0.5, 0.5, 0.5]]),
        ),
        actors={
            1: waymo_training_cli.collect_colored_lidar_point_clouds.__globals__[
                "ColoredPointCloud"
            ](
                torch.tensor([[1.0, 0.0, 0.0]]),
                torch.tensor([[1.0, 0.0, 0.0]]),
            ),
            2: waymo_training_cli.collect_colored_lidar_point_clouds.__globals__[
                "ColoredPointCloud"
            ](
                torch.tensor([[2.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 1.0, 0.0]]),
            ),
        },
    )
    training_manifest = SimpleNamespace(
        actor_tracks=(SimpleNamespace(actor_id=1),)
    )

    filtered = waymo_training_cli.retain_training_actors(
        background, training_manifest
    )

    assert filtered.background is background.background
    assert set(filtered.actors) == {1}


def test_dataset_identity_covers_component_parquets_cache_sky_and_colmap(
    tmp_path: Path,
) -> None:
    root = tmp_path / "waymo"
    for component in ("camera_image", "vehicle_pose", "lidar"):
        path = root / "validation" / component / "context-001.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(component.encode("utf-8"))
    image = tmp_path / "cache" / "front.png"
    lidar = tmp_path / "cache" / "lidar.pt"
    sky = tmp_path / "sky" / "front.png"
    colmap = tmp_path / "points3D.txt"
    castrack = tmp_path / "castrack.json"
    for path, content in (
        (image, b"image"),
        (lidar, b"lidar-cache"),
        (sky, b"sky"),
        (colmap, b"1 0 0 0 255 255 255 0\n"),
        (castrack, b"{\"frames\": []}\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    frame = SimpleNamespace(
        image_path=image,
        lidar=SimpleNamespace(source_path=lidar),
        sky_mask_path=sky,
        actor_mask_path=None,
    )

    before = waymo_training_cli.build_waymo_dataset_input_identity(
        [frame],
        root=root,
        parquet_dir="validation",
        sequence="context-001",
        colmap_points3d=colmap,
        castrack_path=castrack,
    )
    component = root / "validation" / "vehicle_pose" / "context-001.parquet"
    component.write_bytes(b"changed-pose")
    after_component = waymo_training_cli.build_waymo_dataset_input_identity(
        [frame],
        root=root,
        parquet_dir="validation",
        sequence="context-001",
        colmap_points3d=colmap,
        castrack_path=castrack,
    )
    colmap.write_bytes(b"1 1 0 0 255 255 255 0\n")
    after_colmap = waymo_training_cli.build_waymo_dataset_input_identity(
        [frame],
        root=root,
        parquet_dir="validation",
        sequence="context-001",
        colmap_points3d=colmap,
        castrack_path=castrack,
    )
    castrack.write_bytes(b"{\"frames\": [1]}\n")
    after_castrack = waymo_training_cli.build_waymo_dataset_input_identity(
        [frame],
        root=root,
        parquet_dir="validation",
        sequence="context-001",
        colmap_points3d=colmap,
        castrack_path=castrack,
    )

    assert before["component_file_count"] == 3
    assert before["file_count"] == 8
    assert before["digest_sha256"] != after_component["digest_sha256"]
    assert after_component["digest_sha256"] != after_colmap["digest_sha256"]
    assert after_colmap["digest_sha256"] != after_castrack["digest_sha256"]


def test_colmap_mapping_verifies_exact_train_only_split_and_rejects_mismatch(
    tmp_path: Path,
) -> None:
    output = tmp_path / "colmap"
    points = output / "triangulated_text" / "points3D.txt"
    points.parent.mkdir(parents=True)
    points.write_text("1 0 0 0 255 255 255 0\n", encoding="utf-8")
    mapping = {
        "dataset": "waymo_v2",
        "sequence": "context-001",
        "status": "complete",
        "final_points3D_path": str(points.resolve()),
        "camera_channels": ["FRONT"],
        "known_pose_contract": {
            "camera_model": "PINHOLE",
            "camera_convention": "opencv",
            "world_frame": "waymo_world",
            "pose_refinement": False,
            "intrinsics_refinement": False,
        },
        "split": {
            "every": 4,
            "offset": 0,
            "start_position": 4,
            "train_source_indices": [0, 1, 2, 3, 5],
            "eval_source_indices": [4],
        },
        "frames": [
            {"source_index": index} for index in (0, 1, 2, 3, 5)
        ],
    }
    mapping_path = output / "mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    provenance = waymo_training_cli.load_colmap_provenance(
        points,
        sequence="context-001",
        train_source_indices=[0, 1, 2, 3, 5],
        eval_source_indices=[4],
    )
    assert provenance == {
        "verified": True,
        "mapping": str(mapping_path.resolve()),
        "reason": "train_rows_match_runtime_split",
        "coordinate_frame": {
            "name": "waymo_world",
            "centered": False,
            "declaration": "implicit_legacy_absolute_world",
            "centering_method": None,
            "world_center_m": None,
        },
    }

    mapping["frames"].append({"source_index": 4})
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(ValueError, match="not exactly training rows"):
        waymo_training_cli.load_colmap_provenance(
            points,
            sequence="context-001",
            train_source_indices=[0, 1, 2, 3, 5],
            eval_source_indices=[4],
        )


def test_colmap_alignment_subtracts_legacy_center_once_and_preserves_declared_centered(
) -> None:
    points = torch.tensor([[12.0, 18.0, 4.0], [10.0, 20.0, 3.0]])
    center = torch.tensor([10.0, 20.0, 3.0])
    legacy_frame = waymo_training_cli._colmap_coordinate_frame_contract(
        {}, {"world_frame": "waymo_world"}
    )
    legacy, legacy_metadata = (
        waymo_training_cli.align_colmap_points_to_centered_world(
            points,
            world_center=center,
            provenance={"coordinate_frame": legacy_frame},
        )
    )
    torch.testing.assert_close(
        legacy, torch.tensor([[2.0, -2.0, 1.0], [0.0, 0.0, 0.0]])
    )
    assert legacy_metadata["operation"] == (
        "subtract_full_context_world_center_once"
    )

    centered_frame = waymo_training_cli._colmap_coordinate_frame_contract(
        {
            "coordinate_frame": {
                "name": "waymo_world_centered",
                "centering_method": (
                    "full_context_mean_vehicle_translation"
                ),
                "world_center_m": [10.0, 20.0, 3.0],
            }
        },
        {"world_frame": "waymo_world_centered"},
    )
    centered, centered_metadata = (
        waymo_training_cli.align_colmap_points_to_centered_world(
            points,
            world_center=center,
            provenance={"coordinate_frame": centered_frame},
        )
    )
    assert centered is points
    assert centered_metadata["operation"] == "none_already_centered"
    with pytest.raises(ValueError, match="does not match this context"):
        waymo_training_cli.align_colmap_points_to_centered_world(
            points,
            world_center=torch.tensor([9.0, 20.0, 3.0]),
            provenance={"coordinate_frame": centered_frame},
        )


def test_colmap_without_mapping_is_recorded_as_unverified(tmp_path: Path) -> None:
    points = tmp_path / "points3D.txt"
    points.write_text("1 0 0 0 255 255 255 0\n", encoding="utf-8")

    assert waymo_training_cli.load_colmap_provenance(
        points,
        sequence="context-001",
        train_source_indices=[0],
        eval_source_indices=[1],
    ) == {
        "verified": False,
        "mapping": None,
        "reason": "prepare_waymo_colmap_mapping_not_found",
        "coordinate_frame": {
            "name": "waymo_world",
            "centered": False,
            "declaration": "implicit_legacy_absolute_world",
            "centering_method": None,
            "world_center_m": None,
        },
    }



def test_paper_mode_hard_fails_unverified_colmap_provenance() -> None:
    provenance = {"verified": False, "reason": "mapping_not_found"}
    with pytest.raises(ValueError, match="verified train-only mapping.json"):
        waymo_training_cli.enforce_colmap_provenance(
            provenance,
            paper_mode=True,
        )
    waymo_training_cli.enforce_colmap_provenance(
        provenance, paper_mode=False
    )


def test_paper_mode_requires_attached_actor_masks_on_every_training_row() -> None:
    manifest = [
        SimpleNamespace(actor_mask_path=Path("first.png")),
        SimpleNamespace(actor_mask_path=None),
    ]
    with pytest.raises(ValueError, match=r"missing rows \[1\]"):
        waymo_training_cli.enforce_paper_actor_mask_supervision(
            manifest,
            paper_mode=True,
        )
    waymo_training_cli.enforce_paper_actor_mask_supervision(
        manifest,
        paper_mode=False,
    )

class _FakeWandbRun:
    def __init__(self) -> None:
        self.id = "waymo-run-123"
        self.entity = "CamoSplat_ICLR_2027"
        self.project = "Ours-ArmGS-Waymo"
        self.name = "waymo-context-001"
        self.url = "https://wandb.invalid/waymo-run-123"
        self.summary: dict[str, Any] = {}
        self.logged: list[tuple[dict[str, Any], int, bool]] = []

    def log(
        self, payload: dict[str, Any], *, step: int, commit: bool = True
    ) -> None:
        self.logged.append((payload, step, commit))


def test_wandb_config_is_waymo_specific_and_records_initialization_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    fake_run = _FakeWandbRun()

    def fake_init(**kwargs: Any) -> _FakeWandbRun:
        captured.update(kwargs)
        return fake_run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))
    output = tmp_path / "output"
    output.mkdir()
    args = waymo_training_cli.parse_args(
        [
            *_base_cli(tmp_path),
            "--wandb",
            "--wandb-mode",
            "offline",
            "--wandb-run-id",
            "waymo-run-123",
        ]
    )
    initialization = {
        "modalities": ["lidar", "sfm"],
        "lidar_frames": "all-selected",
        "sfm_frames": "train-only",
    }
    metadata = {
        "waymo_root": str(tmp_path / "waymo"),
        "parquet_dir": "validation",
        "sequence": "context-001",
        "source_frame_range": {"start": 0, "end_inclusive": 85},
        "camera_channels": ["FRONT"],
        "target_resolution": [1066, 1600],
        "cache_dir": str(tmp_path / "cache"),
        "sky_mask_root": str(tmp_path / "sky"),
        "sky_mask_count": 64,
        "tracker_source": "waymo_gt",
        "split_protocol": {
            "type": "streetgs_periodic",
            "every": 4,
            "offset": 0,
            "start_position": 4,
        },
        "train_source_indices": [0, 1, 2, 3, 5],
        "eval_source_indices": [4],
        "initialization": initialization,
        "dataset_input_identity": {"digest_sha256": "dataset-sha256"},
    }

    assert waymo_training_cli._initialize_wandb(
        args,
        config={"optimization": {"iterations": 30_000}},
        run_metadata=metadata,
    ) is fake_run

    assert captured["entity"] == "CamoSplat_ICLR_2027"
    assert captured["project"] == "Ours-ArmGS-Waymo"
    assert captured["id"] == "waymo-run-123"
    assert captured["resume"] == "allow"
    wandb_config = captured["config"]
    assert wandb_config["dataset"]["type"] == "waymo_v2"
    assert wandb_config["dataset"]["camera_channels"] == ["FRONT"]
    assert wandb_config["dataset"]["tracker_source"] == "waymo_gt"
    assert wandb_config["initialization"] == initialization
    assert wandb_config["logging"]["image_interval"] == 500
    assert wandb_config["evaluation"]["reconstruction_at_end"] is True
    assert wandb_config["evaluation"]["lpips"] is False
    assert wandb_config["evaluation"]["lpips_net"] == "alex"
    assert wandb_config["evaluation"]["metric_protocols"]["psnr"] == (
        "mean-per-image-rgb-mse-data-range-1"
    )
    assert wandb_config["evaluation"]["metric_protocols"]["ssim"] == (
        "3dgs-gaussian-11x11-sigma-1.5-data-range-1"
    )


def test_train_preview_uses_waymo_front_camera_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(Image=lambda data, *, caption: {"caption": caption}),
    )
    batch = SimpleNamespace(
        target_rgb=torch.zeros((1, 2, 2, 3)),
        view=SimpleNamespace(
            camera_id=torch.tensor(0),
            timestamp=torch.tensor(123, dtype=torch.int64),
            training_row=torch.tensor(0),
        ),
    )
    output = SimpleNamespace(
        rendering=SimpleNamespace(rgb=torch.ones((1, 2, 2, 3))),
        step=500,
    )

    payload = waymo_training_cli._wandb_image_payload_factory(
        batch,
        output,
        training_manifest=[SimpleNamespace(frame_index=7)],
        training_source_indices=[9],
    )

    assert payload["train/image_camera"] == "FRONT"
    assert "Camera: FRONT (0)" in payload["train/gt_vs_render"]["caption"]
    assert payload["train/image_frame_index"] == 7
    assert payload["train/image_source_index"] == 9




def test_same_step_novel_and_reconstruction_wandb_logs_commit_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(Image=lambda data, *, caption: caption),
    )
    run = _FakeWandbRun()

    def record(split_name: str) -> dict[str, Any]:
        return {
            "split": split_name,
            "step": 30_000,
            "aggregate": {"psnr": 30.0},
            "per_camera": {"FRONT": {"ssim": 0.9}},
        }

    waymo_training_cli._log_to_wandb(
        run,
        {"step": 30_000, "loss": 1.0},
        commit=False,
    )
    waymo_training_cli._log_evaluation_to_wandb(
        run,
        record("novel_view"),
        {"FRONT": "novel-preview"},
        commit=False,
    )
    waymo_training_cli._log_evaluation_to_wandb(
        run,
        record("reconstruction"),
        {"FRONT": "reconstruction-preview"},
        commit=True,
    )

    train_payload, train_step, train_commit = run.logged[0]
    novel_payload, novel_step, novel_commit = run.logged[1]
    recon_payload, recon_step, recon_commit = run.logged[2]
    assert (train_step, train_commit) == (30_000, False)
    assert (novel_step, novel_commit) == (30_000, False)
    assert (recon_step, recon_commit) == (30_000, True)
    assert train_payload["train/loss"] == 1.0
    assert "novel_view/psnr" in novel_payload
    assert "novel_view/FRONT/gt_vs_render" in novel_payload
    assert "reconstruction/psnr" in recon_payload
    assert "reconstruction/FRONT/gt_vs_render" in recon_payload


def test_eval_only_resume_appends_at_current_wandb_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(Image=lambda data, *, caption: caption),
    )
    run = _FakeWandbRun()
    run.step = 501
    record = {
        "split": "novel_view",
        "step": 500,
        "aggregate": {"psnr": 28.0},
        "per_camera": {"FRONT": {"ssim": 0.85}},
    }

    waymo_training_cli._log_evaluation_to_wandb(
        run,
        record,
        {"FRONT": "preview"},
    )

    payload, step, commit = run.logged[0]
    assert (step, commit) == (501, True)
    assert payload["evaluation/checkpoint_step"] == 500
    assert payload["novel_view/checkpoint_step"] == 500


@pytest.mark.parametrize(
    ("step", "evaluation_interval", "final_due", "expected"),
    [
        (100, 0, False, True),
        (100, 100, False, False),
        (500, 0, True, False),
        (500, 100, True, False),
    ],
)
def test_wandb_training_record_commit_waits_for_same_step_evaluation(
    step: int,
    evaluation_interval: int,
    final_due: bool,
    expected: bool,
) -> None:
    assert (
        waymo_training_cli._wandb_training_record_commit(
            step=step,
            total_iterations=500,
            evaluation_interval=evaluation_interval,
            final_evaluation_due=final_due,
        )
        is expected
    )

class _EvaluationRenderer(torch.nn.Module):
    def forward(self, view: Any) -> Any:
        assert self.training is False
        assert torch.is_inference_mode_enabled()
        return SimpleNamespace(rgb=torch.full((1, 12, 12, 3), 0.1))


class _EvaluationManifest(list[Any]):
    actor_tracks: tuple[Any, ...] = ()


def test_held_out_evaluation_reports_front_psnr_ssim_and_optional_lpips_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _EvaluationManifest([SimpleNamespace(camera_id=0, frame_index=4)])
    monkeypatch.setattr(
        waymo_training_cli,
        "canonical_frame_to_training_batch",
        lambda frame, training_row, device: SimpleNamespace(
            view=SimpleNamespace(frame_index=frame.frame_index),
            target_rgb=torch.zeros((1, 12, 12, 3)),
        ),
    )
    monkeypatch.setattr(
        waymo_training_cli,
        "project_actor_boxes_to_mask",
        lambda frame, tracks, box_scale: torch.zeros((12, 12), dtype=torch.bool),
    )
    renderer = _EvaluationRenderer()
    renderer.train()
    policy = {
        "lpips": False,
        "lpips_net": "alex",
        "metric_protocols": {
            "psnr": waymo_training_cli._PSNR_PROTOCOL,
            "ssim": waymo_training_cli._SSIM_PROTOCOL,
            "lpips": None,
        },
    }

    record = waymo_training_cli.evaluate_waymo_split(
        renderer,
        manifest,
        device="cpu",
        output_directory=tmp_path,
        step=30_000,
        evaluation_policy=policy,
    )

    assert renderer.training is True
    assert record["split"] == "novel_view"
    assert set(record["per_camera"]) == {"FRONT"}
    assert record["aggregate"]["num_images"] == 1
    assert record["aggregate"]["psnr"] == pytest.approx(20.0)
    assert record["aggregate"]["ssim"] is not None
    assert record["aggregate"]["lpips"] is None
    assert record["policy"] == policy
    assert (
        tmp_path
        / "evaluation"
        / "novel_view"
        / "step_00030000_FRONT_gt_render.png"
    ).is_file()



def test_reconstruction_evaluation_uses_exact_embedding_rows_and_split_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _EvaluationManifest(
        [
            SimpleNamespace(camera_id=0, frame_index=0),
            SimpleNamespace(camera_id=0, frame_index=1),
        ]
    )
    observed_rows: list[int | None] = []

    def fake_batch(frame: Any, training_row: int | None, device: Any) -> Any:
        observed_rows.append(training_row)
        return SimpleNamespace(
            view=SimpleNamespace(frame_index=frame.frame_index),
            target_rgb=torch.zeros((1, 12, 12, 3)),
        )

    monkeypatch.setattr(
        waymo_training_cli, "canonical_frame_to_training_batch", fake_batch
    )
    monkeypatch.setattr(
        waymo_training_cli,
        "project_actor_boxes_to_mask",
        lambda frame, tracks, box_scale: torch.zeros((12, 12), dtype=torch.bool),
    )
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(Image=lambda data, *, caption: caption),
    )
    run = _FakeWandbRun()
    record = waymo_training_cli.evaluate_waymo_split(
        _EvaluationRenderer(),
        manifest,
        device="cpu",
        output_directory=tmp_path,
        step=30_000,
        split_name="reconstruction",
        training_rows=range(len(manifest)),
        wandb_run=run,
    )

    assert observed_rows == [0, 1]
    assert record["split"] == "reconstruction"
    assert record["aggregate"]["num_images"] == 2
    assert (
        tmp_path
        / "evaluation"
        / "reconstruction"
        / "step_00030000_FRONT_gt_render.png"
    ).is_file()
    payload, step, commit = run.logged[0]
    assert step == 30_000
    assert commit is True
    assert "reconstruction/psnr" in payload
    assert "reconstruction/FRONT/ssim" in payload
    assert "reconstruction/FRONT/gt_vs_render" in payload
    assert not any(key.startswith("novel_view/") for key in payload)

    with pytest.raises(ValueError, match="exactly index"):
        waymo_training_cli.evaluate_waymo_split(
            _EvaluationRenderer(),
            manifest,
            device="cpu",
            output_directory=tmp_path,
            step=30_000,
            split_name="reconstruction",
            training_rows=(1, 0),
        )


def test_streetgs_actor_point_policy_preserves_dense_lidar_and_is_deterministic(
) -> None:
    cloud_type = waymo_training_cli.ColoredPointCloud
    clouds_type = waymo_training_cli.CanonicalScenePointClouds
    background = cloud_type(torch.zeros(1, 3), torch.zeros(1, 3))
    sparse = cloud_type(torch.rand(2, 3), torch.rand(2, 3))
    retained = cloud_type(torch.rand(5, 3), torch.rand(5, 3))
    oversized = cloud_type(torch.arange(24).reshape(8, 3).float(), torch.rand(8, 3))
    tracks = tuple(
        SimpleNamespace(
            actor_id=actor_id,
            class_name="vehicle",
            dimensions_lwh=torch.tensor([4.0, 2.0, 2.0]),
        )
        for actor_id in (1, 2, 3, 4)
    )
    manifest = SimpleNamespace(actor_tracks=tracks)
    source = clouds_type(
        background=background,
        actors={2: sparse, 3: retained, 4: oversized},
    )

    first, records = waymo_training_cli.apply_streetgs_actor_point_policy(
        source,
        manifest,
        actor_box_scale=1.0,
        min_points=4,
        grid_resolution=3,
        seed=0,
    )
    second, second_records = waymo_training_cli.apply_streetgs_actor_point_policy(
        source,
        manifest,
        actor_box_scale=1.0,
        min_points=4,
        grid_resolution=3,
        seed=0,
    )

    assert waymo_training_cli.STREETGS_ACTOR_MIN_POINTS == 2_000
    assert waymo_training_cli.STREETGS_ACTOR_MAX_POINTS is None
    assert waymo_training_cli.STREETGS_ACTOR_GRID_RESOLUTION**3 == 8_000
    assert set(first.actors) == {1, 2, 3, 4}
    assert records["1"]["strategy"] == "bbox_grid_fallback_missing"
    assert records["2"]["strategy"] == "bbox_grid_fallback_sparse"
    assert records["3"]["strategy"] == "lidar_uncapped"
    assert records["4"]["strategy"] == "lidar_uncapped"
    assert first.actors[1].points.shape[0] == 27
    assert first.actors[2].points.shape[0] == 27
    assert first.actors[3] is retained
    assert first.actors[4] is oversized
    assert first.actors[4].points.shape[0] == 8
    assert records["1"]["generated_fallback_point_count"] == 27
    assert records["2"]["discarded_lidar_point_count"] == 2
    assert records["3"]["used_lidar_point_count"] == 5
    assert records["3"]["discarded_lidar_point_count"] == 0
    assert records["4"]["source_point_count"] == 8
    assert records["4"]["final_point_count"] == 8
    assert records["4"]["used_lidar_point_count"] == 8
    assert records["4"]["discarded_lidar_point_count"] == 0
    torch.testing.assert_close(first.actors[1].points.amin(0), torch.tensor([-2.0, -1.0, -1.0]))
    torch.testing.assert_close(first.actors[1].points.amax(0), torch.tensor([2.0, 1.0, 1.0]))
    for actor_id in first.actors:
        torch.testing.assert_close(first.actors[actor_id].points, second.actors[actor_id].points)
        torch.testing.assert_close(first.actors[actor_id].colors, second.actors[actor_id].colors)
    assert records == second_records

    capped, capped_records = (
        waymo_training_cli.apply_streetgs_actor_point_policy(
            source,
            manifest,
            actor_box_scale=1.0,
            min_points=4,
            max_points=6,
            grid_resolution=3,
            seed=0,
        )
    )
    assert capped.actors[4].points.shape[0] == 6
    assert capped_records["4"]["strategy"] == "seeded_cap_non_reference"
    assert capped_records["4"]["used_lidar_point_count"] == 6
    assert capped_records["4"]["discarded_lidar_point_count"] == 2


def test_actor_fallback_box_scale_is_planar_and_never_changes_height() -> None:
    background = waymo_training_cli.ColoredPointCloud(
        torch.zeros(1, 3), torch.zeros(1, 3)
    )
    manifest = SimpleNamespace(
        actor_tracks=(
            SimpleNamespace(
                actor_id=7,
                class_name="vehicle",
                dimensions_lwh=torch.tensor([4.0, 2.0, 2.0]),
            ),
        )
    )
    result, records = waymo_training_cli.apply_streetgs_actor_point_policy(
        waymo_training_cli.CanonicalScenePointClouds(
            background=background,
            actors={},
        ),
        manifest,
        actor_box_scale=2.0,
        min_points=4,
        grid_resolution=3,
    )

    torch.testing.assert_close(
        result.actors[7].points.amin(0), torch.tensor([-4.0, -2.0, -1.0])
    )
    torch.testing.assert_close(
        result.actors[7].points.amax(0), torch.tensor([4.0, 2.0, 1.0])
    )
    assert records["7"]["effective_dimensions_lwh"] == [8.0, 4.0, 2.0]


def test_initial_scale_diagnostics_report_quantiles_counts_and_reject_nonfinite(
) -> None:
    def gaussian(scales: list[float]) -> Any:
        expanded = torch.tensor(scales)[:, None].expand(-1, 3)
        return SimpleNamespace(log_scales=expanded.log())

    scene = SimpleNamespace(
        background=gaussian([1.0, 2.0, 3.0, 4.0]),
        actors=[SimpleNamespace(actor_id=9, gaussians=gaussian([0.5, 1.5]))],
    )
    diagnostics = waymo_training_cli.gaussian_initial_scale_diagnostics(scene)

    assert diagnostics["background"]["gaussian_count"] == 4
    assert diagnostics["background"]["q50_m"] == pytest.approx(2.5)
    assert diagnostics["background"]["max_m"] == pytest.approx(4.0)
    assert diagnostics["actors"]["actor_model_count"] == 1
    assert diagnostics["actors"]["gaussian_count"] == 2
    assert diagnostics["actors"]["per_actor"]["9"]["max_m"] == pytest.approx(1.5)
    assert diagnostics["composite"]["gaussian_count"] == 6

    scene.background.log_scales[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        waymo_training_cli.gaussian_initial_scale_diagnostics(scene)


def test_training_launcher_independently_uses_official_scale_and_castrack(
    tmp_path: Path,
) -> None:
    sequence = "12374656037744638388_1412_711_1432_711"
    environment = _official_launcher_environment(
        tmp_path,
        sequence=sequence,
        start_frame=0,
        end_frame=100,
    )
    launcher = SCRIPT_PATH.parent / "train_armgs_waymo.sh"

    completed = subprocess.run(
        ["bash", str(launcher), sequence, "0", "100"],
        cwd=SCRIPT_PATH.parents[1],
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "actor tracking/planar box scale:" in completed.stdout
    assert "--actor-box-scale 2.0" in completed.stdout
    assert "--castrack-path" in completed.stdout
    assert f"tracking/castrack/{sequence}.json" in completed.stdout

    mismatched = dict(environment, ACTOR_BOX_SCALE="1.0")
    rejected = subprocess.run(
        ["bash", str(launcher), sequence, "0", "100"],
        cwd=SCRIPT_PATH.parents[1],
        env=mismatched,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert rejected.returncode != 0
    assert "requires ACTOR_BOX_SCALE=2.0" in rejected.stderr


def test_official_pipeline_passes_scale_castrack_and_reuses_legacy_colmap(
    tmp_path: Path,
) -> None:
    repository = SCRIPT_PATH.parents[1]
    launcher = repository / "scripts" / "prepare_waymo_streetgs_scene.sh"
    scene_026 = "12374656037744638388_1412_711_1432_711"
    environment = _official_launcher_environment(
        tmp_path / "scene026",
        sequence=scene_026,
        start_frame=0,
        end_frame=100,
    )
    environment.update(
        {
            "GSAM_PYTHON": "/bin/true",
            "COLMAP_BINARY": "/bin/true",
            "RUN_SKY": "0",
            "RUN_COLMAP": "1",
            "RUN_TRAIN": "0",
            "REUSE_COLMAP": "0",
        }
    )

    completed = subprocess.run(
        ["bash", str(launcher), "--official", "026"],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "actor tracking/planar box scale:" in completed.stdout
    assert "--actor-box-scale 2.0" in completed.stdout
    assert "--castrack-path" in completed.stdout

    scene_006 = "10448102132863604198_472_000_492_000"
    legacy_environment = _official_launcher_environment(
        tmp_path / "scene006",
        sequence=scene_006,
        start_frame=0,
        end_frame=85,
    )
    legacy_environment.update(
        {
            "GSAM_PYTHON": "/bin/true",
            "COLMAP_BINARY": "/bin/true",
            "RUN_SKY": "0",
            "RUN_COLMAP": "1",
            "RUN_TRAIN": "0",
            "REUSE_COLMAP": "1",
        }
    )
    reused = subprocess.run(
        ["bash", str(launcher), "--official", "006"],
        cwd=repository,
        env=legacy_environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert reused.returncode == 0, reused.stderr
    assert "reuse complete COLMAP output" in reused.stdout


def test_paper_launcher_requires_nonempty_castrack(tmp_path: Path) -> None:
    sequence = "10448102132863604198_472_000_492_000"
    environment = _official_launcher_environment(
        tmp_path,
        sequence=sequence,
        start_frame=0,
        end_frame=85,
        with_castrack=False,
    )
    launcher = SCRIPT_PATH.parent / "train_armgs_waymo.sh"

    completed = subprocess.run(
        ["bash", str(launcher), sequence, "0", "85"],
        cwd=SCRIPT_PATH.parents[1],
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode != 0
    assert "requires non-empty CAStrack JSON" in completed.stderr
