from __future__ import annotations

import argparse
import hashlib
import json
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "train_armgs_nuscenes.py"
)
MODULE_NAME = "_armgs_nuscenes_train_cli_for_tests"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
nuscenes_training_cli = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = nuscenes_training_cli
SPEC.loader.exec_module(nuscenes_training_cli)


def test_parser_accepts_scene_camera_runtime_wandb_and_resume_options(
    tmp_path: Path,
) -> None:
    args = nuscenes_training_cli.parse_args(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--nuscenes-root",
            str(tmp_path / "nuscenes"),
            "--sky-mask-root",
            str(tmp_path / "sky_masks"),
            "--sky-mask-reject-list",
            str(tmp_path / "reject_tokens.txt"),
            "--colmap-points3d",
            str(tmp_path / "points3D.txt"),
            "--scene",
            "61",
            "--version",
            "v1.0-trainval",
            "--cameras",
            "cam_front, CAM_BACK",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "output"),
            "--resume",
            str(tmp_path / "checkpoint.pt"),
            "--iterations",
            "17",
            "--checkpoint-interval",
            "5",
            "--log-interval",
            "2",
            "--image-log-interval",
            "13",
            "--eval-interval",
            "7",
            "--no-eval-at-end",
            "--eval-lpips",
            "--eval-lpips-net",
            "vgg",
            "--eval-only",
            "--wandb",
            "--wandb-entity",
            "CamoSplat",
            "--wandb-project",
            "ArmGS-nuScenes",
            "--wandb-run-name",
            "scene-0061-test",
            "--wandb-run-id",
            "resume-run-123",
            "--wandb-mode",
            "offline",
            "--wandb-dir",
            str(tmp_path / "wandb"),
            "--wandb-fail-fast",
            "--wandb-log-checkpoint-artifact",
        ]
    )

    assert args.scene == "61"
    assert args.sky_mask_root == tmp_path / "sky_masks"
    assert args.sky_mask_reject_list == tmp_path / "reject_tokens.txt"
    assert args.colmap_points3d == tmp_path / "points3D.txt"
    assert args.cameras == ("CAM_FRONT", "CAM_BACK")
    assert args.resume == tmp_path / "checkpoint.pt"
    assert args.iterations == 17
    assert args.checkpoint_interval == 5
    assert args.log_interval == 2
    assert args.image_log_interval == 13
    assert args.wandb is True
    assert args.eval_interval == 7
    assert args.eval_at_end is False
    assert args.eval_lpips is True
    assert args.eval_lpips_net == "vgg"
    assert args.eval_only is True
    assert args.wandb_entity == "CamoSplat"
    assert args.wandb_project == "ArmGS-nuScenes"
    assert args.wandb_run_name == "scene-0061-test"
    assert args.wandb_run_id == "resume-run-123"
    assert args.wandb_mode == "offline"
    assert args.wandb_dir == tmp_path / "wandb"
    assert args.wandb_fail_fast is True
    assert args.wandb_log_checkpoint_artifact is True

def test_parser_evaluation_defaults_and_eval_only_guard(tmp_path: Path) -> None:
    base = [
        "--nuscenes-root",
        str(tmp_path / "nuscenes"),
        "--output-dir",
        str(tmp_path / "output"),
    ]
    args = nuscenes_training_cli.parse_args(base)

    assert args.eval_interval == 1000
    assert args.eval_at_end is True
    assert args.eval_lpips is False
    assert args.eval_lpips_net == "alex"
    assert args.eval_only is False
    assert args.image_log_interval == 500
    assert args.wandb_run_id is None
    assert args.wandb_fail_fast is False
    assert args.wandb_log_checkpoint_artifact is False

    disabled = nuscenes_training_cli.parse_args(
        [
            *base,
            "--eval-interval",
            "0",
            "--image-log-interval",
            "0",
            "--no-eval-at-end",
        ]
    )
    assert disabled.eval_interval == 0
    assert disabled.eval_at_end is False
    assert disabled.image_log_interval == 0

    with pytest.raises(SystemExit):
        nuscenes_training_cli.parse_args([*base, "--eval-only"])
    with pytest.raises(SystemExit):
        nuscenes_training_cli.parse_args([*base, "--eval-interval", "-1"])
    with pytest.raises(SystemExit):
        nuscenes_training_cli.parse_args([*base, "--image-log-interval", "-1"])




def test_camera_parser_supports_all_and_rejects_invalid_lists() -> None:
    assert nuscenes_training_cli.parse_camera_channels("all") == tuple(
        nuscenes_training_cli.NUSCENES_CAMERA_CHANNELS
    )
    with pytest.raises(argparse.ArgumentTypeError, match="unknown"):
        nuscenes_training_cli.parse_camera_channels("CAM_UNKNOWN")
    with pytest.raises(argparse.ArgumentTypeError, match="unique"):
        nuscenes_training_cli.parse_camera_channels("CAM_FRONT,CAM_FRONT")
    with pytest.raises(argparse.ArgumentTypeError, match="comma-separated"):
        nuscenes_training_cli.parse_camera_channels("CAM_FRONT,")


def test_nuscenes_dataset_identity_covers_metadata_image_lidar_and_sky_mask(
    tmp_path: Path,
) -> None:
    root = tmp_path / "nuscenes"
    metadata_directory = root / "v1.0-trainval"
    metadata_directory.mkdir(parents=True)
    metadata = metadata_directory / "sample.json"
    metadata.write_text('{"value": 1}', encoding="utf-8")
    image = root / "sample.jpg"
    image.write_bytes(b"jpeg")
    lidar = root / "scan.bin"
    lidar.write_bytes(b"lidar")
    sky_mask = root / "sky.png"
    sky_mask.write_bytes(b"sky-mask")
    frame = SimpleNamespace(
        image_path=image,
        lidar=SimpleNamespace(source_path=lidar),
        sky_mask_path=sky_mask,
        actor_mask_path=None,
    )

    before = nuscenes_training_cli.build_nuscenes_dataset_input_identity(
        [frame],
        root=root,
        version="v1.0-trainval",
    )
    metadata.write_text('{"value": 2}', encoding="utf-8")
    after = nuscenes_training_cli.build_nuscenes_dataset_input_identity(
        [frame],
        root=root,
        version="v1.0-trainval",
    )
    sky_mask.write_bytes(b"changed-sky-mask")
    after_mask_change = (
        nuscenes_training_cli.build_nuscenes_dataset_input_identity(
            [frame],
            root=root,
            version="v1.0-trainval",
        )
    )

    assert before["file_count"] == 4
    assert before["metadata_file_count"] == 1
    assert before["digest_sha256"] != after["digest_sha256"]
    assert after["digest_sha256"] != after_mask_change["digest_sha256"]


def test_nuscenes_dataset_identity_covers_reject_file_and_count(
    tmp_path: Path,
) -> None:
    root = tmp_path / "nuscenes"
    metadata_directory = root / "v1.0-trainval"
    metadata_directory.mkdir(parents=True)
    (metadata_directory / "sample.json").write_text("[]", encoding="utf-8")
    image = root / "sample.jpg"
    image.write_bytes(b"jpeg")
    reject_list = tmp_path / "reject.txt"
    reject_list.write_text("a" * 32 + "\n", encoding="utf-8")
    frame = SimpleNamespace(
        image_path=image,
        lidar=None,
        sky_mask_path=None,
        actor_mask_path=None,
    )

    before = nuscenes_training_cli.build_nuscenes_dataset_input_identity(
        [frame],
        root=root,
        version="v1.0-trainval",
        sky_mask_reject_list=reject_list,
        sky_mask_rejected_count=1,
    )
    reject_list.write_text("b" * 32 + "\n", encoding="utf-8")
    after = nuscenes_training_cli.build_nuscenes_dataset_input_identity(
        [frame],
        root=root,
        version="v1.0-trainval",
        sky_mask_reject_list=reject_list,
        sky_mask_rejected_count=1,
    )

    assert before["version"] == 3
    assert before["file_count"] == 3
    assert before["sky_mask_reject_list"] == str(reject_list.resolve())
    assert before["sky_mask_rejected_count"] == 1
    assert before["digest_sha256"] != after["digest_sha256"]


def test_nuscenes_dataset_identity_covers_colmap_points(tmp_path: Path) -> None:
    root = tmp_path / "nuscenes"
    metadata_directory = root / "v1.0-trainval"
    metadata_directory.mkdir(parents=True)
    (metadata_directory / "sample.json").write_text("[]", encoding="utf-8")
    image = root / "sample.jpg"
    image.write_bytes(b"jpeg")
    points = tmp_path / "points3D.txt"
    points.write_text("1 0 0 0 255 255 255 0\n", encoding="utf-8")
    frame = SimpleNamespace(
        image_path=image,
        lidar=None,
        sky_mask_path=None,
        actor_mask_path=None,
    )

    before = nuscenes_training_cli.build_nuscenes_dataset_input_identity(
        [frame],
        root=root,
        version="v1.0-trainval",
        colmap_points3d=points,
    )
    points.write_text("1 1 0 0 255 255 255 0\n", encoding="utf-8")
    after = nuscenes_training_cli.build_nuscenes_dataset_input_identity(
        [frame],
        root=root,
        version="v1.0-trainval",
        colmap_points3d=points,
    )

    assert before["colmap_points3d"] == str(points.resolve())
    assert before["digest_sha256"] != after["digest_sha256"]


class _FakeWandbRun:
    def __init__(self) -> None:
        self.id = "resume-run-123"
        self.entity = "CamoSplat"
        self.project = "ArmGS-nuScenes"
        self.name = "scene-0061-test"
        self.url = "https://wandb.invalid/resume-run-123"
        self.summary: dict[str, Any] = {}
        self.logged: list[tuple[dict[str, Any], int]] = []
        self.finish_exit_codes: list[int] = []
        self.artifacts: list[Any] = []

    def log(self, payload: dict[str, Any], *, step: int) -> None:
        self.logged.append((payload, step))

    def finish(self, *, exit_code: int) -> None:
        self.finish_exit_codes.append(exit_code)

    def log_artifact(self, artifact: Any) -> None:
        self.artifacts.append(artifact)


def test_wandb_initialization_log_and_finish_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    fake_run = _FakeWandbRun()

    def fake_init(**kwargs: Any) -> _FakeWandbRun:
        captured.update(kwargs)
        return fake_run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))
    args = SimpleNamespace(
        wandb=True,
        wandb_dir=None,
        output_dir=tmp_path / "output",
        wandb_entity="CamoSplat",
        wandb_project="ArmGS-nuScenes",
        wandb_run_name="scene-0061-test",
        wandb_run_id="resume-run-123",
        wandb_mode="offline",
        log_interval=100,
        image_log_interval=500,
        wandb_fail_fast=False,
        wandb_log_checkpoint_artifact=False,
    )
    run_metadata = {
        "nuscenes_root": str(tmp_path / "nuscenes"),
        "version": "v1.0-trainval",
        "scene": "scene-0061",
        "sky_mask_root": str(tmp_path / "sky_masks"),
        "sky_mask_count": 234,
        "sky_mask_reject_list": str(tmp_path / "reject.txt"),
        "sky_mask_rejected_count": 18,
        "camera_channels": ["CAM_FRONT"],
        "train_source_indices": [0, 1],
        "eval_source_indices": [2],
        "dataset_input_identity": {"digest_sha256": "dataset-sha256"},
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "wandb_run.json").write_text(
        json.dumps({"run_id": "stale-sidecar-id"}), encoding="utf-8"
    )

    initialized = nuscenes_training_cli._initialize_wandb(
        args,
        config={"optimization": {"iterations": 3}},
        run_metadata=run_metadata,
    )
    assert initialized is fake_run
    assert captured["entity"] == "CamoSplat"
    assert captured["project"] == "ArmGS-nuScenes"
    assert captured["name"] == "scene-0061-test"
    assert captured["id"] == "resume-run-123"
    assert captured["resume"] == "allow"
    assert captured["mode"] == "offline"
    assert captured["dir"] == str(tmp_path / "output" / "wandb")
    assert captured["config"]["dataset"]["scene"] == "scene-0061"
    assert captured["config"]["dataset"]["sky_mask_root"] == str(
        tmp_path / "sky_masks"
    )
    assert captured["config"]["dataset"]["sky_mask_count"] == 234
    assert captured["config"]["dataset"]["sky_mask_reject_list"] == str(
        tmp_path / "reject.txt"
    )
    assert captured["config"]["dataset"]["sky_mask_rejected_count"] == 18
    assert captured["config"]["dataset"]["input_identity_sha256"] == (
        "dataset-sha256"
    )
    assert captured["config"]["evaluation"] == {
        "interval": 1000,
        "at_end": True,
        "lpips": False,
        "lpips_net": "alex",
        "eval_only": False,
        "metric_protocols": {
            "psnr": "mean-per-image-rgb-mse-data-range-1",
            "ssim": "3dgs-gaussian-11x11-sigma-1.5-data-range-1",
            "lpips": None,
            "actor_mask": "streetgs-projected-cuboid-silhouette-union",
        },
    }
    assert captured["config"]["logging"] == {
        "scalar_interval": 100,
        "image_interval": 500,
        "fail_fast": False,
        "checkpoint_artifact": False,
    }
    assert json.loads(
        (tmp_path / "output" / "wandb_run.json").read_text(encoding="utf-8")
    ) == {
        "entity": "CamoSplat",
        "format_version": 1,
        "mode": "offline",
        "name": "scene-0061-test",
        "project": "ArmGS-nuScenes",
        "resume_requested": True,
        "resume_source": "explicit",
        "run_id": "resume-run-123",
        "url": "https://wandb.invalid/resume-run-123",
    }

    nuscenes_training_cli._log_to_wandb(
        fake_run,
        {"step": 3, "loss": 1.25, "gaussians": 42},
    )
    nuscenes_training_cli._finish_wandb(fake_run, exit_code=1)

    assert fake_run.logged == [
        ({"train/loss": 1.25, "train/gaussians": 42}, 3)
    ]
    assert fake_run.finish_exit_codes == [1]


def test_wandb_initialization_recovers_run_id_from_output_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    fake_run = _FakeWandbRun()
    fake_run.id = "sidecar-run-456"

    def fake_init(**kwargs: Any) -> _FakeWandbRun:
        captured.update(kwargs)
        return fake_run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))
    output = tmp_path / "output"
    output.mkdir()
    (output / "wandb_run.json").write_text(
        json.dumps({"format_version": 1, "run_id": "sidecar-run-456"}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        wandb=True,
        wandb_dir=None,
        output_dir=output,
        wandb_entity="CamoSplat",
        wandb_project="ArmGS-nuScenes",
        wandb_run_name=None,
        wandb_run_id=None,
        wandb_mode="offline",
        wandb_fail_fast=False,
    )
    run_metadata = {
        "nuscenes_root": str(tmp_path / "nuscenes"),
        "version": "v1.0-trainval",
        "scene": "scene-0061",
        "camera_channels": ["CAM_FRONT"],
        "train_source_indices": [0],
        "eval_source_indices": [1],
    }

    assert (
        nuscenes_training_cli._initialize_wandb(
            args, config={}, run_metadata=run_metadata
        )
        is fake_run
    )
    assert captured["id"] == "sidecar-run-456"
    assert captured["resume"] == "allow"
    rewritten = json.loads(
        (output / "wandb_run.json").read_text(encoding="utf-8")
    )
    assert rewritten["run_id"] == "sidecar-run-456"
    assert rewritten["resume_requested"] is True
    assert rewritten["resume_source"] == "sidecar"


def test_wandb_gt_render_payload_places_gt_left_and_render_right(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_image(data: Any, *, caption: str) -> str:
        captured["data"] = data
        captured["caption"] = caption
        return "comparison-image"

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Image=fake_image))
    target = torch.tensor(
        [[[[0.0, 0.5, 1.0], [1.0, 0.0, 0.25]]]],
        dtype=torch.float32,
    )
    rendered = torch.tensor(
        [[[[0.25, 0.0, 1.0], [0.0, 1.0, 0.5]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    batch = SimpleNamespace(
        target_rgb=target,
        view=SimpleNamespace(
            camera_id=torch.tensor(0),
            timestamp=torch.tensor(1_234_567_890, dtype=torch.int64),
            training_row=torch.tensor(7),
        ),
    )
    output = SimpleNamespace(
        rendering=SimpleNamespace(rgb=rendered),
        step=500,
    )

    payload = nuscenes_training_cli._wandb_image_payload_factory(
        batch,
        output,
        training_manifest=[
            SimpleNamespace(frame_index=index + 20) for index in range(8)
        ],
        training_source_indices=list(range(100, 108)),
    )

    assert payload == {
        "train/gt_vs_render": "comparison-image",
        "train/image_camera_id": 0,
        "train/image_camera": "CAM_FRONT",
        "train/image_timestamp_ns": "1234567890",
        "train/image_training_row": 7,
        "train/image_frame_index": 27,
        "train/image_source_index": 107,
    }
    assert captured["caption"] == (
        "Step: 500 | Camera: CAM_FRONT (0) | Training row: 7 | "
        "Timestamp ns: 1234567890 | Frame: 27 | Source row: 107 | "
        "Left: GT | Right: Render"
    )
    comparison = captured["data"]
    assert comparison.shape == (1, 4, 3)
    assert str(comparison.dtype) == "uint8"
    assert comparison[:, :2].tolist() == [
        [[0, 128, 255], [255, 0, 64]]
    ]
    assert comparison[:, 2:].tolist() == [
        [[64, 0, 255], [0, 255, 128]]
    ]
    assert rendered.grad_fn is None


def test_wandb_logging_fails_soft_by_default_and_can_fail_fast(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingRun:
        def log(self, payload: Any, *, step: int) -> None:
            raise ConnectionError("offline")

    assert (
        nuscenes_training_cli._log_to_wandb(
            FailingRun(), {"step": 10, "train/loss": 1.0}
        )
        is False
    )
    assert "continuing without aborting training" in capsys.readouterr().err
    with pytest.raises(RuntimeError, match="metric/image logging failed"):
        nuscenes_training_cli._log_to_wandb(
            FailingRun(),
            {"step": 10, "train/loss": 1.0},
            fail_fast=True,
        )


def test_wandb_checkpoint_artifact_has_checksum_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "final.pt"
    checkpoint.write_bytes(b"armgs-checkpoint")
    captured: dict[str, Any] = {}

    class FakeArtifact:
        def __init__(self, *, name: str, type: str, metadata: Any) -> None:
            captured.update(name=name, type=type, metadata=metadata)

        def add_file(self, path: str, *, name: str) -> None:
            captured.update(file_path=path, file_name=name)

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Artifact=FakeArtifact))
    run = _FakeWandbRun()
    metadata = nuscenes_training_cli._log_checkpoint_artifact(
        run,
        checkpoint,
        metadata={"scene": "scene-0061", "step": 30_000},
    )

    assert metadata == {
        "scene": "scene-0061",
        "step": 30_000,
        "sha256": hashlib.sha256(b"armgs-checkpoint").hexdigest(),
        "size_bytes": len(b"armgs-checkpoint"),
    }
    assert captured["name"] == (
        "armgs-scene-0061-resume-run-123-checkpoint"
    )
    assert captured["type"] == "model"
    assert captured["metadata"] == metadata
    assert captured["file_path"] == str(checkpoint.resolve())
    assert captured["file_name"] == "final.pt"
    assert len(run.artifacts) == 1


def test_wandb_disabled_does_not_import_or_initialize(tmp_path: Path) -> None:
    args = SimpleNamespace(wandb=False)
    assert (
        nuscenes_training_cli._initialize_wandb(
            args,
            config={},
            run_metadata={},
        )
        is None
    )


def test_scene_0061_launcher_wires_default_reject_list_and_empty_disable() -> None:
    launcher = SCRIPT_PATH.with_name("train_nuscenes_scene_0061.sh")
    source = launcher.read_text(encoding="utf-8")
    reject_list = SCRIPT_PATH.parents[1] / "configs" / (
        "nuscenes_scene_0061_sky_mask_reject_tokens.txt"
    )

    tokens = nuscenes_training_cli.parse_nuscenes_sky_mask_reject_list(
        reject_list
    )
    assert len(tokens) == 18
    assert "${SKY_MASK_REJECT_LIST-" in source
    assert "--sky-mask-reject-list" in source
    assert "SKY_MASK_REJECT_LIST_PATH:-disabled" in source
    assert 'EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"' in source
    assert 'EVAL_AT_END="${EVAL_AT_END:-1}"' in source
    assert 'EVAL_LPIPS="${EVAL_LPIPS:-1}"' in source
    assert 'EVAL_ONLY="${EVAL_ONLY:-0}"' in source
    assert '--eval-interval "${EVAL_INTERVAL}"' in source
    assert "TRAIN_ARGS+=(--eval-at-end)" in source
    assert "TRAIN_ARGS+=(--no-eval-at-end)" in source
    assert "TRAIN_ARGS+=(--eval-lpips)" in source

class _EvaluationRenderer(torch.nn.Module):
    def __init__(self, predictions: dict[int, torch.Tensor]) -> None:
        super().__init__()
        self.predictions = predictions
        self.forward_states: list[tuple[bool, bool]] = []

    def forward(self, view: Any) -> Any:
        self.forward_states.append(
            (self.training, torch.is_inference_mode_enabled())
        )
        return SimpleNamespace(rgb=self.predictions[view.frame_index])


class _EvaluationManifest(list[Any]):
    def __init__(self, frames: list[Any]) -> None:
        super().__init__(frames)
        self.actor_tracks = ("held-out-track",)


def test_held_out_evaluation_writes_weighted_metrics_previews_and_wandb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = [
        SimpleNamespace(camera_id=0, frame_index=0),
        SimpleNamespace(camera_id=0, frame_index=1),
        SimpleNamespace(camera_id=3, frame_index=2),
    ]
    manifest = _EvaluationManifest(frames)
    targets = {
        index: torch.zeros((1, 12, 12, 3), dtype=torch.float32)
        for index in range(3)
    }
    predictions = {
        0: torch.full((1, 12, 12, 3), 0.1),
        1: torch.full((1, 12, 12, 3), 0.2),
        2: torch.full((1, 12, 12, 3), 1.4),
    }
    renderer = _EvaluationRenderer(predictions)
    renderer.train()
    training_rows: list[Any] = []

    def fake_batch(frame: Any, training_row: Any, *, device: Any) -> Any:
        training_rows.append(training_row)
        return SimpleNamespace(
            view=SimpleNamespace(frame_index=frame.frame_index),
            target_rgb=targets[frame.frame_index],
        )

    projection_calls: list[tuple[int, Any, float]] = []

    def fake_actor_projection(
        frame: Any,
        actor_tracks: Any,
        *,
        box_scale: float,
    ) -> torch.Tensor:
        projection_calls.append((frame.frame_index, actor_tracks, box_scale))
        mask = torch.ones((12, 12), dtype=torch.bool)
        if frame.frame_index == 2:
            mask.zero_()
        return mask

    monkeypatch.setattr(
        nuscenes_training_cli,
        "canonical_frame_to_training_batch",
        fake_batch,
    )
    monkeypatch.setattr(
        nuscenes_training_cli,
        "project_actor_boxes_to_mask",
        fake_actor_projection,
    )
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            Image=lambda data, *, caption: {
                "shape": data.shape,
                "caption": caption,
            }
        ),
    )
    wandb_run = _FakeWandbRun()
    policy = {
        "interval": 7,
        "at_end": True,
        "lpips": False,
        "lpips_net": "alex",
        "eval_only": False,
    }

    record = nuscenes_training_cli.evaluate_nuscenes_split(
        renderer,
        manifest,
        device="cpu",
        output_directory=tmp_path,
        step=17,
        actor_box_scale=1.25,
        wandb_run=wandb_run,
        evaluation_policy=policy,
    )

    assert training_rows == [None, None, None]
    assert renderer.training is True
    assert renderer.forward_states == [(False, True)] * 3
    assert projection_calls == [
        (0, manifest.actor_tracks, 1.25),
        (1, manifest.actor_tracks, 1.25),
        (2, manifest.actor_tracks, 1.25),
    ]
    assert record["per_camera"]["CAM_FRONT"]["num_images"] == 2
    assert record["per_camera"]["CAM_BACK"]["num_images"] == 1
    assert record["aggregate"]["num_images"] == 3
    assert record["aggregate"]["num_actor_images"] == 2
    expected_psnr = (
        record["per_camera"]["CAM_FRONT"]["psnr"] * 2
        + record["per_camera"]["CAM_BACK"]["psnr"]
    ) / 3
    assert record["aggregate"]["psnr"] == pytest.approx(expected_psnr)
    assert record["per_camera"]["CAM_BACK"]["psnr"] == pytest.approx(0.0)
    assert record["aggregate"]["actor_psnr"] == pytest.approx(
        record["per_camera"]["CAM_FRONT"]["actor_psnr"]
    )
    assert record["policy"] == policy

    json_path = tmp_path / "evaluation" / "step_00000017.json"
    assert json.loads(json_path.read_text(encoding="utf-8")) == record
    for camera_name in ("CAM_FRONT", "CAM_BACK"):
        preview = (
            tmp_path
            / "evaluation"
            / f"step_00000017_{camera_name}_gt_render.png"
        )
        assert preview.is_file()
    assert list((tmp_path / "evaluation").glob(".*.tmp")) == []

    assert len(wandb_run.logged) == 1
    payload, logged_step = wandb_run.logged[0]
    assert logged_step == 17
    assert payload["eval/psnr"] == pytest.approx(record["aggregate"]["psnr"])
    assert "eval/CAM_FRONT/ssim" in payload
    assert "eval/CAM_BACK/gt_vs_render" in payload


def test_held_out_evaluation_restores_renderer_mode_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = SimpleNamespace(camera_id=0, frame_index=0)
    manifest = _EvaluationManifest([frame])

    class FailingRenderer(torch.nn.Module):
        def forward(self, view: Any) -> Any:
            assert self.training is False
            assert torch.is_inference_mode_enabled()
            raise RuntimeError("render failed")

    renderer = FailingRenderer()
    renderer.train()
    monkeypatch.setattr(
        nuscenes_training_cli,
        "canonical_frame_to_training_batch",
        lambda frame, training_row, device: SimpleNamespace(
            view=SimpleNamespace(),
            target_rgb=torch.zeros((1, 12, 12, 3)),
        ),
    )

    with pytest.raises(RuntimeError, match="render failed"):
        nuscenes_training_cli.evaluate_nuscenes_split(
            renderer,
            manifest,
            device="cpu",
            output_directory=tmp_path,
            step=3,
        )

    assert renderer.training is True


def test_resumed_total_step_still_gets_one_final_evaluation() -> None:
    trainer = SimpleNamespace(step=30_000, sampler=object())
    events: list[tuple[str, int]] = []

    last_step = nuscenes_training_cli._execute_training_and_evaluation(
        trainer,
        ["unused-frame"],
        total_iterations=30_000,
        device="cpu",
        checkpoint_interval=1000,
        log_interval=100,
        checkpoint_callback=lambda step: events.append(("checkpoint", step)),
        log_callback=None,
        log_payload_factory=None,
        evaluation_interval=0,
        evaluate_at_end=True,
        eval_only=False,
        evaluation_callback=lambda step: events.append(("evaluation", step)),
        after_training_callback=lambda: events.append(("final", trainer.step)),
    )

    assert last_step == 30_000
    assert events == [("final", 30_000), ("evaluation", 30_000)]


def test_eval_only_skips_training_and_checkpoint_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = SimpleNamespace(step=30_000)
    evaluations: list[int] = []

    def unexpected_training(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("eval-only called train_until")

    monkeypatch.setattr(
        nuscenes_training_cli,
        "train_until",
        unexpected_training,
    )

    last_step = nuscenes_training_cli._execute_training_and_evaluation(
        trainer,
        ["unused-frame"],
        total_iterations=30_000,
        device="cpu",
        checkpoint_interval=1000,
        log_interval=100,
        checkpoint_callback=lambda step: pytest.fail("checkpoint written"),
        log_callback=None,
        log_payload_factory=None,
        evaluation_interval=1000,
        evaluate_at_end=False,
        eval_only=True,
        evaluation_callback=evaluations.append,
        after_training_callback=lambda: pytest.fail("final checkpoint written"),
    )

    assert last_step == 30_000
    assert evaluations == [30_000]


def test_periodic_final_step_is_not_evaluated_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = SimpleNamespace(step=999)
    evaluations: list[int] = []

    def fake_train_until(*args: Any, completed_step_callback: Any, **kwargs: Any) -> None:
        trainer.step = 1000
        completed_step_callback(trainer.step)

    monkeypatch.setattr(
        nuscenes_training_cli,
        "train_until",
        fake_train_until,
    )
    nuscenes_training_cli._execute_training_and_evaluation(
        trainer,
        ["unused-frame"],
        total_iterations=1000,
        device="cpu",
        checkpoint_interval=1000,
        log_interval=100,
        checkpoint_callback=lambda step: None,
        log_callback=None,
        log_payload_factory=None,
        evaluation_interval=1000,
        evaluate_at_end=True,
        eval_only=False,
        evaluation_callback=evaluations.append,
        after_training_callback=lambda: None,
    )

    assert evaluations == [1000]



def test_evaluation_rng_isolation_preserves_training_rng() -> None:
    torch.manual_seed(12345)
    expected = torch.rand(4)

    torch.manual_seed(12345)
    nuscenes_training_cli._without_rng_side_effects(
        lambda: torch.rand(100),
        device="cpu",
    )
    actual = torch.rand(4)

    torch.testing.assert_close(actual, expected)
