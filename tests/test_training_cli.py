from __future__ import annotations

import argparse
import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest
import torch

from armgs.losses import ArmGSLoss
from armgs.sampling import StatefulShuffleSampler
from armgs.scene_builder import (
    CanonicalScenePointClouds,
    ColoredPointCloud,
)


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_armgs.py"
MODULE_NAME = "_armgs_train_cli_for_tests"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
training_cli = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = training_cli
SPEC.loader.exec_module(training_cli)


def test_parser_accepts_camera_and_runtime_options(tmp_path: Path) -> None:
    args = training_cli.parse_args(
        [
            "--kitti-root",
            str(tmp_path / "sequence"),
            "--output-dir",
            str(tmp_path / "output"),
            "--camera-ids",
            "2, 3",
            "--tracklets",
            "tracklet_labels.xml",
            "--sky-mask-dir",
            "2=sky_2",
            "--sky-mask-dir",
            "3=sky_3",
            "--actor-mask-dir",
            "2=actors_2",
            "--device",
            "cpu",
            "--iterations",
            "17",
            "--checkpoint-interval",
            "5",
            "--log-interval",
            "2",
        ]
    )

    assert args.camera_ids == (2, 3)
    assert args.tracklets == Path("tracklet_labels.xml")
    assert args.sky_mask_dir == ["2=sky_2", "3=sky_3"]
    assert args.actor_mask_dir == ["2=actors_2"]
    assert args.device == "cpu"
    assert args.iterations == 17
    assert args.checkpoint_interval == 5
    assert args.log_interval == 2


@pytest.mark.parametrize("value", ["", "2,", "a", "-1", "2,2"])
def test_camera_id_parser_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        training_cli.parse_camera_ids(value)


def test_camera_directory_mapping_is_explicit_and_validated() -> None:
    mappings = training_cli.parse_camera_directory_mappings(
        ["2=masks/two", "3=/masks/three"],
        (2, 3),
        option_name="--sky-mask-dir",
    )
    assert mappings == {2: Path("masks/two"), 3: Path("/masks/three")}

    with pytest.raises(ValueError, match="CAMERA_ID=DIR"):
        training_cli.parse_camera_directory_mappings(
            ["masks"], (2,), option_name="--sky-mask-dir"
        )
    with pytest.raises(ValueError, match="unrequested camera"):
        training_cli.parse_camera_directory_mappings(
            ["3=masks"], (2,), option_name="--sky-mask-dir"
        )
    with pytest.raises(ValueError, match="repeats camera"):
        training_cli.parse_camera_directory_mappings(
            ["2=one", "2=two"], (2,), option_name="--sky-mask-dir"
        )


def _cloud(points: list[list[float]]) -> ColoredPointCloud:
    xyz = torch.tensor(points, dtype=torch.float64)
    return ColoredPointCloud(xyz, torch.zeros_like(xyz))


def test_camera_scene_extent_matches_3dgs_normalization_radius() -> None:
    transforms = []
    for x in (0.0, 10.0):
        transform = torch.eye(4, dtype=torch.float64)
        transform[0, 3] = x
        transforms.append(transform)
    manifest = (
        SimpleNamespace(camera_to_world=transform)
        for transform in transforms
    )

    assert training_cli.camera_scene_extent(manifest) == pytest.approx(5.5)


def test_conservative_scene_bounds_include_all_actor_trajectory_extents() -> None:
    point_clouds = CanonicalScenePointClouds(
        background=_cloud([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
        actors={7: _cloud([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])},
    )
    samples = (
        SimpleNamespace(
            quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            translation=torch.tensor([5.0, 0.0, 0.0]),
        ),
        SimpleNamespace(
            quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            translation=torch.tensor([10.0, 0.0, 0.0]),
        ),
    )
    manifest = SimpleNamespace(
        actor_tracks=(SimpleNamespace(actor_id=7, samples=samples),)
    )

    bounds = training_cli.conservative_scene_bounds(
        manifest,
        point_clouds,
        padding_fraction=0.05,
        minimum_padding=0.001,
    )

    assert bounds.aabb_min == pytest.approx((-0.55, -0.55, -0.55))
    assert bounds.aabb_max == pytest.approx((11.55, 2.55, 3.55))
    assert bounds.scene_scale == pytest.approx(
        math.sqrt(12.1**2 + 3.1**2 + 4.1**2)
    )


def test_conservative_scene_bounds_reject_unknown_actor_cloud() -> None:
    point_clouds = CanonicalScenePointClouds(
        background=_cloud([[0.0, 0.0, 0.0]]),
        actors={4: _cloud([[0.0, 0.0, 0.0]])},
    )
    with pytest.raises(ValueError, match="no matching tracks"):
        training_cli.conservative_scene_bounds(
            SimpleNamespace(actor_tracks=()), point_clouds
        )


def _frame(
    *,
    has_depth: bool = True,
    has_sky: bool = True,
    has_actor: bool = True,
) -> SimpleNamespace:
    projection = (
        SimpleNamespace(depths=torch.tensor([1.0])) if has_depth else None
    )
    return SimpleNamespace(
        lidar_projection=projection,
        sky_mask_path=Path("sky.png") if has_sky else None,
        actor_mask_path=Path("actor.png") if has_actor else None,
    )


def _manifest(
    frame: SimpleNamespace, *, has_tracks: bool = True
) -> SimpleNamespace:
    tracks = (SimpleNamespace(actor_id=0),) if has_tracks else ()
    return SimpleNamespace(frames=(frame,), actor_tracks=tracks)


def test_strict_supervision_validation_reports_each_missing_objective() -> None:
    depth_loss = ArmGSLoss(
        require_auxiliary=True,
        lambda_depth=0.1,
        lambda_sky=0.0,
        lambda_foreground=0.0,
    )
    with pytest.raises(ValueError, match="strict depth.*rows \\[0\\]"):
        training_cli.validate_training_supervision(
            _manifest(_frame(has_depth=False)), depth_loss
        )

    sky_loss = ArmGSLoss(
        require_auxiliary=True,
        lambda_depth=0.0,
        lambda_sky=0.1,
        lambda_foreground=0.0,
    )
    with pytest.raises(ValueError, match="strict sky.*rows \\[0\\]"):
        training_cli.validate_training_supervision(
            _manifest(_frame(has_sky=False)), sky_loss
        )

    actor_loss = ArmGSLoss(
        require_auxiliary=True,
        lambda_depth=0.0,
        lambda_sky=0.0,
        lambda_foreground=0.1,
    )
    with pytest.raises(ValueError, match="requires.*dynamic actor track"):
        training_cli.validate_training_supervision(
            _manifest(_frame(), has_tracks=False), actor_loss
        )
    training_cli.validate_training_supervision(
        _manifest(_frame(has_actor=False)), actor_loss
    )


def test_non_strict_loss_allows_optional_supervision_to_be_absent() -> None:
    loss = ArmGSLoss(
        require_auxiliary=False,
        lambda_depth=1.0,
        lambda_sky=1.0,
        lambda_foreground=1.0,
    )
    training_cli.validate_training_supervision(
        _manifest(
            _frame(has_depth=False, has_sky=False, has_actor=False),
            has_tracks=False,
        ),
        loss,
    )


class _CheckpointTrainer:
    def __init__(self) -> None:
        self.loaded: dict[str, Any] | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": 4,
            "sampler_state": {"epoch": 2, "cursor": 1},
            "density_state": {"-1": {"seen": 12}},
            "tensor": torch.tensor([3.0]),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.loaded = state



def _dataset_identity_fixture(
    tmp_path: Path,
) -> tuple[
    tuple[SimpleNamespace, ...],
    dict[str, Path],
    dict[str, Path],
]:
    files = {
        name: tmp_path / filename
        for name, filename in {
            "calibration": "calib.txt",
            "poses": "poses.txt",
            "times": "times.txt",
            "tracklets": "tracklet_labels.xml",
            "image_0": "000000.png",
            "image_1": "000001.png",
            "lidar_0": "000000.bin",
            "lidar_1": "000001.bin",
            "sky_0": "000000.sky.png",
            "sky_1": "000001.sky.png",
            "actor_0": "000000.actor.png",
            "actor_1": "000001.actor.png",
        }.items()
    }
    for index, (name, path) in enumerate(files.items()):
        path.write_bytes(f"{index}:{name}".encode("utf-8"))

    frames = tuple(
        SimpleNamespace(
            image_path=files[f"image_{index}"],
            lidar=SimpleNamespace(source_path=files[f"lidar_{index}"]),
            sky_mask_path=files[f"sky_{index}"],
            actor_mask_path=files[f"actor_{index}"],
        )
        for index in range(2)
    )
    metadata_paths = {
        "calibration_path": files["calibration"],
        "poses_path": files["poses"],
        "times_path": files["times"],
        "tracklet_path": files["tracklets"],
    }
    return frames, metadata_paths, files


def test_dataset_input_identity_is_canonically_ordered(tmp_path: Path) -> None:
    frames, metadata_paths, _ = _dataset_identity_fixture(tmp_path)

    forward = training_cli.build_dataset_input_identity(
        frames, **metadata_paths
    )
    reverse = training_cli.build_dataset_input_identity(
        tuple(reversed(frames)), **metadata_paths
    )

    assert forward == reverse
    assert forward["file_count"] == 12
    assert len(forward["digest_sha256"]) == 64
    assert forward["frame_payload_verification"] == "stat_identity"
    assert (
        forward["small_metadata_verification"]
        == "stat_identity+full_content_sha256"
    )


@pytest.mark.parametrize("mutation", ["size", "mtime_ns"])
def test_resume_rejects_changed_payload_stat_identity(
    tmp_path: Path, mutation: str
) -> None:
    frames, metadata_paths, files = _dataset_identity_fixture(tmp_path)
    original_identity = training_cli.build_dataset_input_identity(
        frames, **metadata_paths
    )
    config = {"optimization": {"iterations": 10}, "model": {"kind": "same"}}
    checkpoint = tmp_path / "checkpoint.pt"
    training_cli.save_training_checkpoint(
        checkpoint,
        _CheckpointTrainer(),
        config,
        {"dataset_input_identity": original_identity},
    )

    target = files["image_0"]
    before = target.stat()
    if mutation == "size":
        target.write_bytes(target.read_bytes() + b"!")
    else:
        os.utime(
            target,
            ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
        )
        assert target.stat().st_mtime_ns != before.st_mtime_ns

    current_identity = training_cli.build_dataset_input_identity(
        frames, **metadata_paths
    )
    assert current_identity["digest_sha256"] != original_identity["digest_sha256"]

    restored = _CheckpointTrainer()
    with pytest.raises(ValueError, match="dataset/split identity differs"):
        training_cli.restore_training_checkpoint(
            restored,
            checkpoint,
            map_location="cpu",
            config=config,
            run_metadata={"dataset_input_identity": current_identity},
        )
    assert restored.loaded is None


def test_small_metadata_content_hash_detects_same_stat_content_change(
    tmp_path: Path,
) -> None:
    frames, metadata_paths, files = _dataset_identity_fixture(tmp_path)
    original_identity = training_cli.build_dataset_input_identity(
        frames, **metadata_paths
    )
    target = files["calibration"]
    original_stat = target.stat()

    target.write_bytes(b"X" * original_stat.st_size)
    os.utime(
        target,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    changed_stat = target.stat()
    assert changed_stat.st_size == original_stat.st_size
    assert changed_stat.st_mtime_ns == original_stat.st_mtime_ns

    current_identity = training_cli.build_dataset_input_identity(
        frames, **metadata_paths
    )
    assert current_identity["digest_sha256"] != original_identity["digest_sha256"]


def test_atomic_checkpoint_preserves_sampler_density_and_allows_new_target(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoints" / "step.pt"
    trainer = _CheckpointTrainer()
    old_config = {"optimization": {"iterations": 10}, "model": {"kind": "same"}}
    metadata = {"train_source_indices": [1, 2], "eval_source_indices": [0]}

    training_cli.save_training_checkpoint(path, trainer, old_config, metadata)
    loaded = training_cli.load_training_checkpoint(path, map_location="cpu")
    assert loaded["trainer"]["sampler_state"]["cursor"] == 1
    assert loaded["trainer"]["density_state"]["-1"]["seen"] == 12
    assert loaded["trainer"]["tensor"].device.type == "cpu"

    new_config = {"optimization": {"iterations": 20}, "model": {"kind": "same"}}
    restored = _CheckpointTrainer()
    training_cli.restore_training_checkpoint(
        restored,
        path,
        map_location=torch.device("cpu"),
        config=new_config,
        run_metadata=metadata,
    )
    assert restored.loaded is not None
    assert restored.loaded["step"] == 4


def test_atomic_save_keeps_old_destination_and_cleans_temp_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "checkpoint.pt"
    destination.write_bytes(b"previous-checkpoint")

    def fail_save(payload: Any, path: Path) -> None:
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(training_cli.torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="simulated write failure"):
        training_cli._atomic_torch_save({"new": True}, destination)

    assert destination.read_bytes() == b"previous-checkpoint"
    assert list(tmp_path.glob(".checkpoint.pt.*.tmp")) == []


def test_checkpoint_loader_forwards_map_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"placeholder")
    captured: dict[str, Any] = {}

    def fake_load(path_value: Path, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "format_version": 2,
            "trainer": {},
            "config": {},
            "run_metadata": {},
        }

    monkeypatch.setattr(training_cli.torch, "load", fake_load)
    training_cli.load_training_checkpoint(
        path, map_location=torch.device("cpu")
    )
    assert captured["map_location"] == torch.device("cpu")
    assert captured["weights_only"] is True


class _LoopBatch(NamedTuple):
    frame: Any
    row: int
    device: str
    target_rgb: torch.Tensor


def _loop_batch(frame: Any, row: int, device: Any) -> _LoopBatch:
    return _LoopBatch(
        frame=frame,
        row=row,
        device=str(device),
        target_rgb=torch.zeros((1, 1, 1, 3), dtype=torch.float32),
    )


class _LoopTrainer:
    def __init__(self, *, step: int = 0, sampler: StatefulShuffleSampler | None = None):
        self.step = step
        self.sampler = sampler or StatefulShuffleSampler(
            2, seed=0, shuffle=False
        )
        self.batches: list[Any] = []
        background = SimpleNamespace(count=2)
        actor = SimpleNamespace(
            actor_id=7,
            gaussians=SimpleNamespace(count=3),
        )
        scene = SimpleNamespace(background=background, actors=(actor,))
        self.renderer = SimpleNamespace(scene=scene)
        self.optimizer = SimpleNamespace(
            param_groups=[
                {"name": "means", "lr": 1.0e-3},
                {"name": "appearance", "lr": 2.0e-3},
            ]
        )

    def train_step(self, batch: Any) -> SimpleNamespace:
        self.batches.append(batch)
        self.step += 1
        scalar = torch.tensor(float(self.step))
        losses = SimpleNamespace(
            total=scalar,
            rgb=scalar,
            ssim=scalar / 10.0,
            depth=scalar,
            sky=scalar,
            foreground=scalar,
        )
        rendering = SimpleNamespace(
            rgb=torch.full_like(batch.target_rgb, self.step / 10.0)
        )
        return SimpleNamespace(losses=losses, rendering=rendering)


def test_loss_telemetry_accumulates_exact_psnr_and_ssim_on_device() -> None:
    sums: dict[str, torch.Tensor] = {}
    output = SimpleNamespace(
        losses=SimpleNamespace(
            total=torch.tensor(1.0),
            rgb=torch.tensor(2.0),
            ssim=torch.tensor(0.25),
            depth=torch.tensor(3.0),
            sky=torch.tensor(4.0),
            foreground=torch.tensor(5.0),
        ),
        rendering=SimpleNamespace(
            rgb=torch.full((1, 2, 2, 3), 0.5, dtype=torch.float64)
        ),
    )
    batch = SimpleNamespace(
        target_rgb=torch.zeros((1, 2, 2, 3), dtype=torch.float64)
    )

    training_cli._accumulate_loss_telemetry(sums, output, batch)
    training_cli._accumulate_loss_telemetry(sums, output, batch)

    assert sums["train/psnr"].dtype == torch.float64
    assert sums["train/psnr"].device == output.rendering.rgb.device
    assert sums["train/psnr"].item() == pytest.approx(
        2.0 * -10.0 * math.log10(0.25)
    )
    assert sums["train/ssim"].dtype == torch.float64
    assert sums["train/ssim"].item() == pytest.approx(1.5)


def test_training_loop_hits_exact_target_and_checkpoint_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        training_cli,
        "canonical_frame_to_training_batch",
        _loop_batch,
    )
    trainer = _LoopTrainer()
    checkpoints: list[int] = []

    training_cli.train_until(
        trainer,
        ["frame-0", "frame-1"],
        total_iterations=5,
        device="cpu",
        checkpoint_interval=2,
        log_interval=100,
        checkpoint_callback=checkpoints.append,
    )

    assert trainer.step == 5
    assert [batch[1] for batch in trainer.batches] == [0, 1, 0, 1, 0]
    assert checkpoints == [2, 4]


def test_completed_step_callback_is_resume_safe_and_follows_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        training_cli,
        "canonical_frame_to_training_batch",
        _loop_batch,
    )
    trainer = _LoopTrainer(step=2)
    events: list[tuple[str, int]] = []

    training_cli.train_until(
        trainer,
        ["frame-0", "frame-1"],
        total_iterations=4,
        device="cpu",
        checkpoint_interval=1,
        log_interval=100,
        checkpoint_callback=lambda step: events.append(("checkpoint", step)),
        completed_step_callback=lambda step: events.append(("completed", step)),
    )

    assert events == [
        ("checkpoint", 3),
        ("completed", 3),
        ("checkpoint", 4),
        ("completed", 4),
    ]



def test_training_loop_forwards_exact_json_log_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        training_cli,
        "canonical_frame_to_training_batch",
        _loop_batch,
    )
    trainer = _LoopTrainer()
    records: list[dict[str, Any]] = []

    training_cli.train_until(
        trainer,
        ["frame-0", "frame-1"],
        total_iterations=3,
        device="cpu",
        checkpoint_interval=10,
        log_interval=2,
        checkpoint_callback=lambda step: None,
        log_callback=records.append,
    )

    assert [record["step"] for record in records] == [2, 3]
    assert [record["train/loss"] for record in records] == [1.5, 3.0]
    assert [record["train/rgb_l1"] for record in records] == [1.5, 3.0]
    assert [record["train/psnr"] for record in records] == pytest.approx(
        [
            (
                -10.0 * math.log10(0.1**2)
                + -10.0 * math.log10(0.2**2)
            )
            / 2.0,
            -10.0 * math.log10(0.3**2),
        ]
    )
    assert [record["train/ssim"] for record in records] == pytest.approx(
        [0.85, 0.7]
    )
    assert all(record["train/gaussians/total"] == 5 for record in records)
    assert all(record["train/gaussians/background"] == 2 for record in records)
    assert all(record["train/gaussians/actors"] == 3 for record in records)
    assert all(record["train/gaussians/actor/7"] == 3 for record in records)
    assert all(record["train/lr/means"] == 1.0e-3 for record in records)
    assert all(record["train/lr/appearance"] == 2.0e-3 for record in records)
    assert [record["train/telemetry/window_steps"] for record in records] == [
        2,
        1,
    ]
    assert all(
        record["train/performance/step_time_seconds"] >= 0.0
        for record in records
    )
    assert all(
        record["train/performance/steps_per_second"] > 0.0
        for record in records
    )
    assert all(
        not any(key.startswith("train/cuda/") for key in record)
        for record in records
    )


def test_training_loop_merges_context_payload_only_at_log_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        training_cli,
        "canonical_frame_to_training_batch",
        _loop_batch,
    )
    trainer = _LoopTrainer()
    records: list[dict[str, Any]] = []
    factory_calls: list[tuple[Any, Any]] = []

    def payload_factory(batch: Any, output: Any) -> dict[str, Any]:
        factory_calls.append((batch, output))
        return {"preview": f"{batch[0]}:{output.losses.total.item():.0f}"}

    training_cli.train_until(
        trainer,
        ["frame-0", "frame-1"],
        total_iterations=3,
        device="cpu",
        checkpoint_interval=10,
        log_interval=2,
        checkpoint_callback=lambda step: None,
        log_callback=records.append,
        log_payload_factory=payload_factory,
    )

    assert [record["step"] for record in records] == [2, 3]
    assert [record["preview"] for record in records] == [
        "frame-1:2",
        "frame-0:3",
    ]
    assert len(factory_calls) == 2


def test_training_loop_payload_has_independent_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        training_cli,
        "canonical_frame_to_training_batch",
        _loop_batch,
    )
    trainer = _LoopTrainer()
    records: list[dict[str, Any]] = []
    factory_steps: list[int] = []

    def payload_factory(batch: Any, output: Any) -> dict[str, Any]:
        factory_steps.append(trainer.step)
        return {"train/preview": batch[0]}

    training_cli.train_until(
        trainer,
        ["frame-0", "frame-1"],
        total_iterations=6,
        device="cpu",
        checkpoint_interval=10,
        log_interval=4,
        checkpoint_callback=lambda step: None,
        log_callback=records.append,
        log_payload_factory=payload_factory,
        payload_interval=3,
    )

    assert [record["step"] for record in records] == [3, 4, 6]
    assert set(records[0]) == {"step", "train/preview"}
    assert "train/preview" not in records[1]
    assert records[2]["train/preview"] == "frame-1"
    assert records[1]["train/loss"] == pytest.approx(2.5)
    assert records[2]["train/loss"] == pytest.approx(5.5)
    assert factory_steps == [3, 6]


def test_training_loop_can_disable_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        training_cli,
        "canonical_frame_to_training_batch",
        _loop_batch,
    )
    records: list[dict[str, Any]] = []

    training_cli.train_until(
        _LoopTrainer(),
        ["frame-0", "frame-1"],
        total_iterations=2,
        device="cpu",
        checkpoint_interval=10,
        log_interval=1,
        checkpoint_callback=lambda step: None,
        log_callback=records.append,
        log_payload_factory=lambda batch, output: pytest.fail(
            "disabled payload factory was called"
        ),
        payload_interval=0,
    )

    assert [record["step"] for record in records] == [1, 2]


def test_training_loop_aggregates_density_update_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        training_cli,
        "canonical_frame_to_training_batch",
        _loop_batch,
    )
    trainer = _LoopTrainer()
    original_train_step = trainer.train_step

    def train_step(batch: Any) -> SimpleNamespace:
        output = original_train_step(batch)
        if trainer.step == 1:
            output.density_updates = None
        else:
            output.density_updates = {
                -1: SimpleNamespace(
                    topology_changed=True,
                    duplicated_count=2,
                    split_parent_count=1,
                    split_child_count=3,
                    pruned_count=4,
                    opacity_was_reset=True,
                ),
                7: SimpleNamespace(
                    topology_changed=False,
                    duplicated_count=0,
                    split_parent_count=0,
                    split_child_count=0,
                    pruned_count=0,
                    opacity_was_reset=False,
                ),
            }
        return output

    trainer.train_step = train_step
    records: list[dict[str, Any]] = []
    training_cli.train_until(
        trainer,
        ["frame-0", "frame-1"],
        total_iterations=2,
        device="cpu",
        checkpoint_interval=10,
        log_interval=2,
        checkpoint_callback=lambda step: None,
        log_callback=records.append,
    )

    assert records[0]["train/density/steps_with_results"] == 1
    assert records[0]["train/density/updated_groups"] == 2
    assert records[0]["train/density/topology_changed_groups"] == 1
    assert records[0]["train/density/duplicated_gaussians"] == 2
    assert records[0]["train/density/split_parent_gaussians"] == 1
    assert records[0]["train/density/split_child_gaussians"] == 3
    assert records[0]["train/density/pruned_gaussians"] == 4
    assert records[0]["train/density/opacity_reset_groups"] == 1


def test_training_loop_resumes_an_exactly_exhausted_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        training_cli,
        "canonical_frame_to_training_batch",
        _loop_batch,
    )
    sampler = StatefulShuffleSampler(2, seed=0, shuffle=False)
    unfinished_iterator = iter(sampler)
    assert next(unfinished_iterator) == 0
    assert next(unfinished_iterator) == 1
    assert sampler.cursor == 2
    trainer = _LoopTrainer(step=2, sampler=sampler)

    training_cli.train_until(
        trainer,
        ["frame-0", "frame-1"],
        total_iterations=3,
        device="cpu",
        checkpoint_interval=10,
        log_interval=10,
        checkpoint_callback=lambda step: None,
    )

    assert trainer.step == 3
    assert [(batch[0], batch[1]) for batch in trainer.batches] == [
        ("frame-0", 0)
    ]


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_help_is_cpu_only_and_runtime_errors_use_exit_two(
    tmp_path: Path,
) -> None:
    help_result = _run_cli("--help")
    assert help_result.returncode == 0
    assert "exact checkpoint resume" in help_result.stdout
    assert "--sky-mask-dir CAMERA_ID=DIR" in help_result.stdout

    error_result = _run_cli(
        "--kitti-root",
        str(tmp_path / "missing-sequence"),
        "--output-dir",
        str(tmp_path / "output"),
        "--device",
        "cpu",
    )
    assert error_result.returncode == 2
    assert error_result.stderr.startswith("error:")
