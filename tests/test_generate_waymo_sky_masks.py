from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import pytest
import torch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "generate_waymo_sky_masks.py"
)
MODULE_NAME = "_armgs_generate_waymo_sky_masks_for_tests"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
waymo_sky = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = waymo_sky
SPEC.loader.exec_module(waymo_sky)


_SEQUENCE = "1005081002024129653_5313_150_5333_150"


def _arguments(root: Path, output: Path, *extra: str) -> argparse.Namespace:
    return waymo_sky.parse_args(
        [
            "--waymo-root",
            str(root),
            "--sequence",
            _SEQUENCE,
            "--start-frame",
            "7",
            "--end-frame",
            "8",
            "--target-height",
            "4",
            "--target-width",
            "6",
            "--source-image-height",
            "8",
            "--top-edge-original-pixels",
            "1",
            "--output-root",
            str(output),
            *extra,
        ]
    )


class _FakeManifestLoader:
    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root
        self.calls: list[tuple[Path, dict[str, Any]]] = []

    def __call__(self, root: str | Path, **kwargs: Any) -> Any:
        self.calls.append((Path(root), kwargs))
        height, width = kwargs["target_size"]
        start = int(kwargs["start_frame"])
        end = int(kwargs["end_frame"])
        frames = []
        for relative_index, source_index in enumerate(range(start, end + 1)):
            image_path = (
                self.cache_root
                / kwargs["sequence"]
                / "images"
                / "FRONT"
                / f"{source_index:08d}.png"
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.full(
                (height, width, 3),
                40 + 20 * relative_index,
                dtype=np.uint8,
            )
            Image.fromarray(pixels, mode="RGB").save(image_path)
            frames.append(
                SimpleNamespace(
                    camera_id=0,
                    image_size=(height, width),
                    frame_index=relative_index,
                    image_path=image_path,
                )
            )
        return SimpleNamespace(frames=tuple(frames))


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def infer(self, image_path: Path) -> Any:
        self.calls.append(image_path)
        mask = np.zeros((4, 6), dtype=np.bool_)
        if len(self.calls) % 2 == 1:
            mask[:2, 1:4] = True
            return waymo_sky.WaymoSkyInferenceResult(
                mask=mask,
                candidate_detection_count=2,
                detection_count=1,
                top_edge_rejected_count=1,
                phrases=("sky",),
                logits=(0.9,),
            )
        return waymo_sky.WaymoSkyInferenceResult(
            mask=mask,
            candidate_detection_count=1,
            detection_count=0,
            top_edge_rejected_count=1,
        )


def test_parser_defaults_match_streetgs_waymo_recipe(tmp_path: Path) -> None:
    args = waymo_sky.parse_args(
        [
            "--waymo-root",
            str(tmp_path / "waymo"),
            "--sequence",
            _SEQUENCE,
            "--output-root",
            str(tmp_path / "masks"),
        ]
    )

    assert args.parquet_dir == "validation"
    assert (args.target_height, args.target_width) == (1066, 1600)
    assert args.text_prompt == "sky"
    assert args.box_threshold == pytest.approx(0.3)
    assert args.text_threshold == pytest.approx(0.25)
    assert args.source_image_height == 1280
    assert args.top_edge_original_pixels == 100
    assert args.existing_mode == "resume"
    assert args.bert_path == (
        SCRIPT_PATH.parents[1]
        / "checkpoints"
        / "grounded_sam"
        / "bert-base-uncased"
    )


def test_parser_rejects_bad_thresholds_and_indices(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _arguments(
            tmp_path / "waymo",
            tmp_path / "masks",
            "--box-threshold",
            "1.1",
        )
    with pytest.raises(SystemExit):
        _arguments(
            tmp_path / "waymo",
            tmp_path / "masks",
            "--start-frame",
            "-1",
        )
    with pytest.raises(SystemExit):
        _arguments(
            tmp_path / "waymo",
            tmp_path / "masks",
            "--prepare-manifest",
            str(tmp_path / "prepared.json"),
            "--input-frame-manifest",
            str(tmp_path / "input.json"),
        )


def test_enumeration_uses_camera_only_loader_and_absolute_source_indices(
    tmp_path: Path,
) -> None:
    loader = _FakeManifestLoader(tmp_path / "cache")

    frames = waymo_sky.enumerate_waymo_front_frames(
        tmp_path / "waymo",
        sequence=_SEQUENCE,
        start_frame=7,
        end_frame=8,
        target_size=(4, 6),
        cache_dir=tmp_path / "cache",
        manifest_loader=loader,
    )

    assert [(frame.frame_index, frame.source_frame_index) for frame in frames] == [
        (0, 7),
        (1, 8),
    ]
    assert [frame.image_path.name for frame in frames] == [
        "00000007.png",
        "00000008.png",
    ]
    assert len(loader.calls) == 1
    _, kwargs = loader.calls[0]
    assert kwargs["camera_channels"] == ("FRONT",)
    assert kwargs["sky_mask_root"] is None
    assert kwargs["require_lidar"] is False

    assert waymo_sky.mask_output_path(tmp_path / "masks", frame=frames[0]) == (
        tmp_path
        / "masks"
        / _SEQUENCE
        / "FRONT"
        / "00000007.png"
    )


def test_top_edge_filter_scales_original_top_100_pixels() -> None:
    assert waymo_sky.scaled_top_edge_pixels(target_height=1066) == pytest.approx(
        83.28125
    )
    height = 0.1
    top_edges = torch.tensor(
        [99.0 / 1280.0, 100.5 / 1280.0, 101.0 / 1280.0],
        dtype=torch.float64,
    )
    boxes = torch.stack(
        (
            torch.full_like(top_edges, 0.5),
            top_edges + 0.5 * height,
            torch.full_like(top_edges, 0.2),
            torch.full_like(top_edges, height),
        ),
        dim=-1,
    )

    accepted = waymo_sky.top_edge_acceptance_mask(boxes, target_height=1066)

    assert accepted.tolist() == [True, False, False]


def test_backend_filters_dino_boxes_before_sam(tmp_path: Path) -> None:
    image_path = tmp_path / "00000000.png"
    Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8), mode="RGB").save(
        image_path
    )
    transformed = object()

    def predict(**kwargs: Any) -> tuple[Any, Any, list[str]]:
        assert kwargs["caption"] == "sky"
        boxes = torch.tensor(
            [
                [0.5, 0.10, 0.6, 0.10],
                [0.5, 0.40, 0.2, 0.10],
            ]
        )
        return boxes, torch.tensor([0.9, 0.8]), ["sky", "sky"]

    class _BoxOps:
        @staticmethod
        def box_cxcywh_to_xyxy(boxes: Any) -> Any:
            return boxes

    class _SamTransform:
        accepted_box_count = 0

        @classmethod
        def apply_boxes_torch(cls, boxes: Any, image_size: Any) -> Any:
            cls.accepted_box_count = int(boxes.shape[0])
            return boxes

    class _Predictor:
        def __init__(self) -> None:
            self.transform = _SamTransform()
            self.set_image_calls = 0

        def set_image(self, image: Any) -> None:
            self.set_image_calls += 1

        def predict_torch(self, **kwargs: Any) -> tuple[Any, None, None]:
            mask = torch.zeros((1, 1, 4, 6), dtype=torch.bool)
            mask[:, :, :2, :] = True
            return mask, None, None

    predictor = _Predictor()
    backend = object.__new__(waymo_sky.WaymoGroundedSamBackend)
    backend._np = np
    backend._torch = torch
    backend._transform = lambda image, target: (transformed, target)
    backend._box_ops = _BoxOps()
    backend._predict = predict
    backend._grounding_model = object()
    backend._sam_predictor = predictor
    backend._text_prompt = "sky"
    backend._box_threshold = 0.3
    backend._text_threshold = 0.25
    backend._device = "cpu"
    backend._source_image_height = 1280
    backend._top_edge_original_pixels = 100

    result = backend.infer(image_path)

    assert result.candidate_detection_count == 2
    assert result.detection_count == 1
    assert result.top_edge_rejected_count == 1
    assert result.phrases == ("sky",)
    assert result.logits == pytest.approx((0.9,))
    assert predictor.set_image_calls == 1
    assert _SamTransform.accepted_box_count == 1
    assert result.mask.sum() == 12


def test_generation_writes_loader_compatible_masks_manifest_and_qa(
    tmp_path: Path,
) -> None:
    root = tmp_path / "waymo"
    output = tmp_path / "sky_masks"
    contact_sheet = tmp_path / "qa" / "contact.jpg"
    args = _arguments(
        root,
        output,
        "--save-overlays",
        "--contact-sheet",
        str(contact_sheet),
    )
    loader = _FakeManifestLoader(tmp_path / "cache")
    backend = _FakeBackend()

    payload = waymo_sky.generate_waymo_sky_masks(
        args,
        backend=backend,
        manifest_loader=loader,
    )

    assert len(backend.calls) == 2
    assert payload["dataset"]["require_lidar"] is False
    assert payload["summary"] == {
        "total": 2,
        "statuses": {"generated": 1, "no_detection": 1},
        "no_detection": 1,
        "candidate_detections": 3,
        "accepted_detections": 1,
        "top_edge_rejected_detections": 2,
        "mean_coverage": pytest.approx(0.125),
        "min_coverage": 0.0,
        "max_coverage": pytest.approx(0.25),
    }
    first_mask = output / _SEQUENCE / "FRONT" / "00000007.png"
    with Image.open(first_mask) as image:
        assert image.mode == "L"
        assert image.size == (6, 4)
        assert set(np.unique(np.asarray(image)).tolist()) == {0, 255}
    manifest = output / _SEQUENCE / "generation_manifest.json"
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert saved["frames"][0]["source_frame_index"] == 7
    assert saved["generation"]["scaled_top_edge_pixels"] == pytest.approx(0.5)
    assert (
        output / _SEQUENCE / "overlays" / "FRONT" / "00000007.jpg"
    ).is_file()
    assert contact_sheet.is_file()


def test_resume_skips_valid_masks_and_regenerates_corrupt_mask(
    tmp_path: Path,
) -> None:
    root = tmp_path / "waymo"
    output = tmp_path / "sky_masks"
    loader = _FakeManifestLoader(tmp_path / "cache")
    first_backend = _FakeBackend()
    waymo_sky.generate_waymo_sky_masks(
        _arguments(root, output),
        backend=first_backend,
        manifest_loader=loader,
    )

    resumed_backend = _FakeBackend()
    resumed = waymo_sky.generate_waymo_sky_masks(
        _arguments(root, output, "--resume"),
        backend=resumed_backend,
        manifest_loader=loader,
    )
    assert resumed_backend.calls == []
    assert resumed["summary"]["statuses"] == {"skipped_existing": 2}
    assert resumed["summary"]["candidate_detections"] == 3
    assert resumed["summary"]["top_edge_rejected_detections"] == 2

    corrupt = output / _SEQUENCE / "FRONT" / "00000007.png"
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB").save(corrupt)
    repair_backend = _FakeBackend()
    repaired = waymo_sky.generate_waymo_sky_masks(
        _arguments(root, output),
        backend=repair_backend,
        manifest_loader=loader,
    )
    assert len(repair_backend.calls) == 1
    assert repaired["summary"]["statuses"] == {
        "generated": 1,
        "skipped_existing": 1,
    }
    with Image.open(corrupt) as image:
        assert image.mode == "L"
        assert image.size == (6, 4)


def test_dry_run_does_not_load_backend_or_write_mask_outputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "waymo"
    output = tmp_path / "sky_masks"
    loader = _FakeManifestLoader(tmp_path / "cache")

    payload = waymo_sky.generate_waymo_sky_masks(
        _arguments(root, output, "--dry-run"),
        manifest_loader=loader,
    )

    assert payload["summary"]["statuses"] == {"dry_run_pending": 2}
    assert payload["frames"][0]["mask_path"].endswith(
        f"{_SEQUENCE}/FRONT/00000007.png"
    )
    assert not output.exists()


def test_two_stage_manifest_avoids_waymo_loader_during_inference(
    tmp_path: Path,
) -> None:
    root = tmp_path / "waymo"
    output = tmp_path / "sky_masks"
    cache = tmp_path / "cache"
    frame_manifest = tmp_path / "stage" / "front_frames.json"
    prepare_args = _arguments(
        root,
        output,
        "--cache-dir",
        str(cache),
        "--prepare-manifest",
        str(frame_manifest),
    )
    loader = _FakeManifestLoader(cache)

    prepared = waymo_sky.prepare_waymo_frame_manifest(
        prepare_args,
        manifest_loader=loader,
    )

    assert frame_manifest.is_file()
    assert prepared["kind"] == "armgs_waymo_front_frame_manifest"
    assert prepared["summary"] == {"total": 2}
    assert prepared["dataset"]["require_lidar"] is False
    assert prepared["frames"][0]["source_frame_index"] == 7
    assert prepared["frames"][0]["image_path"].endswith("00000007.png")
    assert prepared["frames"][0]["image_stat"]["size_bytes"] > 0

    inference_args = _arguments(
        root,
        output,
        "--cache-dir",
        str(cache),
        "--input-frame-manifest",
        str(frame_manifest),
    )

    def forbidden_waymo_loader(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the inference stage must not call the Waymo loader")

    backend = _FakeBackend()
    generated = waymo_sky.generate_waymo_sky_masks(
        inference_args,
        backend=backend,
        manifest_loader=forbidden_waymo_loader,
    )

    assert len(backend.calls) == 2
    assert generated["dataset"]["input_frame_manifest"] == str(
        frame_manifest.resolve()
    )
    assert (
        output / _SEQUENCE / "FRONT" / "00000007.png"
    ).is_file()


def test_prepared_manifest_rejects_changed_image_and_source_index(
    tmp_path: Path,
) -> None:
    root = tmp_path / "waymo"
    output = tmp_path / "sky_masks"
    cache = tmp_path / "cache"
    frame_manifest = tmp_path / "front_frames.json"
    prepare_args = _arguments(
        root,
        output,
        "--cache-dir",
        str(cache),
        "--prepare-manifest",
        str(frame_manifest),
    )
    waymo_sky.prepare_waymo_frame_manifest(
        prepare_args,
        manifest_loader=_FakeManifestLoader(cache),
    )
    inference_args = _arguments(
        root,
        output,
        "--cache-dir",
        str(cache),
        "--input-frame-manifest",
        str(frame_manifest),
    )

    prepared = json.loads(frame_manifest.read_text(encoding="utf-8"))
    image_path = Path(prepared["frames"][0]["image_path"])
    image_path.write_bytes(image_path.read_bytes() + b"changed-after-prepare")
    with pytest.raises(ValueError, match="size_bytes changed"):
        waymo_sky.load_prepared_waymo_front_frames(
            frame_manifest,
            args=inference_args,
        )

    prepared["frames"][0]["image_stat"]["size_bytes"] = image_path.stat().st_size
    prepared["frames"][0]["image_stat"]["mtime_ns"] = image_path.stat().st_mtime_ns
    prepared["frames"][0]["source_frame_index"] = 9
    frame_manifest.write_text(json.dumps(prepared), encoding="utf-8")
    with pytest.raises(ValueError, match="absolute source index"):
        waymo_sky.load_prepared_waymo_front_frames(
            frame_manifest,
            args=inference_args,
        )


def test_shell_wrapper_orchestrates_camosplat_then_armgs_gsam(
    tmp_path: Path,
) -> None:
    wrapper = SCRIPT_PATH.with_suffix(".sh")
    waymo_root = tmp_path / "waymo"
    (waymo_root / "validation").mkdir(parents=True)
    fake_conda = tmp_path / "fake-conda"
    log_path = tmp_path / "conda.log"
    fake_conda.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "$FAKE_CONDA_LOG"
previous=''
for argument in "$@"; do
  if [[ "$previous" == "--prepare-manifest" ]]; then
    mkdir -p -- "$(dirname -- "$argument")"
    printf '{"prepared": true}\n' > "$argument"
  fi
  previous="$argument"
done
""",
        encoding="utf-8",
    )
    fake_conda.chmod(0o755)
    frame_manifest = tmp_path / "stage" / "frames.json"
    environment = os.environ.copy()
    environment.update(
        {
            "CONDA_BIN": str(fake_conda),
            "PREPARE_PYTHON": str(fake_conda),
            "INFERENCE_PYTHON": str(fake_conda),
            "FAKE_CONDA_LOG": str(log_path),
            "WAYMO_ROOT": str(waymo_root),
            "CACHE_DIR": str(tmp_path / "cache"),
            "OUTPUT_ROOT": str(tmp_path / "masks"),
            "FRAME_MANIFEST": str(frame_manifest),
            "DRY_RUN": "1",
            "SAVE_OVERLAYS": "0",
        }
    )

    completed = subprocess.run(
        ["bash", str(wrapper), _SEQUENCE, "2"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    calls = log_path.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert "generate_waymo_sky_masks.py" in calls[0]
    assert "run -n" not in calls[0]
    assert f"--prepare-manifest {frame_manifest}" in calls[0]
    assert "generate_waymo_sky_masks.py" in calls[1]
    assert "run -n" not in calls[1]
    assert f"--input-frame-manifest {frame_manifest}" in calls[1]
    assert "--dry-run" in calls[1]
