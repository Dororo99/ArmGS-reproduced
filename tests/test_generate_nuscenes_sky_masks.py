from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "generate_nuscenes_sky_masks.py"
)
MODULE_NAME = "_armgs_generate_nuscenes_sky_masks_for_tests"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
sky_cli = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = sky_cli
SPEC.loader.exec_module(sky_cli)


_CAMERAS = ("CAM_FRONT", "CAM_BACK")


def _write_json(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def _make_nuscenes(root: Path) -> Path:
    metadata = root / "v1.0-trainval"
    metadata.mkdir(parents=True)
    for channel in _CAMERAS:
        (root / "samples" / channel).mkdir(parents=True)

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
    _write_json(
        metadata / "sensor.json",
        [
            {
                "token": f"sensor-{channel.lower()}",
                "channel": channel,
                "modality": "camera",
            }
            for channel in _CAMERAS
        ],
    )
    _write_json(
        metadata / "calibrated_sensor.json",
        [
            {
                "token": f"calib-{channel.lower()}",
                "sensor_token": f"sensor-{channel.lower()}",
                "translation": [0.0, 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "camera_intrinsic": [
                    [10.0, 0.0, 2.0],
                    [0.0, 10.0, 2.0],
                    [0.0, 0.0, 1.0],
                ],
            }
            for channel in _CAMERAS
        ],
    )

    sample_data: list[dict[str, object]] = []
    for frame_index, sample in enumerate(samples):
        for channel in _CAMERAS:
            token = f"data-{frame_index}-{channel.lower()}"
            relative = Path("samples") / channel / f"{token}.jpg"
            pixels = np.full((4, 6, 3), 40 + frame_index * 30, dtype=np.uint8)
            Image.fromarray(pixels, mode="RGB").save(root / relative)
            sample_data.append(
                {
                    "token": token,
                    "sample_token": sample["token"],
                    "ego_pose_token": f"pose-{token}",
                    "calibrated_sensor_token": f"calib-{channel.lower()}",
                    "timestamp": int(sample["timestamp"]),
                    "is_key_frame": True,
                    "height": 4,
                    "width": 6,
                    "filename": relative.as_posix(),
                }
            )
    sample_data.append(
        {
            **sample_data[0],
            "token": "ignored-sweep",
            "is_key_frame": False,
        }
    )
    _write_json(metadata / "sample_data.json", sample_data)
    return root


def _arguments(root: Path, output: Path, *extra: str) -> argparse.Namespace:
    return sky_cli.parse_args(
        [
            "--nuscenes-root",
            str(root),
            "--scene",
            "61",
            "--version",
            "v1.0-trainval",
            "--cameras",
            ",".join(_CAMERAS),
            "--output-root",
            str(output),
            *extra,
        ]
    )


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def infer(self, image_path: Path) -> Any:
        self.calls.append(image_path)
        mask = np.zeros((4, 6), dtype=np.bool_)
        call_index = len(self.calls) - 1
        if call_index % 2 == 0:
            mask[0:2, 1:4] = True
            return sky_cli.SkyInferenceResult(
                mask=mask,
                detection_count=2,
                phrases=("sky", "sky"),
                logits=(0.9, 0.8),
            )
        return sky_cli.SkyInferenceResult(mask=mask, detection_count=0)


def test_groundingdino_arguments_preserve_pinned_bert_encoder_name(tmp_path: Path) -> None:
    arguments = SimpleNamespace(text_encoder_type="bert-base-uncased")
    bert_path = tmp_path / "bert-base-uncased"

    configured = sky_cli.configure_groundingdino_arguments(
        arguments, device="cuda:2", bert_path=bert_path
    )

    expected = str(bert_path.resolve())
    assert configured is arguments
    assert arguments.device == "cuda:2"
    assert arguments.bert_base_uncased_path == expected
    assert arguments.text_encoder_type == "bert-base-uncased"


def test_parser_exposes_model_threshold_resume_and_qa_options(tmp_path: Path) -> None:
    root = tmp_path / "nuscenes"
    output = tmp_path / "masks"
    args = _arguments(
        root,
        output,
        "--groundingdino-config",
        str(tmp_path / "dino.py"),
        "--groundingdino-checkpoint",
        str(tmp_path / "dino.pth"),
        "--sam-checkpoint",
        str(tmp_path / "sam.pth"),
        "--sam-model-type",
        "vit_b",
        "--bert-path",
        str(tmp_path / "bert"),
        "--text-prompt",
        "blue sky",
        "--box-threshold",
        "0.4",
        "--text-threshold",
        "0.2",
        "--device",
        "cuda:1",
        "--overwrite",
        "--save-overlays",
        "--overlay-every",
        "3",
        "--contact-sheet",
        str(tmp_path / "sheet.jpg"),
    )

    assert args.cameras == _CAMERAS
    assert args.sam_model_type == "vit_b"
    assert args.bert_path == tmp_path / "bert"
    assert args.text_prompt == "blue sky"
    assert args.box_threshold == pytest.approx(0.4)
    assert args.text_threshold == pytest.approx(0.2)
    assert args.device == "cuda:1"
    assert args.existing_mode == "overwrite"
    assert args.save_overlays is True
    assert args.overlay_every == 3


def test_parser_rejects_bad_threshold_and_camera_lists(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _arguments(tmp_path / "data", tmp_path / "out", "--box-threshold", "1.1")
    with pytest.raises(argparse.ArgumentTypeError, match="unknown"):
        sky_cli.parse_camera_channels("CAM_UNKNOWN")
    with pytest.raises(argparse.ArgumentTypeError, match="unique"):
        sky_cli.parse_camera_channels("CAM_FRONT,CAM_FRONT")


def test_enumeration_uses_keyframes_scene_order_and_sample_data_tokens(
    tmp_path: Path,
) -> None:
    root = _make_nuscenes(tmp_path / "nuscenes")

    frames = sky_cli.enumerate_nuscenes_camera_keyframes(
        root,
        scene="scene-0061",
        camera_channels=_CAMERAS,
    )

    assert len(frames) == 4
    assert [(frame.frame_index, frame.channel) for frame in frames] == [
        (0, "CAM_FRONT"),
        (0, "CAM_BACK"),
        (1, "CAM_FRONT"),
        (1, "CAM_BACK"),
    ]
    assert frames[0].sample_token == "sample-0"
    assert frames[0].sample_data_token == "data-0-cam_front"
    assert frames[0].image_size == (4, 6)

    output = sky_cli.mask_output_path(
        tmp_path / "sky_masks" / "nuscenes",
        version="v1.0-trainval",
        scene_name="scene-0061",
        frame=frames[0],
    )
    assert output == (
        tmp_path
        / "sky_masks"
        / "nuscenes"
        / "v1.0-trainval"
        / "scene-0061"
        / "CAM_FRONT"
        / "data-0-cam_front.png"
    )


def test_mask_union_accepts_sam_shape_and_ors_instances() -> None:
    masks = np.zeros((2, 1, 3, 4), dtype=np.bool_)
    masks[0, 0, 0, 0] = True
    masks[1, 0, 2, 3] = True

    result = sky_cli.union_instance_masks(masks, (3, 4))

    assert result.dtype == np.bool_
    assert result.sum() == 2
    assert result[0, 0] and result[2, 3]


def test_generation_writes_raw_binary_masks_manifest_overlays_and_contact_sheet(
    tmp_path: Path,
) -> None:
    root = _make_nuscenes(tmp_path / "nuscenes")
    output = tmp_path / "sky_masks" / "nuscenes"
    sheet = tmp_path / "qa" / "contact.jpg"
    args = _arguments(
        root,
        output,
        "--save-overlays",
        "--contact-sheet",
        str(sheet),
    )
    backend = _FakeBackend()

    payload = sky_cli.generate_sky_masks(args, backend=backend)

    assert len(backend.calls) == 4
    assert payload["summary"]["total"] == 4
    assert payload["summary"]["statuses"] == {"generated": 2, "no_detection": 2}
    assert payload["summary"]["no_detection"] == 2
    assert payload["summary"]["mean_coverage"] == pytest.approx(0.125)
    records = payload["frames"]
    first = Path(records[0]["mask_path"])
    with Image.open(first) as mask:
        assert mask.mode == "L"
        assert mask.size == (6, 4)
        values = set(np.unique(np.asarray(mask)).tolist())
    assert values == {0, 255}
    assert records[0]["coverage"] == pytest.approx(0.25)
    assert records[1]["status"] == "no_detection"
    assert records[1]["coverage"] == 0.0

    manifest = output / "v1.0-trainval" / "scene-0061" / "generation_manifest.json"
    assert json.loads(manifest.read_text(encoding="utf-8"))["summary"]["total"] == 4
    overlay = (
        output
        / "v1.0-trainval"
        / "scene-0061"
        / "overlays"
        / "CAM_FRONT"
        / "data-0-cam_front.jpg"
    )
    assert overlay.is_file()
    assert sheet.is_file()


def test_resume_skips_valid_masks_and_overwrite_regenerates(tmp_path: Path) -> None:
    root = _make_nuscenes(tmp_path / "nuscenes")
    output = tmp_path / "masks"
    first_backend = _FakeBackend()
    sky_cli.generate_sky_masks(_arguments(root, output), backend=first_backend)
    assert len(first_backend.calls) == 4

    resume_backend = _FakeBackend()
    resumed = sky_cli.generate_sky_masks(
        _arguments(root, output, "--resume"), backend=resume_backend
    )
    assert resume_backend.calls == []
    assert resumed["summary"]["statuses"] == {"skipped_existing": 4}
    assert resumed["summary"]["no_detection"] == 2

    overwrite_backend = _FakeBackend()
    overwritten = sky_cli.generate_sky_masks(
        _arguments(root, output, "--overwrite"), backend=overwrite_backend
    )
    assert len(overwrite_backend.calls) == 4
    assert overwritten["summary"]["statuses"] == {"generated": 2, "no_detection": 2}


def test_dry_run_never_loads_backend_or_writes_outputs(tmp_path: Path) -> None:
    root = _make_nuscenes(tmp_path / "nuscenes")
    output = tmp_path / "masks"
    args = _arguments(root, output, "--dry-run")

    payload = sky_cli.generate_sky_masks(args)

    assert payload["summary"]["statuses"] == {"dry_run_pending": 4}
    assert not output.exists()



def test_negative_prompt_cli_defaults_disabled_and_accepts_value(tmp_path: Path) -> None:
    default_args = _arguments(tmp_path / "data", tmp_path / "out")
    enabled_args = _arguments(
        tmp_path / "data",
        tmp_path / "out",
        "--negative-text-prompt",
        "building",
    )

    assert default_args.negative_text_prompt is None
    assert enabled_args.negative_text_prompt == "building"


def test_apply_exclusion_mask_removes_only_overlapping_pixels() -> None:
    sky = np.zeros((3, 4), dtype=np.bool_)
    sky[0:2, :] = True
    exclusion = np.zeros((3, 4), dtype=np.uint8)
    exclusion[0, 1:3] = 255
    exclusion[2, 3] = 255

    refined = sky_cli.apply_exclusion_mask(sky, exclusion, (3, 4))

    assert refined.dtype == np.bool_
    assert refined.flags.c_contiguous
    assert refined.sum() == 6
    assert not refined[0, 1]
    assert not refined[0, 2]
    assert refined[1, 1]
    assert not refined[2, 3]
    assert sky.sum() == 8


def test_backend_reuses_one_transform_and_sam_encoding_for_exclusion(
    tmp_path: Path,
) -> None:
    import torch

    image_path = tmp_path / "frame.jpg"
    Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8), mode="RGB").save(image_path)
    transformed = object()
    transform_calls: list[tuple[int, int]] = []
    predict_calls: list[tuple[str, int]] = []

    def transform(image: Any, target: Any) -> tuple[Any, Any]:
        transform_calls.append(image.size)
        return transformed, target

    def predict(**kwargs: Any) -> tuple[Any, Any, list[str]]:
        predict_calls.append((str(kwargs["caption"]), id(kwargs["image"])))
        if kwargs["caption"] == "sky":
            return torch.tensor([[0.0, 0.0, 1.0, 1.0]]), torch.tensor([0.9]), ["sky"]
        return (
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            torch.tensor([0.8]),
            ["building"],
        )

    class _BoxOps:
        @staticmethod
        def box_cxcywh_to_xyxy(boxes: Any) -> Any:
            return boxes

    class _SamTransform:
        @staticmethod
        def apply_boxes_torch(boxes: Any, image_size: Any) -> Any:
            return boxes

    class _Predictor:
        def __init__(self) -> None:
            self.transform = _SamTransform()
            self.set_image_calls = 0
            self.predict_calls = 0

        def set_image(self, image: Any) -> None:
            self.set_image_calls += 1
            assert image.shape == (4, 6, 3)

        def predict_torch(self, **kwargs: Any) -> tuple[Any, None, None]:
            mask = torch.zeros((1, 1, 4, 6), dtype=torch.bool)
            if self.predict_calls == 0:
                mask[:, :, 0:2, :] = True
            else:
                mask[:, :, 0, 0] = True
                mask[:, :, 3, 5] = True
            self.predict_calls += 1
            return mask, None, None

    predictor = _Predictor()
    backend = object.__new__(sky_cli.GroundedSamBackend)
    backend._np = np
    backend._torch = torch
    backend._transform = transform
    backend._box_ops = _BoxOps()
    backend._predict = predict
    backend._grounding_model = object()
    backend._sam_predictor = predictor
    backend._text_prompt = "sky"
    backend._negative_text_prompt = "building"
    backend._box_threshold = 0.3
    backend._text_threshold = 0.25
    backend._device = "cpu"

    result = backend.infer(image_path)

    assert transform_calls == [(6, 4)]
    assert predict_calls == [("sky", id(transformed)), ("building", id(transformed))]
    assert predictor.set_image_calls == 1
    assert predictor.predict_calls == 2
    assert result.mask.sum() == 11
    assert not result.mask[0, 0]
    assert result.mask[1, 0]
    assert result.detection_count == 1
    assert result.excluded_detection_count == 1
    assert result.excluded_phrases == ("building",)
    assert result.excluded_logits == pytest.approx((0.8,))


def test_manifest_records_and_resume_carries_exclusion_metadata(tmp_path: Path) -> None:
    root = _make_nuscenes(tmp_path / "nuscenes")
    output = tmp_path / "masks"
    args = _arguments(root, output, "--negative-text-prompt", "building")

    class _Backend:
        def __init__(self) -> None:
            self.calls = 0

        def infer(self, image_path: Path) -> Any:
            self.calls += 1
            mask = np.zeros((4, 6), dtype=np.bool_)
            mask[0, 0] = True
            return sky_cli.SkyInferenceResult(
                mask=mask,
                detection_count=1,
                phrases=("sky",),
                logits=(0.95,),
                excluded_detection_count=2,
                excluded_phrases=("building", "building"),
                excluded_logits=(0.8, 0.7),
            )

    backend = _Backend()
    generated = sky_cli.generate_sky_masks(args, backend=backend)

    assert backend.calls == 4
    assert generated["schema_version"] == 2
    assert generated["generation"]["negative_text_prompt"] == "building"
    assert generated["summary"]["frames_with_exclusions"] == 4
    first = generated["frames"][0]
    assert first["excluded_detection_count"] == 2
    assert first["excluded_phrases"] == ["building", "building"]
    assert first["excluded_logits"] == pytest.approx([0.8, 0.7])

    resumed_backend = _Backend()
    resumed = sky_cli.generate_sky_masks(args, backend=resumed_backend)

    assert resumed_backend.calls == 0
    resumed_first = resumed["frames"][0]
    assert resumed_first["status"] == "skipped_existing"
    assert resumed_first["excluded_detection_count"] == 2
    assert resumed_first["excluded_phrases"] == ["building", "building"]
    assert resumed_first["excluded_logits"] == pytest.approx([0.8, 0.7])


def test_scene_wrapper_defaults_building_tree_and_omits_empty_negative_prompt(
    tmp_path: Path,
) -> None:
    import os
    import subprocess

    root = _make_nuscenes(tmp_path / "nuscenes")
    fake_conda = tmp_path / "fake-conda"
    fake_conda.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "${ARG_LOG}"\n',
        encoding="utf-8",
    )
    fake_conda.chmod(0o755)
    wrapper = SCRIPT_PATH.with_name("generate_nuscenes_scene_0061_sky_masks.sh")
    base_env = {
        **os.environ,
        "CONDA_BIN": str(fake_conda),
        "NUSCENES_ROOT": str(root),
        "CAMERAS": ",".join(_CAMERAS),
        "OUTPUT_ROOT": str(tmp_path / "masks"),
        "DRY_RUN": "1",
        "SAVE_OVERLAYS": "0",
    }
    base_env.pop("NEGATIVE_TEXT_PROMPT", None)

    default_log = tmp_path / "default-args.txt"
    subprocess.run(
        ["bash", str(wrapper), "0"],
        check=True,
        env={**base_env, "ARG_LOG": str(default_log)},
        capture_output=True,
        text=True,
    )
    default_args = default_log.read_text(encoding="utf-8").splitlines()
    prompt_index = default_args.index("--negative-text-prompt")
    assert default_args[prompt_index + 1] == "building . tree"

    disabled_log = tmp_path / "disabled-args.txt"
    subprocess.run(
        ["bash", str(wrapper), "0"],
        check=True,
        env={
            **base_env,
            "ARG_LOG": str(disabled_log),
            "NEGATIVE_TEXT_PROMPT": "",
        },
        capture_output=True,
        text=True,
    )
    disabled_args = disabled_log.read_text(encoding="utf-8").splitlines()
    assert "--negative-text-prompt" not in disabled_args
