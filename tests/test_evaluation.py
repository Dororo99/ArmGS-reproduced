from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from armgs.evaluation import (
    EvaluationAccumulator,
    LPIPSUnavailableError,
    evaluate_image_pair,
    peak_signal_noise_ratio,
)


def test_psnr_known_value_and_exact_match() -> None:
    target = torch.zeros(5, 7, 3)
    prediction = torch.full_like(target, 0.5)
    score = peak_signal_noise_ratio(prediction, target)
    torch.testing.assert_close(score, torch.tensor(6.020599913, dtype=torch.float64))
    assert torch.isinf(peak_signal_noise_ratio(target, target))


def test_uint8_rgb_is_normalized_once_for_all_metric_paths() -> None:
    class RecordingLPIPS:
        prediction: torch.Tensor | None = None
        target: torch.Tensor | None = None
        data_range: float | None = None

        def __call__(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor,
            *,
            valid_mask: torch.Tensor | None,
            data_range: float,
        ) -> torch.Tensor:
            self.prediction = prediction.detach().clone()
            self.target = target.detach().clone()
            self.data_range = data_range
            assert valid_mask is not None
            return torch.tensor(0.125)

    uint8_prediction = torch.full((6, 7, 3), 128, dtype=torch.uint8)
    uint8_target = torch.zeros_like(uint8_prediction)
    normalized_prediction = uint8_prediction.to(torch.float32) / 255.0
    normalized_target = uint8_target.to(torch.float32)
    recorder = RecordingLPIPS()

    uint8_metrics = evaluate_image_pair(
        uint8_prediction,
        uint8_target,
        ssim_window_size=3,
        lpips_metric=recorder,
    )
    float_metrics = evaluate_image_pair(
        normalized_prediction,
        normalized_target,
        ssim_window_size=3,
    )

    assert uint8_metrics.psnr == pytest.approx(float_metrics.psnr)
    assert uint8_metrics.ssim == pytest.approx(float_metrics.ssim)
    assert uint8_metrics.lpips == pytest.approx(0.125)
    assert recorder.data_range == 1.0
    assert recorder.prediction is not None
    assert recorder.target is not None
    torch.testing.assert_close(
        recorder.prediction,
        normalized_prediction.permute(2, 0, 1).unsqueeze(0),
    )
    torch.testing.assert_close(
        recorder.target,
        normalized_target.permute(2, 0, 1).unsqueeze(0),
    )


def test_float_rgb_contract_uses_data_range_and_preserves_masked_padding() -> None:
    prediction = torch.full((5, 6, 3), 128.0)
    target = torch.zeros_like(prediction)
    scaled = evaluate_image_pair(
        prediction,
        target,
        data_range=255.0,
        ssim_window_size=3,
    )
    normalized = evaluate_image_pair(
        prediction / 255.0,
        target,
        ssim_window_size=3,
    )
    assert scaled.psnr == pytest.approx(normalized.psnr)
    assert scaled.ssim == pytest.approx(normalized.ssim)

    with pytest.raises(ValueError, match=r"prediction floating-point RGB values.*\[0, 1\]"):
        peak_signal_noise_ratio(torch.full((4, 5, 3), 1.01), torch.zeros(4, 5, 3))
    with pytest.raises(ValueError, match=r"target floating-point RGB values.*\[0, 255\]"):
        peak_signal_noise_ratio(
            torch.zeros(4, 5, 3),
            torch.full((4, 5, 3), 256.0),
            data_range=255.0,
        )
    with pytest.raises(ValueError, match="finite positive"):
        peak_signal_noise_ratio(target, target, data_range=float("nan"))

    padded_prediction = torch.zeros(4, 5, 3)
    padded_target = torch.zeros_like(padded_prediction)
    padded_prediction[0, 0] = 2.0
    padded_target[0, 0] = -1.0
    valid = torch.ones(4, 5, dtype=torch.bool)
    valid[0, 0] = False
    assert torch.isinf(
        peak_signal_noise_ratio(
            padded_prediction,
            padded_target,
            valid_mask=valid,
        )
    )


@pytest.mark.parametrize(
    "dtype",
    [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64],
)
def test_non_uint8_integer_rgb_is_rejected(dtype: torch.dtype) -> None:
    image = torch.zeros((4, 5, 3), dtype=dtype)
    with pytest.raises(TypeError, match="unsupported RGB dtype.*torch.uint8"):
        peak_signal_noise_ratio(image, image)


def test_valid_mask_excludes_invalid_values_and_non_finite_pixels() -> None:
    target = torch.zeros(4, 5, 3)
    prediction = target.clone()
    prediction[0, 0] = torch.tensor([float("nan"), float("nan"), float("nan")])
    valid = torch.ones(4, 5, dtype=torch.bool)
    valid[0, 0] = False

    score = peak_signal_noise_ratio(prediction, target, valid_mask=valid)
    assert torch.isinf(score)


def test_ssim_reuses_training_definition_and_actor_psnr_uses_intersection() -> None:
    target = torch.zeros(8, 9, 3)
    prediction = target.clone()
    prediction[2, 3] = 1.0
    actor = torch.zeros(8, 9, dtype=torch.bool)
    actor[2, 3] = True
    actor[0, 0] = True
    valid = torch.ones(8, 9, dtype=torch.bool)
    valid[0, 0] = False

    metrics = evaluate_image_pair(
        prediction, target, valid_mask=valid, actor_mask=actor
    )

    assert metrics.ssim < 1.0
    assert metrics.actor_pixels == 1
    assert metrics.actor_psnr == pytest.approx(0.0)
    assert metrics.valid_pixels == 71


def test_empty_actor_mask_is_reported_without_fabricating_a_score() -> None:
    image = torch.rand(6, 7, 3)
    metrics = evaluate_image_pair(image, image, actor_mask=torch.zeros(6, 7))

    assert metrics.actor_psnr is None
    assert metrics.actor_pixels == 0


def test_streaming_accumulator_handles_batches_and_is_json_serializable() -> None:
    accumulator = EvaluationAccumulator(ssim_window_size=3)
    target = torch.zeros(2, 6, 7, 3)
    prediction = torch.stack(
        (torch.full((6, 7, 3), 0.5), torch.full((6, 7, 3), 0.25))
    )
    actor = torch.zeros(2, 6, 7, dtype=torch.bool)
    actor[0, :2, :2] = True
    accumulator.update(prediction, target, actor_mask=actor)

    summary = accumulator.summary()
    assert summary["num_images"] == 2
    assert summary["num_actor_images"] == 1
    assert summary["valid_pixels"] == 2 * 6 * 7
    assert summary["actor_pixels"] == 4
    assert summary["psnr"] == pytest.approx((6.020599913 + 12.041199827) / 2)
    assert summary["actor_psnr"] == pytest.approx(6.020599913)
    json.dumps(summary, allow_nan=False)


def test_empty_accumulator_has_stable_json_schema() -> None:
    summary = EvaluationAccumulator().summary()
    assert summary["num_images"] == 0
    assert summary["psnr"] is None
    assert summary["ssim"] is None
    assert summary["lpips"] is None
    assert summary["actor_psnr"] is None
    json.dumps(summary, allow_nan=False)


def test_empty_valid_mask_is_rejected() -> None:
    image = torch.zeros(4, 5, 3)
    with pytest.raises(ValueError, match="selects no pixels"):
        evaluate_image_pair(image, image, valid_mask=torch.zeros(4, 5))


def test_lpips_missing_dependency_has_actionable_error() -> None:
    if importlib.util.find_spec("lpips") is not None:
        pytest.skip("optional lpips package is installed")
    with pytest.raises(LPIPSUnavailableError, match="optional 'lpips' package"):
        EvaluationAccumulator(compute_lpips=True)


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_armgs.py"
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_evaluates_tensor_pair_on_cpu(tmp_path: Path) -> None:
    prediction_path = tmp_path / "prediction.pt"
    target_path = tmp_path / "target.pt"
    actor_path = tmp_path / "actor.pt"
    output_path = tmp_path / "summary.json"
    torch.save(torch.full((5, 6, 3), 0.5), prediction_path)
    torch.save(torch.zeros(5, 6, 3), target_path)
    torch.save(torch.ones(5, 6, dtype=torch.bool), actor_path)

    result = _run_cli(
        "--prediction",
        str(prediction_path),
        "--target",
        str(target_path),
        "--actor-mask",
        str(actor_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["num_images"] == 1
    assert summary["psnr"] == pytest.approx(6.020599913)
    assert summary["actor_psnr"] == pytest.approx(6.020599913)
    assert json.loads(output_path.read_text()) == summary


def test_cli_normalizes_uint8_pt_and_rejects_signed_integer_rgb(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "uint8_prediction.pt"
    target_path = tmp_path / "uint8_target.pt"
    torch.save(torch.full((5, 6, 3), 128, dtype=torch.uint8), prediction_path)
    torch.save(torch.zeros((5, 6, 3), dtype=torch.uint8), target_path)

    result = _run_cli(
        "--prediction",
        str(prediction_path),
        "--target",
        str(target_path),
        "--ssim-window-size",
        "3",
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["psnr"] == pytest.approx(20.0 * math.log10(255.0 / 128.0))

    torch.save(torch.zeros((5, 6, 3), dtype=torch.int16), prediction_path)
    rejected = _run_cli(
        "--prediction",
        str(prediction_path),
        "--target",
        str(target_path),
    )
    assert rejected.returncode == 2
    assert "unsupported RGB dtype torch.int16" in rejected.stderr
    assert "expected torch.uint8 values in [0, 255]" in rejected.stderr


def test_cli_evaluates_png_jpeg_manifest_with_tensor_actor_mask(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "prediction.png"
    target_path = tmp_path / "target.jpeg"
    actor_path = tmp_path / "actor.pt"
    Image.new("RGB", (6, 5), (128, 128, 128)).save(prediction_path)
    Image.new("RGB", (6, 5), (0, 0, 0)).save(
        target_path, quality=100, subsampling=0
    )
    torch.save(torch.ones(5, 6, dtype=torch.bool), actor_path)
    manifest_path = tmp_path / "image_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "prediction": prediction_path.name,
                        "target": target_path.name,
                        "actor_mask": actor_path.name,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = _run_cli(
        "--manifest",
        str(manifest_path),
        "--ssim-window-size",
        "3",
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    expected_psnr = 20.0 * math.log10(255.0 / 128.0)
    assert summary["num_images"] == 1
    assert summary["psnr"] == pytest.approx(expected_psnr)
    assert summary["actor_psnr"] == pytest.approx(expected_psnr)


def test_cli_auto_discovers_rgb_images_in_directory_manifest(
    tmp_path: Path,
) -> None:
    prediction_dir = tmp_path / "predictions" / "nested"
    target_dir = tmp_path / "targets" / "nested"
    prediction_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    Image.new("RGB", (5, 4), (64, 64, 64)).save(
        prediction_dir / "000.PNG"
    )
    Image.new("RGB", (5, 4), (0, 0, 0)).save(target_dir / "000.PNG")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "prediction_dir": "predictions",
                "target_dir": "targets",
            }
        ),
        encoding="utf-8",
    )

    result = _run_cli(
        "--manifest", str(manifest_path), "--ssim-window-size", "3"
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["psnr"] == pytest.approx(
        20.0 * math.log10(255.0 / 64.0)
    )


def test_cli_accepts_directory_manifest_and_reports_missing_files(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    targets = tmp_path / "targets"
    predictions.mkdir()
    targets.mkdir()
    torch.save(torch.full((4, 5, 3), 0.25), predictions / "000.pt")
    torch.save(torch.zeros(4, 5, 3), targets / "000.pt")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "prediction_dir": "predictions",
                "target_dir": "targets",
                "files": ["000.pt"],
            }
        )
    )

    result = _run_cli("--manifest", str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["psnr"] == pytest.approx(12.041199827)

    missing = _run_cli(
        "--prediction", str(tmp_path / "missing.pt"), "--target", str(targets / "000.pt")
    )
    assert missing.returncode == 2
    assert "error: tensor file does not exist" in missing.stderr


def test_cli_lpips_error_is_clear_when_dependency_is_missing(tmp_path: Path) -> None:
    if importlib.util.find_spec("lpips") is not None:
        pytest.skip("optional lpips package is installed")
    image_path = tmp_path / "image.pt"
    torch.save(torch.zeros(4, 5, 3), image_path)

    result = _run_cli(
        "--prediction", str(image_path), "--target", str(image_path), "--lpips"
    )

    assert result.returncode == 2
    assert "optional 'lpips' package" in result.stderr
