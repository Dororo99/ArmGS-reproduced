from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

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
