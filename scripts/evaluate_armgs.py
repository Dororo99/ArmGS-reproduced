#!/usr/bin/env python3
"""Evaluate ArmGS RGB tensors without image-library dependencies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

# Make a source checkout directly runnable without requiring an editable install.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from armgs.evaluation import EvaluationAccumulator, LPIPSUnavailableError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PSNR/SSIM/optional LPIPS and actor-mask PSNR from .pt tensors."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prediction", type=Path, help="prediction RGB tensor (.pt)")
    source.add_argument(
        "--manifest",
        type=Path,
        help="JSON manifest, or directory containing manifest.json",
    )
    parser.add_argument("--target", type=Path, help="target RGB tensor (.pt)")
    parser.add_argument("--valid-mask", type=Path, help="optional valid mask tensor (.pt)")
    parser.add_argument("--actor-mask", type=Path, help="optional actor mask tensor (.pt)")
    parser.add_argument("--data-range", type=float, default=1.0)
    parser.add_argument("--ssim-window-size", type=int, default=11)
    parser.add_argument("--ssim-sigma", type=float, default=1.5)
    parser.add_argument("--lpips", action="store_true", help="enable optional LPIPS")
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--output", type=Path, help="also write the JSON summary here")
    return parser.parse_args(argv)


def _torch_load(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"tensor file does not exist: {path}")
    try:
        try:
            # weights_only avoids unpickling arbitrary objects when supported by Torch.
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")
    except Exception as error:
        raise ValueError(f"failed to load tensor file {path}: {error}") from error


def _load_tensor(path: Path, *, role: str) -> torch.Tensor:
    value = _torch_load(path)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{role} file must contain one torch.Tensor: {path}")
    return value.cpu()


def _resolve_path(value: str | Path, *, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _directory_entries(payload: dict[str, Any], *, root: Path) -> list[dict[str, Path]]:
    if "prediction_dir" not in payload or "target_dir" not in payload:
        raise ValueError("manifest must contain 'pairs' or prediction_dir/target_dir")
    prediction_dir = _resolve_path(payload["prediction_dir"], root=root)
    target_dir = _resolve_path(payload["target_dir"], root=root)
    files = payload.get("files")
    if files is None:
        if not prediction_dir.is_dir() or not target_dir.is_dir():
            raise FileNotFoundError("manifest prediction_dir and target_dir must exist")
        prediction_files = {
            path.relative_to(prediction_dir).as_posix()
            for path in prediction_dir.rglob("*.pt")
        }
        target_files = {
            path.relative_to(target_dir).as_posix() for path in target_dir.rglob("*.pt")
        }
        if prediction_files != target_files:
            missing_predictions = sorted(target_files - prediction_files)
            missing_targets = sorted(prediction_files - target_files)
            raise ValueError(
                "prediction/target directory contents differ; "
                f"missing predictions={missing_predictions}, missing targets={missing_targets}"
            )
        files = sorted(prediction_files)
    if not isinstance(files, list) or not files:
        raise ValueError("manifest 'files' must be a non-empty list")

    valid_dir = (
        _resolve_path(payload["valid_mask_dir"], root=root)
        if payload.get("valid_mask_dir") is not None
        else None
    )
    actor_dir = (
        _resolve_path(payload["actor_mask_dir"], root=root)
        if payload.get("actor_mask_dir") is not None
        else None
    )
    entries: list[dict[str, Path]] = []
    for filename in files:
        if not isinstance(filename, str) or not filename:
            raise ValueError("every manifest filename must be a non-empty string")
        entry = {
            "prediction": prediction_dir / filename,
            "target": target_dir / filename,
        }
        if valid_dir is not None:
            entry["valid_mask"] = valid_dir / filename
        if actor_dir is not None:
            entry["actor_mask"] = actor_dir / filename
        entries.append(entry)
    return entries


def _manifest_entries(manifest: Path) -> list[dict[str, Path]]:
    manifest_path = manifest / "manifest.json" if manifest.is_dir() else manifest
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON manifest {manifest_path}: {error}") from error
    root = manifest_path.parent
    if isinstance(payload, list):
        raw_entries = payload
    elif isinstance(payload, dict) and "pairs" in payload:
        raw_entries = payload["pairs"]
    elif isinstance(payload, dict):
        return _directory_entries(payload, root=root)
    else:
        raise ValueError("manifest root must be a list or object")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("manifest pairs must be a non-empty list")

    entries: list[dict[str, Path]] = []
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"manifest pair {index} must be an object")
        missing = {"prediction", "target"} - raw_entry.keys()
        if missing:
            raise ValueError(f"manifest pair {index} is missing: {', '.join(sorted(missing))}")
        entry = {
            "prediction": _resolve_path(raw_entry["prediction"], root=root),
            "target": _resolve_path(raw_entry["target"], root=root),
        }
        for mask_name in ("valid_mask", "actor_mask"):
            if raw_entry.get(mask_name) is not None:
                entry[mask_name] = _resolve_path(raw_entry[mask_name], root=root)
        entries.append(entry)
    return entries


def _input_entries(args: argparse.Namespace) -> list[dict[str, Path]]:
    if args.prediction is not None:
        if args.target is None:
            raise ValueError("--target is required with --prediction")
        return [
            {
                "prediction": args.prediction,
                "target": args.target,
                **({"valid_mask": args.valid_mask} if args.valid_mask else {}),
                **({"actor_mask": args.actor_mask} if args.actor_mask else {}),
            }
        ]
    if args.target is not None or args.valid_mask is not None or args.actor_mask is not None:
        raise ValueError("--target/--valid-mask/--actor-mask cannot be combined with --manifest")
    return _manifest_entries(args.manifest)


def run(args: argparse.Namespace) -> dict[str, Any]:
    accumulator = EvaluationAccumulator(
        data_range=args.data_range,
        ssim_window_size=args.ssim_window_size,
        ssim_sigma=args.ssim_sigma,
        compute_lpips=args.lpips,
        lpips_net=args.lpips_net,
        lpips_device="cpu",
    )
    entries = _input_entries(args)
    for entry in entries:
        accumulator.update(
            _load_tensor(entry["prediction"], role="prediction"),
            _load_tensor(entry["target"], role="target"),
            valid_mask=(
                _load_tensor(entry["valid_mask"], role="valid mask")
                if "valid_mask" in entry
                else None
            ),
            actor_mask=(
                _load_tensor(entry["actor_mask"], role="actor mask")
                if "actor_mask" in entry
                else None
            ),
        )
    return accumulator.summary()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run(args)
        serialized = json.dumps(summary, indent=2, sort_keys=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n", encoding="utf-8")
        print(serialized)
        return 0
    except (
        FileNotFoundError,
        LPIPSUnavailableError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
