#!/usr/bin/env python3
"""Generate StreetGS-compatible Waymo FRONT sky masks with Grounded SAM.

The canonical Waymo loader decodes only camera components for this job
(require_lidar=False). Training masks follow its exact lookup convention:
<output-root>/<sequence>/FRONT/<source-frame-index:08d>.png.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Protocol, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for import_root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from generate_nuscenes_sky_masks import (  # noqa: E402
    GroundedSamBackend,
    _atomic_write_json,
    _evenly_sample,
    _normalized_binary_mask,
    _overlay_image,
    inspect_binary_mask_png,
    union_instance_masks,
    write_binary_mask_png,
)


_DEFAULT_GSAM_ROOT = REPOSITORY_ROOT / "third_party" / "Grounded-Segment-Anything"
_DEFAULT_CHECKPOINT_ROOT = REPOSITORY_ROOT / "checkpoints" / "grounded_sam"
_MANIFEST_SCHEMA_VERSION = 1
_FRAME_MANIFEST_SCHEMA_VERSION = 1
_FRONT_CHANNEL = "FRONT"
_FRONT_CANONICAL_CAMERA_ID = 0
_PAPER_HEIGHT = 1066
_PAPER_WIDTH = 1600
_WAYMO_FRONT_SOURCE_HEIGHT = 1280
_STREETGS_TOP_EDGE_PIXELS = 100


@dataclass(frozen=True)
class WaymoFrontFrame:
    """One decoded FRONT observation and its source-context frame index."""

    sequence: str
    source_frame_index: int
    frame_index: int
    channel: str
    image_path: Path
    image_size: tuple[int, int]

    @property
    def key(self) -> str:
        return f"{self.channel}/{self.source_frame_index:08d}"


@dataclass(frozen=True)
class WaymoSkyInferenceResult:
    """Grounded-SAM result after the StreetGS top-edge box filter."""

    mask: Any
    candidate_detection_count: int
    detection_count: int
    top_edge_rejected_count: int
    phrases: tuple[str, ...] = ()
    logits: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for value, label in (
            (self.candidate_detection_count, "candidate_detection_count"),
            (self.detection_count, "detection_count"),
            (self.top_edge_rejected_count, "top_edge_rejected_count"),
        ):
            if value < 0:
                raise ValueError(f"{label} must be non-negative")
        if self.candidate_detection_count != (
            self.detection_count + self.top_edge_rejected_count
        ):
            raise ValueError("candidate detections must equal accepted plus rejected")
        if self.phrases and len(self.phrases) != self.detection_count:
            raise ValueError("phrases must align with accepted detections")
        if self.logits and len(self.logits) != self.detection_count:
            raise ValueError("logits must align with accepted detections")


class WaymoSkySegmentationBackend(Protocol):
    def infer(self, image_path: Path) -> WaymoSkyInferenceResult: ...


class WaymoManifestLoader(Protocol):
    def __call__(self, root: str | Path, **kwargs: Any) -> Any: ...


def _probability(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a floating-point value") from error
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be finite and in [0, 1]")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 1600x1066 Waymo FRONT sky masks using GroundingDINO "
            "(prompt='sky'), StreetGS top-edge filtering, and SAM."
        )
    )
    parser.add_argument("--waymo-root", type=Path, required=True)
    parser.add_argument(
        "--parquet-dir",
        default="validation",
        help="Waymo split directory below the root (default: validation)",
    )
    parser.add_argument("--sequence", required=True, help="Waymo context name")
    parser.add_argument("--start-frame", type=_non_negative_int, default=0)
    parser.add_argument("--end-frame", type=_non_negative_int)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="decoded RGB cache (default: <waymo-root>/armgs_cache)",
    )
    parser.add_argument("--target-height", type=_positive_int, default=_PAPER_HEIGHT)
    parser.add_argument("--target-width", type=_positive_int, default=_PAPER_WIDTH)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        help="generation JSON path (default: <output-root>/<sequence>/generation_manifest.json)",
    )
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument(
        "--prepare-manifest",
        type=Path,
        help=(
            "decode FRONT frames with the Waymo environment and atomically write "
            "a portable frame manifest; Grounded-SAM is not loaded"
        ),
    )
    stage.add_argument(
        "--input-frame-manifest",
        type=Path,
        help=(
            "read decoded FRONT frames from a prepare-stage manifest without "
            "importing the Waymo SDK"
        ),
    )

    parser.add_argument(
        "--groundingdino-config",
        type=Path,
        default=(
            _DEFAULT_GSAM_ROOT
            / "GroundingDINO"
            / "groundingdino"
            / "config"
            / "GroundingDINO_SwinT_OGC.py"
        ),
    )
    parser.add_argument(
        "--groundingdino-checkpoint",
        type=Path,
        default=_DEFAULT_CHECKPOINT_ROOT / "groundingdino_swint_ogc.pth",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=_DEFAULT_CHECKPOINT_ROOT / "sam_vit_h_4b8939.pth",
    )
    parser.add_argument(
        "--sam-model-type",
        choices=("vit_h", "vit_l", "vit_b"),
        default="vit_h",
    )
    parser.add_argument(
        "--bert-path",
        type=Path,
        default=_DEFAULT_CHECKPOINT_ROOT / "bert-base-uncased",
        help="offline BERT snapshot",
    )
    parser.add_argument("--text-prompt", default="sky")
    parser.add_argument("--box-threshold", type=_probability, default=0.3)
    parser.add_argument("--text-threshold", type=_probability, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--source-image-height",
        type=_positive_int,
        default=_WAYMO_FRONT_SOURCE_HEIGHT,
        help="original Waymo FRONT height used to scale the top-edge rule",
    )
    parser.add_argument(
        "--top-edge-original-pixels",
        type=_positive_int,
        default=_STREETGS_TOP_EDGE_PIXELS,
        help="accept boxes starting inside this many original top pixels",
    )

    existing = parser.add_mutually_exclusive_group()
    existing.add_argument(
        "--resume",
        dest="existing_mode",
        action="store_const",
        const="resume",
        help="skip valid existing masks (default)",
    )
    existing.add_argument(
        "--overwrite",
        dest="existing_mode",
        action="store_const",
        const="overwrite",
        help="regenerate valid existing masks",
    )
    parser.set_defaults(existing_mode="resume")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="enumerate decoded inputs/outputs without loading models or writing masks",
    )
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument(
        "--overlay-root",
        type=Path,
        help="overlay directory (default: <output-root>/<sequence>/overlays)",
    )
    parser.add_argument("--overlay-every", type=_positive_int, default=1)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--contact-sheet-max-items", type=_positive_int, default=48)
    return parser.parse_args(argv)


def _load_waymo_v2_manifest(root: str | Path, **kwargs: Any) -> Any:
    from armgs.data.waymo import load_waymo_v2_manifest

    return load_waymo_v2_manifest(root, **kwargs)


def enumerate_waymo_front_frames(
    root: str | Path,
    *,
    sequence: str,
    parquet_dir: str = "validation",
    start_frame: int = 0,
    end_frame: int | None = None,
    target_size: tuple[int, int] = (_PAPER_HEIGHT, _PAPER_WIDTH),
    cache_dir: str | Path | None = None,
    manifest_loader: WaymoManifestLoader | None = None,
) -> tuple[WaymoFrontFrame, ...]:
    """Decode and enumerate FRONT frames through the canonical camera-only loader."""

    if not sequence or Path(sequence).name != sequence:
        raise ValueError("sequence must be one safe non-empty context name")
    loader = _load_waymo_v2_manifest if manifest_loader is None else manifest_loader
    manifest = loader(
        root,
        sequence=sequence,
        parquet_dir=parquet_dir,
        camera_channels=(_FRONT_CHANNEL,),
        start_frame=start_frame,
        end_frame=end_frame,
        target_size=target_size,
        cache_dir=cache_dir,
        sky_mask_root=None,
        require_lidar=False,
    )

    frames: list[WaymoFrontFrame] = []
    source_indices: set[int] = set()
    for canonical in manifest.frames:
        if canonical.camera_id != _FRONT_CANONICAL_CAMERA_ID:
            raise ValueError("Waymo sky generation received a non-FRONT frame")
        if canonical.image_size != target_size:
            raise ValueError(
                f"Waymo FRONT frame has size {canonical.image_size}, expected {target_size}"
            )
        relative_index = int(canonical.frame_index)
        if relative_index < 0:
            raise ValueError("Waymo frame_index must be non-negative")
        source_index = start_frame + relative_index
        if source_index in source_indices:
            raise ValueError(f"duplicate Waymo FRONT source index: {source_index}")
        source_indices.add(source_index)
        image_path = Path(canonical.image_path)
        expected_name = f"{source_index:08d}.png"
        if image_path.name != expected_name:
            raise ValueError(
                f"Waymo cache image {image_path.name!r} does not match "
                f"source frame {expected_name!r}"
            )
        if not image_path.is_file():
            raise FileNotFoundError(f"missing decoded Waymo FRONT image: {image_path}")
        frames.append(
            WaymoFrontFrame(
                sequence=sequence,
                source_frame_index=source_index,
                frame_index=relative_index,
                channel=_FRONT_CHANNEL,
                image_path=image_path,
                image_size=target_size,
            )
        )
    if not frames:
        raise ValueError("Waymo loader returned no FRONT frames")
    frames.sort(key=lambda frame: frame.source_frame_index)
    return tuple(frames)


def _frame_manifest_dataset(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "waymo_root": str(args.waymo_root.resolve()),
        "parquet_dir": args.parquet_dir,
        "sequence": args.sequence,
        "camera_channels": [_FRONT_CHANNEL],
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "target_size": [args.target_height, args.target_width],
        "cache_dir": (
            str(args.cache_dir.resolve()) if args.cache_dir is not None else None
        ),
        "require_lidar": False,
    }


def _prepared_frame_record(frame: WaymoFrontFrame) -> dict[str, Any]:
    image_path = frame.image_path.resolve()
    image_stat = image_path.stat()
    return {
        "frame_key": frame.key,
        "sequence": frame.sequence,
        "source_frame_index": frame.source_frame_index,
        "frame_index": frame.frame_index,
        "channel": frame.channel,
        "image_path": str(image_path),
        "image_size": list(frame.image_size),
        "image_stat": {
            "size_bytes": image_stat.st_size,
            "mtime_ns": image_stat.st_mtime_ns,
        },
    }


def prepare_waymo_frame_manifest(
    args: argparse.Namespace,
    *,
    manifest_loader: WaymoManifestLoader | None = None,
) -> dict[str, Any]:
    """Decode FRONT images and atomically publish an inference-stage manifest."""

    if args.prepare_manifest is None:
        raise ValueError("--prepare-manifest is required for the prepare stage")
    if args.input_frame_manifest is not None:
        raise ValueError("prepare and input frame manifests are mutually exclusive")
    if args.dry_run:
        raise ValueError("--dry-run cannot be combined with --prepare-manifest")
    if args.end_frame is not None and args.end_frame < args.start_frame:
        raise ValueError("end_frame cannot be smaller than start_frame")
    frames = enumerate_waymo_front_frames(
        args.waymo_root,
        sequence=args.sequence,
        parquet_dir=args.parquet_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        target_size=(args.target_height, args.target_width),
        cache_dir=args.cache_dir,
        manifest_loader=manifest_loader,
    )
    payload = {
        "schema_version": _FRAME_MANIFEST_SCHEMA_VERSION,
        "kind": "armgs_waymo_front_frame_manifest",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": _frame_manifest_dataset(args),
        "frames": [_prepared_frame_record(frame) for frame in frames],
        "summary": {"total": len(frames)},
    }
    _atomic_write_json(args.prepare_manifest, payload)
    return payload


def _required_manifest_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def load_prepared_waymo_front_frames(
    manifest_path: str | Path,
    *,
    args: argparse.Namespace,
) -> tuple[WaymoFrontFrame, ...]:
    """Load and revalidate decoded frames without importing the Waymo SDK."""

    path = Path(manifest_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read prepared frame manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("prepared frame manifest root must be an object")
    if payload.get("schema_version") != _FRAME_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported prepared frame manifest schema_version")
    if payload.get("kind") != "armgs_waymo_front_frame_manifest":
        raise ValueError("invalid prepared frame manifest kind")

    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("prepared frame manifest dataset must be an object")
    expected_dataset = _frame_manifest_dataset(args)
    for key in (
        "waymo_root",
        "parquet_dir",
        "sequence",
        "camera_channels",
        "start_frame",
        "end_frame",
        "target_size",
        "require_lidar",
    ):
        if dataset.get(key) != expected_dataset[key]:
            raise ValueError(
                f"prepared frame manifest dataset.{key} does not match CLI arguments"
            )

    records = payload.get("frames")
    if not isinstance(records, list) or not records:
        raise ValueError("prepared frame manifest must contain at least one frame")
    summary = payload.get("summary")
    if not isinstance(summary, dict) or summary.get("total") != len(records):
        raise ValueError("prepared frame manifest summary.total is inconsistent")

    from PIL import Image

    frames: list[WaymoFrontFrame] = []
    seen_sources: set[int] = set()
    target_size = (args.target_height, args.target_width)
    for record_index, record in enumerate(records):
        label = f"frames[{record_index}]"
        if not isinstance(record, dict):
            raise ValueError(f"{label} must be an object")
        source_index = _required_manifest_int(
            record.get("source_frame_index"), f"{label}.source_frame_index"
        )
        frame_index = _required_manifest_int(
            record.get("frame_index"), f"{label}.frame_index"
        )
        if source_index in seen_sources:
            raise ValueError(f"duplicate prepared source frame index: {source_index}")
        seen_sources.add(source_index)
        if source_index != args.start_frame + frame_index:
            raise ValueError(f"{label} has an inconsistent absolute source index")
        if record.get("sequence") != args.sequence:
            raise ValueError(f"{label}.sequence does not match the requested sequence")
        if record.get("channel") != _FRONT_CHANNEL:
            raise ValueError(f"{label}.channel must be FRONT")
        expected_key = f"{_FRONT_CHANNEL}/{source_index:08d}"
        if record.get("frame_key") != expected_key:
            raise ValueError(f"{label}.frame_key is inconsistent")
        if record.get("image_size") != list(target_size):
            raise ValueError(f"{label}.image_size does not match the target size")

        raw_image_path = record.get("image_path")
        if not isinstance(raw_image_path, str) or not raw_image_path:
            raise ValueError(f"{label}.image_path must be a non-empty string")
        image_path = Path(raw_image_path)
        if not image_path.is_absolute():
            raise ValueError(f"{label}.image_path must be absolute")
        resolved_image_path = image_path.resolve()
        if str(resolved_image_path) != raw_image_path:
            raise ValueError(f"{label}.image_path must be normalized")
        if resolved_image_path.name != f"{source_index:08d}.png":
            raise ValueError(f"{label}.image_path filename has the wrong source index")
        if not resolved_image_path.is_file():
            raise ValueError(f"{label}.image_path is missing: {resolved_image_path}")

        recorded_stat = record.get("image_stat")
        if not isinstance(recorded_stat, dict):
            raise ValueError(f"{label}.image_stat must be an object")
        actual_stat = resolved_image_path.stat()
        if recorded_stat.get("size_bytes") != actual_stat.st_size:
            raise ValueError(f"{label}.image_stat.size_bytes changed after prepare")
        if recorded_stat.get("mtime_ns") != actual_stat.st_mtime_ns:
            raise ValueError(f"{label}.image_stat.mtime_ns changed after prepare")
        try:
            with Image.open(resolved_image_path) as image:
                actual_size = (image.height, image.width)
                image.load()
        except (OSError, ValueError) as error:
            raise ValueError(f"{label}.image_path is not a valid image: {error}") from error
        if actual_size != target_size:
            raise ValueError(
                f"{label} decoded size {actual_size} does not match {target_size}"
            )
        frames.append(
            WaymoFrontFrame(
                sequence=args.sequence,
                source_frame_index=source_index,
                frame_index=frame_index,
                channel=_FRONT_CHANNEL,
                image_path=resolved_image_path,
                image_size=target_size,
            )
        )

    actual_sources = [frame.source_frame_index for frame in frames]
    if args.end_frame is None:
        expected_sources = list(
            range(args.start_frame, args.start_frame + len(frames))
        )
    else:
        expected_sources = list(range(args.start_frame, args.end_frame + 1))
    if actual_sources != expected_sources:
        raise ValueError(
            "prepared frames are not the complete ordered requested source range"
        )
    return tuple(frames)


def mask_output_path(output_root: str | Path, *, frame: WaymoFrontFrame) -> Path:
    return (
        Path(output_root)
        / frame.sequence
        / frame.channel
        / f"{frame.source_frame_index:08d}.png"
    )


def scaled_top_edge_pixels(
    *,
    target_height: int,
    source_height: int = _WAYMO_FRONT_SOURCE_HEIGHT,
    original_pixels: int = _STREETGS_TOP_EDGE_PIXELS,
) -> float:
    """Scale the StreetGS top-100 source-pixel rule to a resized image."""

    if min(target_height, source_height, original_pixels) <= 0:
        raise ValueError("top-edge scale dimensions must be positive")
    if original_pixels > source_height:
        raise ValueError("top-edge original pixels cannot exceed source height")
    return float(original_pixels) * float(target_height) / float(source_height)


def top_edge_acceptance_mask(
    boxes_cxcywh: Any,
    *,
    target_height: int,
    source_height: int = _WAYMO_FRONT_SOURCE_HEIGHT,
    original_pixels: int = _STREETGS_TOP_EDGE_PIXELS,
) -> Any:
    """Return a boolean mask for normalized boxes starting in the top band."""

    if getattr(boxes_cxcywh, "ndim", None) != 2 or boxes_cxcywh.shape[1] != 4:
        raise ValueError("GroundingDINO boxes must have shape [N,4]")
    if source_height <= 0 or original_pixels <= 0 or original_pixels > source_height:
        raise ValueError("invalid StreetGS top-edge filter dimensions")
    top_edges = boxes_cxcywh[:, 1] - 0.5 * boxes_cxcywh[:, 3]
    scaled_limit = scaled_top_edge_pixels(
        target_height=target_height,
        source_height=source_height,
        original_pixels=original_pixels,
    )
    return top_edges * float(target_height) < scaled_limit


class WaymoGroundedSamBackend(GroundedSamBackend):
    """Grounded SAM with StreetGS' Waymo top-edge detection filter."""

    def __init__(
        self,
        *,
        source_image_height: int = _WAYMO_FRONT_SOURCE_HEIGHT,
        top_edge_original_pixels: int = _STREETGS_TOP_EDGE_PIXELS,
        **kwargs: Any,
    ) -> None:
        scaled_top_edge_pixels(
            target_height=_PAPER_HEIGHT,
            source_height=source_image_height,
            original_pixels=top_edge_original_pixels,
        )
        super().__init__(negative_text_prompt=None, **kwargs)
        self._source_image_height = source_image_height
        self._top_edge_original_pixels = top_edge_original_pixels

    def infer(self, image_path: Path) -> WaymoSkyInferenceResult:
        from PIL import Image

        with Image.open(image_path) as source_image:
            rgb_image = source_image.convert("RGB")
            image_source = self._np.asarray(rgb_image).copy()
            transformed, _ = self._transform(rgb_image, None)
        height, width = image_source.shape[:2]
        boxes, logits, phrases = self._predict(
            model=self._grounding_model,
            image=transformed,
            caption=self._text_prompt,
            box_threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            device=self._device,
        )
        candidate_count = int(boxes.shape[0])
        accepted = top_edge_acceptance_mask(
            boxes,
            target_height=height,
            source_height=self._source_image_height,
            original_pixels=self._top_edge_original_pixels,
        )
        accepted_boxes = boxes[accepted]
        accepted_logits = logits[accepted]
        accepted_flags = accepted.detach().cpu().tolist()
        accepted_phrases = tuple(
            str(phrase)
            for phrase, keep in zip(phrases, accepted_flags)
            if bool(keep)
        )
        detection_count = int(accepted_boxes.shape[0])

        if detection_count == 0:
            mask = self._np.zeros((height, width), dtype=self._np.bool_)
        else:
            self._sam_predictor.set_image(image_source)
            scale = self._torch.tensor(
                [width, height, width, height],
                dtype=accepted_boxes.dtype,
                device=accepted_boxes.device,
            )
            boxes_xyxy = self._box_ops.box_cxcywh_to_xyxy(accepted_boxes) * scale
            transformed_boxes = self._sam_predictor.transform.apply_boxes_torch(
                boxes_xyxy, image_source.shape[:2]
            ).to(self._device)
            instance_masks, _, _ = self._sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            mask = union_instance_masks(
                instance_masks.detach().cpu().numpy(),
                (height, width),
            )

        return WaymoSkyInferenceResult(
            mask=mask,
            candidate_detection_count=candidate_count,
            detection_count=detection_count,
            top_edge_rejected_count=candidate_count - detection_count,
            phrases=accepted_phrases,
            logits=tuple(
                float(value) for value in accepted_logits.detach().cpu().tolist()
            ),
        )


def _save_overlay(
    frame: WaymoFrontFrame,
    mask_path: Path,
    overlay_path: Path,
    coverage: float,
) -> None:
    overlay = _overlay_image(
        frame.image_path,
        mask_path,
        f"FRONT source={frame.source_frame_index} sky={coverage:.3f}",
    )
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = overlay_path.with_name(f".{overlay_path.name}.tmp")
    overlay.save(temporary, format="JPEG", quality=90)
    temporary.replace(overlay_path)


def save_contact_sheet(
    entries: Sequence[tuple[WaymoFrontFrame, Path, float]],
    output_path: Path,
    *,
    max_items: int = 48,
) -> None:
    """Save evenly sampled Waymo RGB/mask overlays for manual QA."""

    from PIL import Image, ImageOps

    if not entries:
        return
    selected = _evenly_sample(entries, min(max_items, len(entries)))
    tile_size = (320, 213)
    columns = min(4, len(selected))
    rows = math.ceil(len(selected) / columns)
    sheet = Image.new("RGB", (columns * tile_size[0], rows * tile_size[1]), "black")
    for index, (frame, mask_path, coverage) in enumerate(selected):
        overlay = _overlay_image(
            frame.image_path,
            mask_path,
            f"FRONT source={frame.source_frame_index} sky={coverage:.3f}",
        )
        thumbnail = ImageOps.fit(
            overlay,
            tile_size,
            method=Image.Resampling.LANCZOS,
        )
        sheet.paste(
            thumbnail,
            ((index % columns) * tile_size[0], (index // columns) * tile_size[1]),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    image_format = "PNG" if output_path.suffix.lower() == ".png" else "JPEG"
    sheet.save(temporary, format=image_format, quality=90)
    temporary.replace(output_path)


def _new_manifest(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _frame_manifest_dataset(args)
    dataset["input_frame_manifest"] = (
        str(args.input_frame_manifest.resolve())
        if args.input_frame_manifest is not None
        else None
    )
    return {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "generation": {
            "method": (
                "GroundingDINO sky boxes with StreetGS top-edge filter, "
                "unioned SAM masks"
            ),
            "text_prompt": args.text_prompt,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "source_image_height": args.source_image_height,
            "top_edge_original_pixels": args.top_edge_original_pixels,
            "scaled_top_edge_pixels": scaled_top_edge_pixels(
                target_height=args.target_height,
                source_height=args.source_image_height,
                original_pixels=args.top_edge_original_pixels,
            ),
            "device": args.device,
            "groundingdino_config": str(args.groundingdino_config),
            "groundingdino_checkpoint": str(args.groundingdino_checkpoint),
            "sam_checkpoint": str(args.sam_checkpoint),
            "sam_model_type": args.sam_model_type,
            "bert_path": str(args.bert_path) if args.bert_path is not None else None,
        },
        "frames": [],
        "summary": {},
    }


def _result_counts(result: Any) -> tuple[int, int, int]:
    accepted = int(result.detection_count)
    candidate = int(getattr(result, "candidate_detection_count", accepted))
    rejected = int(getattr(result, "top_edge_rejected_count", candidate - accepted))
    if min(candidate, accepted, rejected) < 0 or candidate != accepted + rejected:
        raise ValueError("backend returned inconsistent detection counts")
    return candidate, accepted, rejected


def _frame_record(
    frame: WaymoFrontFrame,
    mask_path: Path,
    *,
    status: str,
    coverage: float | None,
    candidate_detection_count: int | None,
    detection_count: int | None,
    top_edge_rejected_count: int | None,
    phrases: Sequence[str] = (),
    logits: Sequence[float] = (),
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "frame_key": frame.key,
        "source_frame_index": frame.source_frame_index,
        "frame_index": frame.frame_index,
        "channel": frame.channel,
        "image_path": str(frame.image_path.resolve()),
        "image_size": list(frame.image_size),
        "mask_path": str(mask_path.resolve()),
        "status": status,
        "candidate_detection_count": candidate_detection_count,
        "detection_count": detection_count,
        "top_edge_rejected_count": top_edge_rejected_count,
        "no_detection": (
            detection_count == 0 if detection_count is not None else coverage == 0.0
        ),
        "coverage": coverage,
        "phrases": list(phrases),
        "logits": [float(value) for value in logits],
        "error": error,
    }


def _update_summary(payload: dict[str, Any]) -> dict[str, Any]:
    frames = payload["frames"]
    statuses: dict[str, int] = {}
    coverages: list[float] = []
    for frame in frames:
        status = str(frame["status"])
        statuses[status] = statuses.get(status, 0) + 1
        if frame["coverage"] is not None:
            coverages.append(float(frame["coverage"]))

    def sum_counts(name: str) -> int:
        return sum(
            int(frame[name])
            for frame in frames
            if isinstance(frame.get(name), int)
        )

    payload["summary"] = {
        "total": len(frames),
        "statuses": statuses,
        "no_detection": sum(frame.get("no_detection") is True for frame in frames),
        "candidate_detections": sum_counts("candidate_detection_count"),
        "accepted_detections": sum_counts("detection_count"),
        "top_edge_rejected_detections": sum_counts("top_edge_rejected_count"),
        "mean_coverage": sum(coverages) / len(coverages) if coverages else None,
        "min_coverage": min(coverages) if coverages else None,
        "max_coverage": max(coverages) if coverages else None,
    }
    return payload


def _make_backend(args: argparse.Namespace) -> WaymoGroundedSamBackend:
    return WaymoGroundedSamBackend(
        groundingdino_config=args.groundingdino_config,
        groundingdino_checkpoint=args.groundingdino_checkpoint,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        bert_path=args.bert_path,
        text_prompt=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
        source_image_height=args.source_image_height,
        top_edge_original_pixels=args.top_edge_original_pixels,
    )


def generate_waymo_sky_masks(
    args: argparse.Namespace,
    *,
    backend: WaymoSkySegmentationBackend | None = None,
    manifest_loader: WaymoManifestLoader | None = None,
) -> dict[str, Any]:
    """Generate or resume one Waymo sequence and return its manifest payload."""

    if args.prepare_manifest is not None:
        raise ValueError(
            "--prepare-manifest is a decode-only stage; use main() or "
            "prepare_waymo_frame_manifest()"
        )
    if args.end_frame is not None and args.end_frame < args.start_frame:
        raise ValueError("end_frame cannot be smaller than start_frame")
    if not args.text_prompt.strip():
        raise ValueError("text_prompt cannot be empty")
    scaled_top_edge_pixels(
        target_height=args.target_height,
        source_height=args.source_image_height,
        original_pixels=args.top_edge_original_pixels,
    )
    if args.input_frame_manifest is not None:
        frames = load_prepared_waymo_front_frames(
            args.input_frame_manifest,
            args=args,
        )
    else:
        frames = enumerate_waymo_front_frames(
            args.waymo_root,
            sequence=args.sequence,
            parquet_dir=args.parquet_dir,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            target_size=(args.target_height, args.target_width),
            cache_dir=args.cache_dir,
            manifest_loader=manifest_loader,
        )
    sequence_output = args.output_root / args.sequence
    manifest_path = (
        args.manifest_path
        if args.manifest_path is not None
        else sequence_output / "generation_manifest.json"
    )
    overlay_root = (
        args.overlay_root
        if args.overlay_root is not None
        else sequence_output / "overlays"
    )
    payload = _new_manifest(args)

    if args.dry_run:
        for frame in frames:
            output_path = mask_output_path(args.output_root, frame=frame)
            payload["frames"].append(
                _frame_record(
                    frame,
                    output_path,
                    status=(
                        "dry_run_existing"
                        if output_path.is_file()
                        else "dry_run_pending"
                    ),
                    coverage=None,
                    candidate_detection_count=None,
                    detection_count=None,
                    top_edge_rejected_count=None,
                )
            )
        return _update_summary(payload)

    previous_records: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file() and args.existing_mode == "resume":
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            previous_records = {
                str(record["frame_key"]): record
                for record in previous.get("frames", ())
                if isinstance(record, dict) and "frame_key" in record
            }
        except (OSError, json.JSONDecodeError, TypeError, KeyError):
            previous_records = {}

    model = backend
    overlay_entries: list[tuple[WaymoFrontFrame, Path, float]] = []
    for index, frame in enumerate(frames):
        output_path = mask_output_path(args.output_root, frame=frame)
        if output_path.is_file() and args.existing_mode == "resume":
            try:
                coverage = inspect_binary_mask_png(output_path, frame.image_size)
            except ValueError:
                pass
            else:
                old = previous_records.get(frame.key, {})

                def old_count(name: str) -> int | None:
                    value = old.get(name)
                    return value if isinstance(value, int) else None

                payload["frames"].append(
                    _frame_record(
                        frame,
                        output_path,
                        status="skipped_existing",
                        coverage=coverage,
                        candidate_detection_count=old_count(
                            "candidate_detection_count"
                        ),
                        detection_count=old_count("detection_count"),
                        top_edge_rejected_count=old_count(
                            "top_edge_rejected_count"
                        ),
                        phrases=old.get("phrases", ()) or (),
                        logits=old.get("logits", ()) or (),
                    )
                )
                overlay_entries.append((frame, output_path, coverage))
                continue

        if model is None:
            model = _make_backend(args)
        try:
            result = model.infer(frame.image_path)
            candidate_count, detection_count, rejected_count = _result_counts(result)
            binary = _normalized_binary_mask(result.mask, frame.image_size)
            coverage = write_binary_mask_png(binary, frame.image_size, output_path)
            payload["frames"].append(
                _frame_record(
                    frame,
                    output_path,
                    status="no_detection" if detection_count == 0 else "generated",
                    coverage=coverage,
                    candidate_detection_count=candidate_count,
                    detection_count=detection_count,
                    top_edge_rejected_count=rejected_count,
                    phrases=result.phrases,
                    logits=result.logits,
                )
            )
            overlay_entries.append((frame, output_path, coverage))
            if args.save_overlays and index % args.overlay_every == 0:
                _save_overlay(
                    frame,
                    output_path,
                    overlay_root / frame.channel / f"{frame.source_frame_index:08d}.jpg",
                    coverage,
                )
        except Exception as error:
            payload["frames"].append(
                _frame_record(
                    frame,
                    output_path,
                    status="error",
                    coverage=None,
                    candidate_detection_count=None,
                    detection_count=None,
                    top_edge_rejected_count=None,
                    error=f"{type(error).__name__}: {error}",
                )
            )
            _atomic_write_json(manifest_path, _update_summary(payload))
            raise
        _atomic_write_json(manifest_path, _update_summary(payload))

    if args.contact_sheet is not None:
        save_contact_sheet(
            overlay_entries,
            args.contact_sheet,
            max_items=args.contact_sheet_max_items,
        )
    _atomic_write_json(manifest_path, _update_summary(payload))
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.prepare_manifest is not None:
        payload = prepare_waymo_frame_manifest(args)
        print(json.dumps(payload["summary"], indent=2, sort_keys=True))
        print(f"prepared frame manifest: {args.prepare_manifest}")
        return 0
    payload = generate_waymo_sky_masks(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    if args.dry_run:
        print("dry-run: models and mask outputs were not written")
    else:
        manifest_path = args.manifest_path or (
            args.output_root / args.sequence / "generation_manifest.json"
        )
        print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
