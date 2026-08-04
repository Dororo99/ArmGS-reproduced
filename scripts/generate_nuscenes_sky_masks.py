#!/usr/bin/env python3
"""Generate raw-resolution nuScenes sky masks with Grounded SAM.

The script deliberately imports GroundingDINO and Segment Anything only when
inference starts.  Metadata enumeration, ``--dry-run``, and unit tests therefore
work in the ArmGS training environment without either third-party package.

Output masks are single-channel PNG files at the original image resolution:
``0`` denotes non-sky and ``255`` denotes sky.  Overlay images are optional QA
artifacts and are never used as training masks.
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
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from armgs.data.nuscenes import (  # noqa: E402
    NUSCENES_CAMERA_CHANNELS,
    _index_by_token,
    _load_table,
    _metadata_roots,
    _sample_chain,
    _select_records,
    _sensor_channels,
    _table_path,
    normalize_nuscenes_scene_name,
)


_DEFAULT_GSAM_ROOT = REPOSITORY_ROOT / "third_party" / "Grounded-Segment-Anything"
_DEFAULT_CHECKPOINT_ROOT = REPOSITORY_ROOT / "checkpoints" / "grounded_sam"
_MANIFEST_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class NuScenesCameraKeyframe:
    """One validated key camera sample selected from a nuScenes scene."""

    sample_token: str
    sample_data_token: str
    frame_index: int
    channel: str
    image_path: Path
    image_size: tuple[int, int]


@dataclass(frozen=True)
class SkyInferenceResult:
    """Backend-independent result for one source image."""

    mask: Any
    detection_count: int
    phrases: tuple[str, ...] = ()
    logits: tuple[float, ...] = ()
    excluded_detection_count: int = 0
    excluded_phrases: tuple[str, ...] = ()
    excluded_logits: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.detection_count < 0:
            raise ValueError("detection_count must be non-negative")
        if self.logits and len(self.logits) != self.detection_count:
            raise ValueError("logits must align with detections")
        if self.excluded_detection_count < 0:
            raise ValueError("excluded_detection_count must be non-negative")
        if (
            self.excluded_logits
            and len(self.excluded_logits) != self.excluded_detection_count
        ):
            raise ValueError("excluded_logits must align with excluded detections")


class SkySegmentationBackend(Protocol):
    def infer(self, image_path: Path) -> SkyInferenceResult: ...


def parse_camera_channels(value: str) -> tuple[str, ...]:
    """Parse ``all`` or a unique comma-separated nuScenes camera list."""

    stripped = value.strip()
    if stripped.lower() == "all":
        return tuple(NUSCENES_CAMERA_CHANNELS)
    channels = tuple(component.strip().upper() for component in stripped.split(","))
    if not channels or any(not channel for channel in channels):
        raise argparse.ArgumentTypeError(
            "cameras must be 'all' or comma-separated channel names"
        )
    unknown = set(channels) - set(NUSCENES_CAMERA_CHANNELS)
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown nuScenes camera channels: " + ", ".join(sorted(unknown))
        )
    if len(channels) != len(set(channels)):
        raise argparse.ArgumentTypeError("camera channels must be unique")
    return channels


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate original-resolution binary sky masks for one nuScenes "
            "scene using GroundingDINO(prompt='sky') and SAM box prompts."
        )
    )
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    parser.add_argument(
        "--version",
        default="v1.0-trainval",
        help="nuScenes metadata version (default: v1.0-trainval)",
    )
    parser.add_argument(
        "--scene",
        default="0061",
        help="scene selector such as 61, 0061, or scene-0061",
    )
    parser.add_argument(
        "--cameras",
        type=parse_camera_channels,
        default=tuple(NUSCENES_CAMERA_CHANNELS),
        help="all or comma-separated channels (default: all six cameras)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="mask root; version/scene/channel/token.png is appended",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        help="generation JSON path (default: scene output/generation_manifest.json)",
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
        help="optional local BERT model directory instead of bert-base-uncased",
    )
    parser.add_argument("--text-prompt", default="sky")
    parser.add_argument(
        "--negative-text-prompt",
        help=(
            "optional GroundingDINO prompt whose SAM mask is removed from the "
            "sky mask (disabled by default)"
        ),
    )
    parser.add_argument("--box-threshold", type=_probability, default=0.3)
    parser.add_argument("--text-threshold", type=_probability, default=0.25)
    parser.add_argument("--device", default="cuda:0")

    existing = parser.add_mutually_exclusive_group()
    existing.add_argument(
        "--resume",
        dest="existing_mode",
        action="store_const",
        const="resume",
        help="skip valid existing masks (this is the default)",
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
        help="enumerate inputs and planned outputs without loading models or writing",
    )

    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="save per-image QA overlays, separate from training masks",
    )
    parser.add_argument(
        "--overlay-root",
        type=Path,
        help="overlay directory (default: scene output/overlays)",
    )
    parser.add_argument(
        "--overlay-every",
        type=_positive_int,
        default=1,
        help="save every Nth selected frame when --save-overlays is set",
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        help="optional QA contact-sheet output (.jpg or .png)",
    )
    parser.add_argument(
        "--contact-sheet-max-items",
        type=_positive_int,
        default=48,
    )
    return parser.parse_args(argv)


def enumerate_nuscenes_camera_keyframes(
    root: str | Path,
    *,
    scene: int | str = "0061",
    version: str = "v1.0-trainval",
    camera_channels: Sequence[str] = NUSCENES_CAMERA_CHANNELS,
) -> tuple[NuScenesCameraKeyframe, ...]:
    """Enumerate validated keyframes using ArmGS's streaming metadata helpers."""

    data_root, metadata_root = _metadata_roots(root, version)
    scene_name = normalize_nuscenes_scene_name(scene)
    channels = tuple(camera_channels)
    if not channels or len(channels) != len(set(channels)):
        raise ValueError("camera_channels must be non-empty and unique")
    unknown = set(channels) - set(NUSCENES_CAMERA_CHANNELS)
    if unknown:
        raise ValueError(f"unknown nuScenes camera channels: {sorted(unknown)}")

    scene_records = _load_table(_table_path(metadata_root, "scene"))
    matches = [record for record in scene_records if record.get("name") == scene_name]
    if len(matches) != 1:
        raise ValueError(f"nuScenes scene not found or ambiguous: {scene_name}")
    samples_by_token = _index_by_token(
        _load_table(_table_path(metadata_root, "sample")), "sample"
    )
    samples = _sample_chain(matches[0], samples_by_token)
    sample_tokens = {str(sample["token"]) for sample in samples}

    sensors = _index_by_token(_load_table(_table_path(metadata_root, "sensor")), "sensor")
    calibrations = _index_by_token(
        _load_table(_table_path(metadata_root, "calibrated_sensor")),
        "calibrated_sensor",
    )
    calibration_channels = _sensor_channels(sensors, calibrations)
    selected_data = _select_records(
        _table_path(metadata_root, "sample_data"),
        field="sample_token",
        wanted=sample_tokens,
        keyframes_only=True,
    )

    records: dict[tuple[str, str], dict[str, Any]] = {}
    for record in selected_data:
        calibration_token = record.get("calibrated_sensor_token")
        if not isinstance(calibration_token, str):
            raise ValueError("sample_data is missing calibrated_sensor_token")
        try:
            channel = calibration_channels[calibration_token]
        except KeyError as error:
            raise ValueError("sample_data refers to missing calibrated_sensor") from error
        if channel not in channels:
            continue
        sample_token = record.get("sample_token")
        if not isinstance(sample_token, str) or not sample_token:
            raise ValueError("sample_data is missing sample_token")
        key = (sample_token, channel)
        if key in records:
            raise ValueError(f"duplicate key sample_data for {sample_token}/{channel}")
        records[key] = record

    keyframes: list[NuScenesCameraKeyframe] = []
    for frame_index, sample in enumerate(samples):
        sample_token = str(sample["token"])
        for channel in channels:
            record = records.get((sample_token, channel))
            if record is None:
                raise ValueError(f"sample {sample_token} is missing key data: {channel}")
            sample_data_token = record.get("token")
            if (
                not isinstance(sample_data_token, str)
                or not sample_data_token
                or Path(sample_data_token).name != sample_data_token
            ):
                raise ValueError("sample_data token must be a safe non-empty filename")
            filename = record.get("filename")
            if not isinstance(filename, str) or not filename:
                raise ValueError(f"{channel} sample_data is missing filename")
            image_path = data_root / filename
            if not image_path.is_file():
                raise FileNotFoundError(f"missing nuScenes image: {image_path}")
            try:
                image_path.resolve().relative_to(data_root.resolve())
            except ValueError as error:
                raise ValueError(f"nuScenes image escapes data root: {filename}") from error
            height, width = record.get("height"), record.get("width")
            if (
                isinstance(height, bool)
                or isinstance(width, bool)
                or not isinstance(height, int)
                or not isinstance(width, int)
                or height <= 0
                or width <= 0
            ):
                raise ValueError(f"{channel} sample_data has invalid image dimensions")
            keyframes.append(
                NuScenesCameraKeyframe(
                    sample_token=sample_token,
                    sample_data_token=sample_data_token,
                    frame_index=frame_index,
                    channel=channel,
                    image_path=image_path,
                    image_size=(height, width),
                )
            )
    return tuple(keyframes)


def mask_output_path(
    output_root: str | Path,
    *,
    version: str,
    scene_name: str,
    frame: NuScenesCameraKeyframe,
) -> Path:
    return (
        Path(output_root)
        / version
        / scene_name
        / frame.channel
        / f"{frame.sample_data_token}.png"
    )


def union_instance_masks(masks: Any, image_size: tuple[int, int]) -> Any:
    """OR ``[N,H,W]`` or ``[N,1,H,W]`` masks into one boolean ``[H,W]`` mask."""

    import numpy as np

    height, width = image_size
    array = np.asarray(masks)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3 or array.shape[1:] != (height, width):
        raise ValueError(
            f"instance masks must have shape [N,{height},{width}] or [N,1,{height},{width}]"
        )
    if array.shape[0] == 0:
        return np.zeros((height, width), dtype=np.bool_)
    return np.any(array.astype(np.bool_, copy=False), axis=0)


def _normalized_binary_mask(mask: Any, image_size: tuple[int, int]) -> Any:
    import numpy as np

    height, width = image_size
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    array = np.asarray(mask)
    if array.shape != (height, width):
        raise ValueError(
            f"sky mask shape {tuple(array.shape)} does not match image {(height, width)}"
        )
    if not np.isfinite(array).all():
        raise ValueError("sky mask must contain only finite values")
    if array.dtype == np.bool_:
        binary = array
    else:
        unique = np.unique(array)
        if not set(unique.tolist()).issubset({0, 1, 255}):
            raise ValueError("sky mask must be boolean or contain only 0/1/255")
        binary = array != 0
    return np.ascontiguousarray(binary)


def apply_exclusion_mask(
    sky_mask: Any, exclusion_mask: Any, image_size: tuple[int, int]
) -> Any:
    """Return ``sky_mask & ~exclusion_mask`` as a contiguous boolean mask."""

    import numpy as np

    sky = _normalized_binary_mask(sky_mask, image_size)
    exclusion = _normalized_binary_mask(exclusion_mask, image_size)
    return np.ascontiguousarray(sky & ~exclusion)


def write_binary_mask_png(mask: Any, image_size: tuple[int, int], path: Path) -> float:
    """Atomically write an exact-size grayscale 0/255 PNG and return coverage."""

    import numpy as np
    from PIL import Image

    binary = _normalized_binary_mask(mask, image_size)
    pixels = binary.astype(np.uint8) * np.uint8(255)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    Image.fromarray(pixels, mode="L").save(temporary, format="PNG")
    temporary.replace(path)
    return float(binary.mean())


def inspect_binary_mask_png(path: Path, image_size: tuple[int, int]) -> float:
    """Validate a resumable mask and return its sky-pixel coverage."""

    import numpy as np
    from PIL import Image

    height, width = image_size
    try:
        with Image.open(path) as image:
            image.load()
            if image.mode != "L" or image.size != (width, height):
                raise ValueError(
                    f"existing mask must be L/{width}x{height}, got {image.mode}/{image.size}"
                )
            array = np.asarray(image)
    except OSError as error:
        raise ValueError(f"cannot read existing mask: {path}") from error
    values = np.unique(array)
    if not set(values.tolist()).issubset({0, 255}):
        raise ValueError(f"existing mask is not binary 0/255: {path}")
    return float((array != 0).mean())


def configure_groundingdino_arguments(
    arguments: Any, *, device: str, bert_path: Path | None
) -> Any:
    """Apply runtime paths for the pinned GroundingDINO BERT contract.

    The model builder dispatches on the canonical encoder name and reads the
    offline snapshot from ``bert_base_uncased_path``. Replacing
    ``text_encoder_type`` with a filesystem path makes the pinned builder reject
    the otherwise valid local model.
    """

    arguments.device = device
    if bert_path is not None:
        local_bert_path = str(bert_path.resolve())
        arguments.bert_base_uncased_path = local_bert_path
        arguments.text_encoder_type = "bert-base-uncased"
    return arguments


class GroundedSamBackend:
    """Original GroundingDINO + SAM backend matching the paper's recipe."""

    def __init__(
        self,
        *,
        groundingdino_config: Path,
        groundingdino_checkpoint: Path,
        sam_checkpoint: Path,
        sam_model_type: str,
        bert_path: Path | None,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
        device: str,
        negative_text_prompt: str | None = None,
    ) -> None:
        # These are intentionally runtime-only imports.  ArmGS training and
        # metadata dry-runs do not need either third-party checkout installed.
        try:
            import numpy as np
            import torch
            import groundingdino.datasets.transforms as T
            from groundingdino.models import build_model
            from groundingdino.util import box_ops
            from groundingdino.util.inference import predict
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.util.utils import clean_state_dict
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as error:
            raise RuntimeError(
                "Grounded SAM dependencies are unavailable. Activate the "
                "armgs-gsam environment and install GroundingDINO plus "
                "segment_anything before generation."
            ) from error

        for label, path in (
            ("GroundingDINO config", groundingdino_config),
            ("GroundingDINO checkpoint", groundingdino_checkpoint),
            ("SAM checkpoint", sam_checkpoint),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} does not exist: {path}")
        if bert_path is not None and not bert_path.exists():
            raise FileNotFoundError(f"BERT path does not exist: {bert_path}")
        if not text_prompt.strip():
            raise ValueError("text_prompt cannot be empty")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")

        arguments = configure_groundingdino_arguments(
            SLConfig.fromfile(str(groundingdino_config)),
            device=device,
            bert_path=bert_path,
        )
        grounding_model = build_model(arguments)
        checkpoint = torch.load(str(groundingdino_checkpoint), map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        grounding_model.load_state_dict(clean_state_dict(state_dict), strict=False)
        grounding_model.eval().to(device)

        if sam_model_type not in sam_model_registry:
            raise ValueError(f"unknown SAM model type: {sam_model_type}")
        sam = sam_model_registry[sam_model_type](checkpoint=str(sam_checkpoint))
        sam.eval().to(device=device)

        self._np = np
        self._torch = torch
        self._transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self._box_ops = box_ops
        self._predict = predict
        self._grounding_model = grounding_model
        self._sam_predictor = SamPredictor(sam)
        self._text_prompt = text_prompt.strip()
        self._negative_text_prompt = (
            negative_text_prompt.strip()
            if negative_text_prompt and negative_text_prompt.strip()
            else None
        )
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._device = device

    def infer(self, image_path: Path) -> SkyInferenceResult:
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
        detection_count = int(boxes.shape[0])
        excluded_boxes: Any | None = None
        excluded_logits: tuple[float, ...] = ()
        excluded_phrases: tuple[str, ...] = ()
        excluded_detection_count = 0
        if self._negative_text_prompt is not None:
            excluded_boxes, raw_excluded_logits, raw_excluded_phrases = self._predict(
                model=self._grounding_model,
                image=transformed,
                caption=self._negative_text_prompt,
                box_threshold=self._box_threshold,
                text_threshold=self._text_threshold,
                device=self._device,
            )
            excluded_detection_count = int(excluded_boxes.shape[0])
            excluded_phrases = tuple(
                str(phrase) for phrase in raw_excluded_phrases
            )
            excluded_logits = tuple(
                float(value)
                for value in raw_excluded_logits.detach().cpu().tolist()
            )

        if detection_count > 0 or excluded_detection_count > 0:
            self._sam_predictor.set_image(image_source)

        def segment_boxes(detected_boxes: Any) -> Any:
            if int(detected_boxes.shape[0]) == 0:
                return self._np.zeros((height, width), dtype=self._np.bool_)
            boxes_xyxy = self._box_ops.box_cxcywh_to_xyxy(
                detected_boxes
            ) * self._torch.tensor(
                [width, height, width, height], dtype=detected_boxes.dtype
            )
            transformed_boxes = self._sam_predictor.transform.apply_boxes_torch(
                boxes_xyxy, image_source.shape[:2]
            ).to(self._device)
            instance_masks, _, _ = self._sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            return union_instance_masks(
                instance_masks.detach().cpu().numpy(), (height, width)
            )

        sky_mask = segment_boxes(boxes)
        mask = (
            apply_exclusion_mask(
                sky_mask,
                segment_boxes(excluded_boxes),
                (height, width),
            )
            if excluded_boxes is not None
            else sky_mask
        )
        return SkyInferenceResult(
            mask=mask,
            detection_count=detection_count,
            phrases=tuple(str(phrase) for phrase in phrases),
            logits=tuple(float(value) for value in logits.detach().cpu().tolist()),
            excluded_detection_count=excluded_detection_count,
            excluded_phrases=excluded_phrases,
            excluded_logits=excluded_logits,
        )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _overlay_image(image_path: Path, mask_path: Path, label: str) -> Any:
    import numpy as np
    from PIL import Image, ImageDraw

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    with Image.open(mask_path) as mask_image:
        mask = np.asarray(mask_image.convert("L")) != 0
    array = np.asarray(rgb).copy()
    tint = np.empty_like(array)
    tint[..., 0] = 30
    tint[..., 1] = 150
    tint[..., 2] = 255
    array[mask] = (
        array[mask].astype(np.float32) * 0.45 + tint[mask].astype(np.float32) * 0.55
    ).astype(np.uint8)
    result = Image.fromarray(array, mode="RGB")
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, min(result.width, 620), 24), fill=(0, 0, 0))
    draw.text((5, 5), label, fill=(255, 255, 255))
    return result


def _save_overlay(
    frame: NuScenesCameraKeyframe,
    mask_path: Path,
    overlay_path: Path,
    coverage: float,
) -> None:
    overlay = _overlay_image(
        frame.image_path,
        mask_path,
        f"{frame.channel} frame={frame.frame_index} sky={coverage:.3f}",
    )
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = overlay_path.with_name(f".{overlay_path.name}.tmp")
    overlay.save(temporary, format="JPEG", quality=90)
    temporary.replace(overlay_path)


def _evenly_sample(items: Sequence[Any], count: int) -> list[Any]:
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[0]]
    return [items[round(index * (len(items) - 1) / (count - 1))] for index in range(count)]


def save_contact_sheet(
    entries: Sequence[tuple[NuScenesCameraKeyframe, Path, float]],
    output_path: Path,
    *,
    max_items: int = 48,
) -> None:
    """Save compact image/mask overlays for manual QA."""

    from PIL import Image, ImageOps

    if not entries:
        return
    selected = _evenly_sample(entries, min(max_items, len(entries)))
    tile_size = (320, 204)
    columns = min(4, len(selected))
    rows = math.ceil(len(selected) / columns)
    sheet = Image.new("RGB", (columns * tile_size[0], rows * tile_size[1]), "black")
    for index, (frame, mask_path, coverage) in enumerate(selected):
        overlay = _overlay_image(
            frame.image_path,
            mask_path,
            f"{frame.channel} f={frame.frame_index} sky={coverage:.3f}",
        )
        thumbnail = ImageOps.fit(overlay, tile_size, method=Image.Resampling.LANCZOS)
        x = (index % columns) * tile_size[0]
        y = (index // columns) * tile_size[1]
        sheet.paste(thumbnail, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    image_format = "PNG" if output_path.suffix.lower() == ".png" else "JPEG"
    sheet.save(temporary, format=image_format, quality=90)
    temporary.replace(output_path)


def _new_manifest(args: argparse.Namespace, scene_name: str) -> dict[str, Any]:
    negative_text_prompt = getattr(args, "negative_text_prompt", None)
    normalized_negative_prompt = (
        negative_text_prompt.strip()
        if negative_text_prompt and negative_text_prompt.strip()
        else None
    )
    return {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "nuscenes_root": str(args.nuscenes_root.resolve()),
            "version": args.version,
            "scene": scene_name,
            "camera_channels": list(args.cameras),
        },
        "generation": {
            "method": "GroundingDINO/SAM sky union minus optional exclusion union",
            "text_prompt": args.text_prompt,
            "negative_text_prompt": normalized_negative_prompt,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
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


def _update_summary(payload: dict[str, Any]) -> dict[str, Any]:
    frames = payload["frames"]
    coverages = [
        float(frame["coverage"])
        for frame in frames
        if frame["coverage"] is not None
    ]
    statuses: dict[str, int] = {}
    for frame in frames:
        status = str(frame["status"])
        statuses[status] = statuses.get(status, 0) + 1
    payload["summary"] = {
        "total": len(frames),
        "statuses": statuses,
        "no_detection": sum(frame.get("no_detection") is True for frame in frames),
        "frames_with_exclusions": sum(
            (frame.get("excluded_detection_count") or 0) > 0 for frame in frames
        ),
        "mean_coverage": sum(coverages) / len(coverages) if coverages else None,
        "min_coverage": min(coverages) if coverages else None,
        "max_coverage": max(coverages) if coverages else None,
    }
    return payload


def _frame_record(
    frame: NuScenesCameraKeyframe,
    mask_path: Path,
    *,
    status: str,
    coverage: float | None,
    detection_count: int | None,
    phrases: Sequence[str] = (),
    logits: Sequence[float] = (),
    excluded_detection_count: int | None = None,
    excluded_phrases: Sequence[str] = (),
    excluded_logits: Sequence[float] = (),
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "sample_token": frame.sample_token,
        "sample_data_token": frame.sample_data_token,
        "frame_index": frame.frame_index,
        "channel": frame.channel,
        "image_path": str(frame.image_path.resolve()),
        "image_size": list(frame.image_size),
        "mask_path": str(mask_path.resolve()),
        "status": status,
        "detection_count": detection_count,
        "no_detection": (
            detection_count == 0 if detection_count is not None else coverage == 0.0
        ),
        "coverage": coverage,
        "phrases": list(phrases),
        "logits": [float(value) for value in logits],
        "excluded_detection_count": excluded_detection_count,
        "excluded_phrases": list(excluded_phrases),
        "excluded_logits": [float(value) for value in excluded_logits],
        "error": error,
    }


def _make_backend(args: argparse.Namespace) -> GroundedSamBackend:
    return GroundedSamBackend(
        groundingdino_config=args.groundingdino_config,
        groundingdino_checkpoint=args.groundingdino_checkpoint,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        bert_path=args.bert_path,
        text_prompt=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
        negative_text_prompt=args.negative_text_prompt,
    )


def generate_sky_masks(
    args: argparse.Namespace,
    *,
    backend: SkySegmentationBackend | None = None,
) -> dict[str, Any]:
    """Generate or resume masks and return the complete JSON payload."""

    scene_name = normalize_nuscenes_scene_name(args.scene)
    frames = enumerate_nuscenes_camera_keyframes(
        args.nuscenes_root,
        scene=scene_name,
        version=args.version,
        camera_channels=args.cameras,
    )
    scene_output = args.output_root / args.version / scene_name
    manifest_path = args.manifest_path or scene_output / "generation_manifest.json"
    overlay_root = args.overlay_root or scene_output / "overlays"
    payload = _new_manifest(args, scene_name)

    if args.dry_run:
        for frame in frames:
            output_path = mask_output_path(
                args.output_root,
                version=args.version,
                scene_name=scene_name,
                frame=frame,
            )
            payload["frames"].append(
                _frame_record(
                    frame,
                    output_path,
                    status="dry_run_existing" if output_path.is_file() else "dry_run_pending",
                    coverage=None,
                    detection_count=None,
                )
            )
        return _update_summary(payload)

    model = backend
    overlay_entries: list[tuple[NuScenesCameraKeyframe, Path, float]] = []
    previous_records: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file() and args.existing_mode == "resume":
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            previous_records = {
                str(record["sample_data_token"]): record
                for record in previous.get("frames", [])
                if isinstance(record, dict) and "sample_data_token" in record
            }
        except (OSError, json.JSONDecodeError, TypeError, KeyError):
            previous_records = {}

    for index, frame in enumerate(frames):
        output_path = mask_output_path(
            args.output_root,
            version=args.version,
            scene_name=scene_name,
            frame=frame,
        )
        coverage: float
        if output_path.is_file() and args.existing_mode == "resume":
            try:
                coverage = inspect_binary_mask_png(output_path, frame.image_size)
            except ValueError:
                # A partial/corrupt/wrong-size output is never accepted as a
                # completed frame. Regenerate it in-place atomically.
                pass
            else:
                old = previous_records.get(frame.sample_data_token, {})
                old_count = old.get("detection_count")
                detection_count = old_count if isinstance(old_count, int) else None
                old_excluded_count = old.get("excluded_detection_count")
                excluded_detection_count = (
                    old_excluded_count if isinstance(old_excluded_count, int) else None
                )
                payload["frames"].append(
                    _frame_record(
                        frame,
                        output_path,
                        status="skipped_existing",
                        coverage=coverage,
                        detection_count=detection_count,
                        phrases=old.get("phrases", ()),
                        logits=old.get("logits", ()),
                        excluded_detection_count=excluded_detection_count,
                        excluded_phrases=old.get("excluded_phrases", ()) or (),
                        excluded_logits=old.get("excluded_logits", ()) or (),
                    )
                )
                overlay_entries.append((frame, output_path, coverage))
                continue

        if model is None:
            model = _make_backend(args)
        try:
            result = model.infer(frame.image_path)
            binary = _normalized_binary_mask(result.mask, frame.image_size)
            coverage = write_binary_mask_png(binary, frame.image_size, output_path)
            status = "no_detection" if result.detection_count == 0 else "generated"
            record = _frame_record(
                frame,
                output_path,
                status=status,
                coverage=coverage,
                detection_count=result.detection_count,
                phrases=result.phrases,
                logits=result.logits,
                excluded_detection_count=result.excluded_detection_count,
                excluded_phrases=result.excluded_phrases,
                excluded_logits=result.excluded_logits,
            )
            payload["frames"].append(record)
            overlay_entries.append((frame, output_path, coverage))
            if args.save_overlays and index % args.overlay_every == 0:
                _save_overlay(
                    frame,
                    output_path,
                    overlay_root / frame.channel / f"{frame.sample_data_token}.jpg",
                    coverage,
                )
        except Exception as error:
            payload["frames"].append(
                _frame_record(
                    frame,
                    output_path,
                    status="error",
                    coverage=None,
                    detection_count=None,
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
    payload = generate_sky_masks(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    if args.dry_run:
        print("dry-run: no models were loaded and no files were written")
    else:
        scene_name = normalize_nuscenes_scene_name(args.scene)
        manifest_path = args.manifest_path or (
            args.output_root / args.version / scene_name / "generation_manifest.json"
        )
        print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
