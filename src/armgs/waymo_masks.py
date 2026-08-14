"""Materialize canonical actor-box supervision masks for Waymo training."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile

import torch
from torch import Tensor

from .data.schema import CanonicalDatasetManifest, CanonicalFrame
from .evaluation import project_actor_boxes_to_mask


def _mask_path(output_root: Path, frame: CanonicalFrame) -> Path:
    return (
        output_root
        / f"camera_{frame.camera_id:03d}"
        / f"frame_{frame.frame_index:08d}.png"
    )


def _mask_pixels(mask: Tensor, image_size: tuple[int, int]) -> object:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - PyTorch installs NumPy.
        raise ImportError("actor-mask materialization requires NumPy") from error

    height, width = image_size
    if mask.shape != (height, width) or mask.dtype != torch.bool:
        raise ValueError(
            "actor box projector must return a bool mask with shape "
            f"[{height},{width}]"
        )
    return (
        mask.detach()
        .to(device="cpu", dtype=torch.uint8)
        .mul_(255)
        .contiguous()
        .numpy()
        .astype(np.uint8, copy=False)
    )


def _existing_mask_matches(path: Path, expected_pixels: object) -> bool:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:  # pragma: no cover - optional dependency guard.
        raise ImportError(
            "actor-mask materialization requires Pillow and NumPy"
        ) from error

    expected = np.asarray(expected_pixels)
    height, width = expected.shape
    try:
        with Image.open(path) as image:
            image.load()
            if image.mode != "L" or image.size != (width, height):
                return False
            actual = np.asarray(image)
            return bool(np.array_equal(actual, expected))
    except (OSError, ValueError):
        return False


def _atomic_write_mask(path: Path, pixels: object) -> None:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - optional dependency guard.
        raise ImportError("actor-mask materialization requires Pillow") from error

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            Image.fromarray(pixels, mode="L").save(handle, format="PNG")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def materialize_actor_bbox_masks(
    manifest: CanonicalDatasetManifest,
    output_root: str | Path,
    *,
    box_scale: float = 1.0,
) -> CanonicalDatasetManifest:
    """Write and attach one exact actor-box union mask for every frame.

    Masks are grayscale PNGs with actor pixels equal to 255 and background
    pixels equal to zero.  Frames with no visible actor still receive a fully
    empty mask; this is essential because a missing mask disables the paper's
    outside-box foreground supervision, whereas an attached empty mask marks
    the entire image as outside actor boxes.

    Paths are stable per canonical ``(camera_id, frame_index)``. Existing files
    are reused only when mode, dimensions, and every binary pixel match the
    current projection. Corrupt, stale, or differently encoded files are
    replaced atomically.
    """

    if not isinstance(manifest, CanonicalDatasetManifest):
        raise TypeError("manifest must be a CanonicalDatasetManifest")
    root = Path(output_root).resolve()
    frame_keys = [(frame.camera_id, frame.frame_index) for frame in manifest.frames]
    if len(frame_keys) != len(set(frame_keys)):
        raise ValueError(
            "manifest contains duplicate camera_id/frame_index actor-mask paths"
        )

    frames: list[CanonicalFrame] = []
    for frame in manifest.frames:
        mask = project_actor_boxes_to_mask(
            frame,
            manifest.actor_tracks,
            box_scale=box_scale,
        )
        pixels = _mask_pixels(mask, frame.image_size)
        destination = _mask_path(root, frame)
        if not destination.is_file() or not _existing_mask_matches(
            destination, pixels
        ):
            _atomic_write_mask(destination, pixels)
        frames.append(replace(frame, actor_mask_path=destination))

    return CanonicalDatasetManifest(
        frames=tuple(frames),
        actor_tracks=manifest.actor_tracks,
        timestamp_unit=manifest.timestamp_unit,
    )


__all__ = ["materialize_actor_bbox_masks"]
