"""Convert canonical dataset frames into one-view ArmGS training batches.

Image decoding is intentionally lazy: Pillow is imported only when the default
reader is called. Callers with another image stack can inject a tensor reader
without adding that dependency to the core package.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypeAlias

import torch
from torch import Tensor

from .data.schema import CanonicalFrame
from .pipeline import CameraView
from .training import ArmGSTrainingBatch


ImageReadMode = Literal["rgb", "mask"]
ImageTensorReader: TypeAlias = Callable[[Path, ImageReadMode], Tensor]

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def pillow_image_reader(path: Path, mode: ImageReadMode) -> Tensor:
    """Read an RGB image or one-channel mask as a uint8 tensor.

    Pillow remains an optional dependency. Supplying an image_reader to
    canonical_frame_to_training_batch bypasses this function entirely.
    """

    try:
        from PIL import Image
    except ImportError as error:
        raise ImportError(
            "Reading canonical frame images requires the optional Pillow "
            "dependency. Install it with 'pip install Pillow' or pass a custom "
            "image_reader to canonical_frame_to_training_batch()."
        ) from error

    if mode not in ("rgb", "mask"):
        raise ValueError("mode must be 'rgb' or 'mask'")
    converted_mode = "RGB" if mode == "rgb" else "L"
    with Image.open(path) as image:
        converted = image.convert(converted_mode)
        width, height = converted.size
        channels = 3 if mode == "rgb" else 1
        # bytearray gives torch writable storage and avoids a frombuffer warning.
        pixels = torch.frombuffer(
            bytearray(converted.tobytes()), dtype=torch.uint8
        ).clone()
    if channels == 3:
        return pixels.reshape(height, width, channels)
    return pixels.reshape(height, width)


def _read_tensor(
    reader: ImageTensorReader,
    path: Path,
    mode: ImageReadMode,
) -> Tensor:
    value = reader(path, mode)
    if not isinstance(value, Tensor):
        raise TypeError(f"image_reader must return a Tensor for {path}")
    if value.is_complex():
        raise ValueError(f"{mode} tensor cannot be complex")
    return value


def _normalise_rgb(image: Tensor, image_size: tuple[int, int]) -> Tensor:
    height, width = image_size
    if image.ndim != 3 or tuple(image.shape) != (height, width, 3):
        raise ValueError(
            "RGB image size mismatch: expected "
            f"[{height},{width},3], got {list(image.shape)}"
        )
    if image.dtype == torch.uint8:
        return image.to(dtype=torch.float32).div_(255.0)
    if not image.is_floating_point():
        raise ValueError("RGB image must be uint8 or floating point")
    if not torch.isfinite(image).all():
        raise ValueError("RGB image must contain only finite values")
    if torch.any((image < 0.0) | (image > 1.0)):
        raise ValueError("floating-point RGB image values must lie in [0,1]")
    return image.to(dtype=torch.float32)


def _normalise_mask(
    mask: Tensor,
    image_size: tuple[int, int],
    *,
    name: str,
) -> Tensor:
    height, width = image_size
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 2 or tuple(mask.shape) != (height, width):
        raise ValueError(
            f"{name} size mismatch: expected [{height},{width}] or "
            f"[{height},{width},1], got {list(mask.shape)}"
        )
    if mask.is_floating_point() and not torch.isfinite(mask).all():
        raise ValueError(f"{name} must contain only finite values")
    if mask.dtype != torch.bool and (
        not mask.is_floating_point() and mask.dtype not in _INTEGER_DTYPES
    ):
        raise ValueError(f"{name} must be boolean, integer, or floating point")
    return mask.to(dtype=torch.bool)


def _scalar_training_row(
    training_row: int | Tensor | None,
    *,
    device: torch.device,
) -> Tensor | None:
    if training_row is None:
        return None
    if isinstance(training_row, bool):
        raise ValueError("training_row must be a non-negative integer scalar")
    row = torch.as_tensor(training_row, device=device)
    if row.numel() != 1 or row.dtype not in _INTEGER_DTYPES:
        raise ValueError("training_row must be a non-negative integer scalar")
    row = row.reshape(()).to(dtype=torch.long)
    if row.item() < 0:
        raise ValueError("training_row must be non-negative")
    return row


def _sparse_lidar_depth(
    frame: CanonicalFrame,
    *,
    device: torch.device,
) -> tuple[Tensor | None, Tensor | None]:
    projection = frame.lidar_projection
    if projection is None:
        return None, None

    height, width = frame.image_size
    pixels = projection.pixel_indices.detach().to(device=device, dtype=torch.long)
    depths = projection.depths.detach().to(device=device, dtype=torch.float32)
    if pixels.ndim != 2 or pixels.shape != (depths.numel(), 2):
        raise ValueError("lidar pixel_indices must have shape [M,2]")
    if depths.ndim != 1:
        raise ValueError("lidar depths must have shape [M]")
    if not torch.isfinite(depths).all() or torch.any(depths <= 0):
        raise ValueError("lidar depths must be finite and positive")
    if pixels.numel():
        x, y = pixels.unbind(dim=-1)
        if torch.any((x < 0) | (x >= width) | (y < 0) | (y >= height)):
            raise ValueError("lidar pixel_indices must lie inside the image")
        linear_indices = y * width + x
    else:
        linear_indices = torch.empty(0, device=device, dtype=torch.long)

    # amin makes duplicate-pixel resolution independent of input ordering.
    flattened = torch.full(
        (height * width,),
        torch.inf,
        device=device,
        dtype=torch.float32,
    )
    if linear_indices.numel():
        flattened.scatter_reduce_(
            0,
            linear_indices,
            depths,
            reduce="amin",
            include_self=True,
        )
    valid = torch.isfinite(flattened)
    flattened = torch.where(valid, flattened, torch.zeros_like(flattened))
    depth = flattened.reshape(1, height, width, 1)
    valid_mask = valid.reshape(1, height, width, 1)
    return depth, valid_mask


def canonical_frame_to_training_batch(
    frame: CanonicalFrame,
    training_row: int | Tensor | None,
    *,
    image_reader: ImageTensorReader | None = None,
    device: torch.device | str | None = None,
) -> ArmGSTrainingBatch:
    """Materialize one canonical frame as a singleton ArmGS training batch.

    RGB inputs are either uint8 or floating point in [0,1]. Masks treat
    zero as false and every non-zero value as true. When multiple projected
    LiDAR samples share a pixel, only the nearest positive depth is retained.
    Held-out views pass ``training_row=None`` so the renderer uses the nearest
    same-camera training-frame appearance embedding.
    """

    if not isinstance(frame, CanonicalFrame):
        raise TypeError("frame must be a CanonicalFrame")
    height, width = frame.image_size
    if height <= 0 or width <= 0:
        raise ValueError("frame image dimensions must be positive")
    resolved_device = (
        frame.camera_to_world.device
        if device is None
        else torch.device(device)
    )
    reader = pillow_image_reader if image_reader is None else image_reader

    rgb = _normalise_rgb(
        _read_tensor(reader, frame.image_path, "rgb"),
        frame.image_size,
    ).to(device=resolved_device)
    target_rgb = rgb.unsqueeze(0).contiguous()

    def read_mask(path: Path | None, name: str) -> Tensor | None:
        if path is None:
            return None
        mask = _normalise_mask(
            _read_tensor(reader, path, "mask"),
            frame.image_size,
            name=name,
        )
        return mask.to(device=resolved_device).reshape(1, height, width, 1).contiguous()

    target_sky_mask = read_mask(frame.sky_mask_path, "sky mask")
    sky_valid_mask = (
        torch.tensor(
            frame.sky_supervision_valid,
            dtype=torch.bool,
            device=resolved_device,
        ).reshape(1, 1, 1, 1)
        if target_sky_mask is not None
        else None
    )
    actor_bbox_mask = read_mask(frame.actor_mask_path, "actor mask")
    lidar_depth, depth_valid_mask = _sparse_lidar_depth(
        frame, device=resolved_device
    )

    view = CameraView(
        camera_to_world=frame.camera_to_world.to(device=resolved_device),
        intrinsics=frame.intrinsics.to(device=resolved_device),
        image_size=frame.image_size,
        timestamp=frame.timestamp.to(device=resolved_device),
        camera_id=torch.tensor(
            frame.camera_id, dtype=torch.long, device=resolved_device
        ),
        training_row=_scalar_training_row(
            training_row, device=resolved_device
        ),
        camera_convention=frame.camera_convention,
    )
    return ArmGSTrainingBatch(
        view=view,
        target_rgb=target_rgb,
        lidar_depth=lidar_depth,
        depth_valid_mask=depth_valid_mask,
        target_sky_mask=target_sky_mask,
        sky_valid_mask=sky_valid_mask,
        actor_bbox_mask=actor_bbox_mask,
    )


__all__ = [
    "ImageReadMode",
    "ImageTensorReader",
    "canonical_frame_to_training_batch",
    "pillow_image_reader",
]
