"""Image-quality evaluation utilities for ArmGS reconstructions.

The paper reports PSNR, SSIM, LPIPS, and PSNR inside projected actor boxes.  This
module keeps those evaluation choices independent from the training objective and
supports streaming over scenes that do not fit in memory.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import torch
from torch import Tensor, nn

from .data.schema import ActorTrack, CanonicalFrame
from .geometry import quaternion_to_rotation_matrix
from .losses import structural_similarity


class LPIPSUnavailableError(RuntimeError):
    """Raised when LPIPS was requested but its optional package is unavailable."""


_CUBOID_EDGES = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)


def _positive_finite_float(value: float, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite positive number") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _near_clipped_cuboid_vertices(corners: Tensor, near_plane: float) -> Tensor:
    """Return vertices of a cuboid clipped to camera-space z >= near."""

    near = corners.new_tensor(near_plane)
    in_front = corners[:, 2] >= near
    vertices = [corner for corner in corners[in_front]]
    for start_index, end_index in _CUBOID_EDGES:
        start = corners[start_index]
        end = corners[end_index]
        start_front = bool(in_front[start_index].item())
        end_front = bool(in_front[end_index].item())
        if start_front == end_front:
            continue
        weight = (near - start[2]) / (end[2] - start[2])
        vertices.append(start + weight * (end - start))
    if not vertices:
        return corners.new_empty((0, 3))
    return torch.stack(vertices)


def _convex_hull_2d(points: Tensor) -> list[tuple[int, int]]:
    """Return the counter-clockwise hull of rounded projected vertices.

    StreetGS rasterizes the projected cuboid faces after rounding image
    coordinates. A cuboid is convex, so the union of those faces has the same
    interior as the convex hull of its projected (near-plane-clipped) vertices.
    """

    rounded = torch.round(points).to(device="cpu", dtype=torch.int64)
    unique = sorted({(int(x), int(y)) for x, y in rounded.tolist()})
    if len(unique) <= 1:
        return unique

    def cross(
        origin: tuple[int, int],
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> int:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[int, int]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[int, int]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _fill_projected_cuboid(mask: Tensor, projected_vertices: Tensor) -> None:
    """Rasterize a projected convex cuboid silhouette into mask in place."""

    hull = _convex_hull_2d(projected_vertices)
    if len(hull) < 3:
        return
    height, width = mask.shape
    first_row = max(0, min(point[1] for point in hull))
    last_row = min(height - 1, max(point[1] for point in hull))
    if first_row > last_row:
        return

    edges = tuple(zip(hull, hull[1:] + hull[:1]))
    for row in range(first_row, last_row + 1):
        intersections: list[float] = []
        for (x0, y0), (x1, y1) in edges:
            if y0 == y1:
                if row == y0:
                    intersections.extend((float(x0), float(x1)))
                continue
            if min(y0, y1) <= row <= max(y0, y1):
                weight = (row - y0) / (y1 - y0)
                intersections.append(x0 + weight * (x1 - x0))
        if not intersections:
            continue
        first_column = max(0, math.ceil(min(intersections) - 1.0e-9))
        last_column = min(
            width - 1, math.floor(max(intersections) + 1.0e-9)
        )
        if first_column <= last_column:
            mask[row, first_column : last_column + 1] = True


@torch.inference_mode()
def project_actor_boxes_to_mask(
    frame: CanonicalFrame,
    actor_tracks: Sequence[ActorTrack],
    *,
    box_scale: float = 1.0,
    near_plane: float = 1.0e-3,
) -> Tensor:
    """Project exact-frame actor boxes into a union image-space silhouette mask.

    The returned bool[H,W] mask is intended for the paper's actor-region PSNR,
    not as pixel-accurate actor supervision. Each track contributes only when
    it contains a sample whose frame_index exactly matches frame; poses are
    never interpolated. Cuboid edges are clipped against the camera near plane
    before projection so boxes crossing the camera plane remain finite and
    conservatively cover their visible image region.

    dimensions_lwh are interpreted along actor-local x/y/z and actor
    quaternions use [w,x,y,z]. OpenCV cameras use +z forward and +y down. For
    OpenGL cameras, native y and z are flipped into OpenCV projection
    coordinates before applying frame.intrinsics.
    """

    if not isinstance(frame, CanonicalFrame):
        raise TypeError("frame must be a CanonicalFrame")
    if isinstance(actor_tracks, (str, bytes)) or not isinstance(
        actor_tracks, Sequence
    ):
        raise TypeError("actor_tracks must be a sequence of ActorTrack objects")
    scale = _positive_finite_float(box_scale, name="box_scale")
    near = _positive_finite_float(near_plane, name="near_plane")

    height, width = frame.image_size
    device = frame.camera_to_world.device
    compute_dtype = (
        torch.float64
        if frame.camera_to_world.dtype == torch.float64
        or frame.intrinsics.dtype == torch.float64
        else torch.float32
    )
    camera_to_world = frame.camera_to_world.to(device=device, dtype=compute_dtype)
    intrinsics = frame.intrinsics.to(device=device, dtype=compute_dtype)
    camera_rotation = camera_to_world[:3, :3]
    camera_translation = camera_to_world[:3, 3]
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    signs = camera_to_world.new_tensor(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    for track in actor_tracks:
        if not isinstance(track, ActorTrack):
            raise TypeError("actor_tracks must contain only ActorTrack objects")
        sample = next(
            (
                candidate
                for candidate in track.samples
                if candidate.frame_index == frame.frame_index
            ),
            None,
        )
        if sample is None:
            continue

        dimensions = track.dimensions_lwh.to(
            device=device, dtype=compute_dtype
        ).clone()
        # Match StreetGS/Waymo preprocessing: box_scale expands actor-local
        # length and width, while the raw tracker height is preserved.
        dimensions[:2] *= scale
        local_corners = signs * (dimensions / 2.0)
        actor_rotation = quaternion_to_rotation_matrix(
            sample.quaternion_wxyz.to(device=device, dtype=compute_dtype)
        )
        actor_translation = sample.translation.to(
            device=device, dtype=compute_dtype
        )
        world_corners = local_corners @ actor_rotation.T + actor_translation
        camera_corners = (world_corners - camera_translation) @ camera_rotation
        if frame.camera_convention == "opengl":
            camera_corners[:, 1:] = -camera_corners[:, 1:]
        elif frame.camera_convention != "opencv":
            # CanonicalFrame validates this, but retain a local defensive check.
            raise ValueError("camera_convention must be 'opencv' or 'opengl'")

        clipped = _near_clipped_cuboid_vertices(camera_corners, near)
        if clipped.numel() == 0:
            continue
        homogeneous_pixels = clipped @ intrinsics.T
        denominators = homogeneous_pixels[:, 2]
        projectable = torch.isfinite(homogeneous_pixels).all(dim=-1) & (
            denominators > torch.finfo(compute_dtype).eps
        )
        if not projectable.any():
            continue
        pixels = (
            homogeneous_pixels[projectable, :2]
            / denominators[projectable, None]
        )
        pixels = pixels[torch.isfinite(pixels).all(dim=-1)]
        if pixels.numel() == 0:
            continue
        _fill_projected_cuboid(mask, pixels)

    return mask


def _to_nchw(images: Tensor, *, name: str) -> Tensor:
    if not isinstance(images, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if images.ndim == 3:
        if images.shape[0] == 3:
            images = images.unsqueeze(0)
        elif images.shape[-1] == 3:
            images = images.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"{name} must have exactly three RGB channels")
    elif images.ndim == 4:
        if images.shape[1] == 3:
            pass
        elif images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"{name} must have exactly three RGB channels")
    else:
        raise ValueError(
            f"{name} must have shape [3,H,W], [H,W,3], [B,3,H,W], or [B,H,W,3]"
        )
    return images.contiguous()


def _validate_data_range(data_range: float) -> float:
    try:
        value = float(data_range)
    except (TypeError, ValueError) as error:
        raise TypeError("data_range must be a finite positive number") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError("data_range must be a finite positive number")
    return value


def _normalize_rgb(
    images: Tensor,
    *,
    selected: Tensor,
    name: str,
    data_range: float,
) -> Tensor:
    """Convert supported RGB encodings to the metric domain ``[0, 1]``.

    ``uint8`` has an intrinsic range of ``[0, 255]``. Floating-point tensors
    use the caller-provided ``data_range``. Validation deliberately follows the
    valid mask so masked non-finite padding retains its historical behavior.
    """

    if images.dtype == torch.uint8:
        return images.to(torch.float32).div_(255.0)
    if not images.is_floating_point():
        raise TypeError(
            f"{name} has unsupported RGB dtype {images.dtype}; expected torch.uint8 "
            "values in [0, 255] or a floating-point tensor in [0, data_range]"
        )

    selected_values = images.masked_select(selected)
    if not torch.isfinite(selected_values).all():
        raise ValueError(f"{name} contains non-finite values in the valid region")
    observed_min = float(selected_values.amin().item())
    observed_max = float(selected_values.amax().item())
    if observed_min < 0.0 or observed_max > data_range:
        raise ValueError(
            f"{name} floating-point RGB values in the valid region must lie in "
            f"[0, {data_range:g}]; observed [{observed_min:g}, {observed_max:g}]"
        )

    if images.device.type == "cpu" and images.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        # CPU conv2d support for low precision depends on the installed Torch build.
        images = images.to(torch.float32)
    return images / data_range


def _to_mask(
    mask: Tensor | None,
    *,
    reference: Tensor,
    name: str,
) -> Tensor:
    batch, _, height, width = reference.shape
    if mask is None:
        return torch.ones(
            (batch, 1, height, width), device=reference.device, dtype=torch.bool
        )
    if not isinstance(mask, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    mask = mask.to(device=reference.device)
    if mask.is_floating_point() and not torch.isfinite(mask).all():
        raise ValueError(f"{name} must contain only finite values")

    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        if tuple(mask.shape[-2:]) == (height, width):
            mask = mask[:, None]
        elif tuple(mask.shape[:2]) == (height, width) and mask.shape[-1] == 1:
            mask = mask.permute(2, 0, 1)[None]
        else:
            raise ValueError(f"{name} has incompatible spatial dimensions")
    elif mask.ndim == 4:
        if mask.shape[1] == 1:
            pass
        elif mask.shape[-1] == 1:
            mask = mask.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"{name} must have one channel")
    else:
        raise ValueError(
            f"{name} must have shape [H,W], [H,W,1], [B,H,W], "
            "[B,1,H,W], or [B,H,W,1]"
        )

    if tuple(mask.shape[-2:]) != (height, width):
        raise ValueError(f"{name} has incompatible spatial dimensions")
    if mask.shape[0] == 1 and batch != 1:
        mask = mask.expand(batch, -1, -1, -1)
    elif mask.shape[0] != batch:
        raise ValueError(f"{name} batch dimension must be 1 or match the images")
    return mask.to(dtype=torch.bool).contiguous()


def _validated_inputs(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor | None,
    *,
    data_range: float,
) -> tuple[Tensor, Tensor, Tensor]:
    data_range = _validate_data_range(data_range)
    prediction = _to_nchw(prediction, name="prediction")
    target = _to_nchw(target, name="target").to(device=prediction.device)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have matching shapes")
    valid = _to_mask(valid_mask, reference=prediction, name="valid_mask")
    if not valid.any():
        raise ValueError("valid_mask selects no pixels")
    selected = valid.expand_as(prediction)
    prediction = _normalize_rgb(
        prediction,
        selected=selected,
        name="prediction",
        data_range=data_range,
    )
    target = _normalize_rgb(
        target,
        selected=selected,
        name="target",
        data_range=data_range,
    )
    common_dtype = torch.promote_types(prediction.dtype, target.dtype)
    prediction = prediction.to(dtype=common_dtype)
    target = target.to(dtype=common_dtype)
    return prediction, target, valid


def _masked_squared_error(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> tuple[Tensor, int]:
    expanded = mask.expand_as(prediction)
    difference = torch.where(expanded, prediction - target, torch.zeros_like(prediction))
    count = int(expanded.sum().item())
    if count == 0:
        raise ValueError("mask selects no RGB values")
    return difference.square().sum(dtype=torch.float64), count


def peak_signal_noise_ratio(
    prediction: Tensor,
    target: Tensor,
    *,
    valid_mask: Tensor | None = None,
    data_range: float = 1.0,
) -> Tensor:
    """Return PSNR over selected RGB values.

    Exact agreement has the conventional value ``+inf``.  ``valid_mask`` is a
    one-channel spatial mask and is broadcast over RGB channels.
    """

    data_range = _validate_data_range(data_range)
    prediction, target, valid = _validated_inputs(
        prediction, target, valid_mask, data_range=data_range
    )
    squared_error, count = _masked_squared_error(prediction, target, valid)
    mse = squared_error / count
    return 10.0 * torch.log10(
        torch.ones((), dtype=mse.dtype, device=mse.device) / mse
    )


psnr = peak_signal_noise_ratio


def _masked_structural_similarity(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    *,
    window_size: int,
    sigma: float,
    data_range: float,
) -> Tensor:
    """Evaluate the existing SSIM implementation on each mask's tight crop.

    Invalid holes inside the crop are set to the same value in both inputs.  This
    prevents invalid RGB values from affecting the score while retaining the
    spatial SSIM definition at valid/invalid boundaries.
    """

    scores: list[Tensor] = []
    for index in range(prediction.shape[0]):
        spatial = valid[index, 0]
        coordinates = torch.nonzero(spatial, as_tuple=False)
        if coordinates.numel() == 0:
            raise ValueError("each image must contain at least one valid pixel")
        y_min, x_min = (int(value.item()) for value in coordinates.amin(dim=0))
        y_max, x_max = (int(value.item()) + 1 for value in coordinates.amax(dim=0))
        image_mask = spatial[y_min:y_max, x_min:x_max][None, None]
        pred_crop = prediction[index : index + 1, :, y_min:y_max, x_min:x_max]
        target_crop = target[index : index + 1, :, y_min:y_max, x_min:x_max]
        pred_crop = torch.where(image_mask, pred_crop, torch.zeros_like(pred_crop))
        target_crop = torch.where(image_mask, target_crop, torch.zeros_like(target_crop))
        scores.append(
            structural_similarity(
                pred_crop,
                target_crop,
                window_size=window_size,
                sigma=sigma,
                data_range=data_range,
            )
        )
    return torch.stack(scores).mean()


class LPIPSMetric:
    """Lazy adapter around the optional ``lpips`` PyPI package."""

    def __init__(self, *, net: str = "alex", device: torch.device | str = "cpu") -> None:
        try:
            lpips_module = import_module("lpips")
        except Exception as error:
            raise LPIPSUnavailableError(
                "LPIPS evaluation requires the optional 'lpips' package; "
                "install it or run evaluation without --lpips"
            ) from error
        try:
            model = lpips_module.LPIPS(net=net)
        except Exception as error:
            raise LPIPSUnavailableError(
                f"LPIPS model '{net}' could not be initialized: {error}"
            ) from error
        self.device = torch.device(device)
        self.model: nn.Module = model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def __call__(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        valid_mask: Tensor | None = None,
        data_range: float = 1.0,
    ) -> Tensor:
        data_range = _validate_data_range(data_range)
        prediction, target, valid = _validated_inputs(
            prediction, target, valid_mask, data_range=data_range
        )
        values: list[Tensor] = []
        for index in range(prediction.shape[0]):
            spatial = valid[index, 0]
            coordinates = torch.nonzero(spatial, as_tuple=False)
            y_min, x_min = (int(value.item()) for value in coordinates.amin(dim=0))
            y_max, x_max = (
                int(value.item()) + 1 for value in coordinates.amax(dim=0)
            )
            image_mask = spatial[y_min:y_max, x_min:x_max][None, None]
            pred_crop = prediction[index : index + 1, :, y_min:y_max, x_min:x_max]
            target_crop = target[index : index + 1, :, y_min:y_max, x_min:x_max]
            pred_crop = torch.where(image_mask, pred_crop, torch.zeros_like(pred_crop))
            target_crop = torch.where(image_mask, target_crop, torch.zeros_like(target_crop))
            pred_crop = pred_crop.to(self.device, torch.float32) * 2.0 - 1.0
            target_crop = target_crop.to(self.device, torch.float32) * 2.0 - 1.0
            values.append(self.model(pred_crop, target_crop).mean())
        return torch.stack(values).mean().cpu()


@dataclass(frozen=True)
class ImageMetrics:
    psnr: float
    ssim: float
    lpips: float | None
    actor_psnr: float | None
    valid_pixels: int
    actor_pixels: int

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "psnr": self.psnr,
            "ssim": self.ssim,
            "lpips": self.lpips,
            "actor_psnr": self.actor_psnr,
            "valid_pixels": self.valid_pixels,
            "actor_pixels": self.actor_pixels,
        }


@torch.inference_mode()
def evaluate_image_pair(
    prediction: Tensor,
    target: Tensor,
    *,
    valid_mask: Tensor | None = None,
    actor_mask: Tensor | None = None,
    data_range: float = 1.0,
    ssim_window_size: int = 11,
    ssim_sigma: float = 1.5,
    lpips_metric: LPIPSMetric | None = None,
) -> ImageMetrics:
    """Evaluate one image or batch, including optional actor-box PSNR."""

    data_range = _validate_data_range(data_range)
    if ssim_window_size <= 0 or ssim_window_size % 2 == 0:
        raise ValueError("ssim_window_size must be a positive odd integer")
    if ssim_sigma <= 0:
        raise ValueError("ssim_sigma must be positive")
    prediction, target, valid = _validated_inputs(
        prediction, target, valid_mask, data_range=data_range
    )
    squared_error, value_count = _masked_squared_error(prediction, target, valid)
    mse = squared_error / value_count
    psnr_value = 10.0 * torch.log10(
        torch.ones((), dtype=mse.dtype, device=mse.device) / mse
    )
    ssim_value = _masked_structural_similarity(
        prediction,
        target,
        valid,
        window_size=ssim_window_size,
        sigma=ssim_sigma,
        data_range=1.0,
    )
    lpips_value = (
        float(
            lpips_metric(
                prediction, target, valid_mask=valid, data_range=1.0
            ).item()
        )
        if lpips_metric is not None
        else None
    )

    actor_psnr: float | None = None
    actor_pixels = 0
    if actor_mask is not None:
        actor = _to_mask(actor_mask, reference=prediction, name="actor_mask") & valid
        actor_pixels = int(actor.sum().item())
        if actor_pixels > 0:
            actor_squared_error, actor_value_count = _masked_squared_error(
                prediction, target, actor
            )
            actor_mse = actor_squared_error / actor_value_count
            actor_score = 10.0 * torch.log10(
                torch.as_tensor(
                    1.0, dtype=actor_mse.dtype, device=actor_mse.device
                )
                / actor_mse
            )
            actor_psnr = float(actor_score.item())

    return ImageMetrics(
        psnr=float(psnr_value.item()),
        ssim=float(ssim_value.item()),
        lpips=lpips_value,
        actor_psnr=actor_psnr,
        valid_pixels=int(valid.sum().item()),
        actor_pixels=actor_pixels,
    )


class EvaluationAccumulator:
    """Streaming, per-image mean accumulator for ArmGS evaluation metrics."""

    def __init__(
        self,
        *,
        data_range: float = 1.0,
        ssim_window_size: int = 11,
        ssim_sigma: float = 1.5,
        compute_lpips: bool = False,
        lpips_net: str = "alex",
        lpips_device: torch.device | str = "cpu",
        lpips_metric: LPIPSMetric | None = None,
    ) -> None:
        data_range = _validate_data_range(data_range)
        if ssim_window_size <= 0 or ssim_window_size % 2 == 0:
            raise ValueError("ssim_window_size must be a positive odd integer")
        if ssim_sigma <= 0:
            raise ValueError("ssim_sigma must be positive")
        if compute_lpips and lpips_metric is not None:
            raise ValueError("supply compute_lpips or lpips_metric, not both")
        self.data_range = data_range
        self.ssim_window_size = int(ssim_window_size)
        self.ssim_sigma = float(ssim_sigma)
        self.lpips_metric = (
            LPIPSMetric(net=lpips_net, device=lpips_device)
            if compute_lpips
            else lpips_metric
        )
        self._metric_sums = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
        self._actor_psnr_sum = 0.0
        self._num_images = 0
        self._num_actor_images = 0
        self._valid_pixels = 0
        self._actor_pixels = 0

    @torch.inference_mode()
    def update(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        valid_mask: Tensor | None = None,
        actor_mask: Tensor | None = None,
    ) -> None:
        prediction_nchw, target_nchw, valid = _validated_inputs(
            prediction, target, valid_mask, data_range=self.data_range
        )
        actor = (
            _to_mask(actor_mask, reference=prediction_nchw, name="actor_mask")
            if actor_mask is not None
            else None
        )
        for index in range(prediction_nchw.shape[0]):
            metrics = evaluate_image_pair(
                prediction_nchw[index],
                target_nchw[index],
                valid_mask=valid[index],
                actor_mask=actor[index] if actor is not None else None,
                data_range=1.0,
                ssim_window_size=self.ssim_window_size,
                ssim_sigma=self.ssim_sigma,
                lpips_metric=self.lpips_metric,
            )
            self._metric_sums["psnr"] += metrics.psnr
            self._metric_sums["ssim"] += metrics.ssim
            if metrics.lpips is not None:
                self._metric_sums["lpips"] += metrics.lpips
            if metrics.actor_psnr is not None:
                self._actor_psnr_sum += metrics.actor_psnr
                self._num_actor_images += 1
            self._valid_pixels += metrics.valid_pixels
            self._actor_pixels += metrics.actor_pixels
            self._num_images += 1

    def summary(self) -> dict[str, Any]:
        """Return a dictionary accepted by :func:`json.dumps`."""

        if self._num_images == 0:
            psnr_value: float | None = None
            ssim_value: float | None = None
            lpips_value: float | None = None
        else:
            psnr_value = self._metric_sums["psnr"] / self._num_images
            ssim_value = self._metric_sums["ssim"] / self._num_images
            lpips_value = (
                self._metric_sums["lpips"] / self._num_images
                if self.lpips_metric is not None
                else None
            )
        return {
            "num_images": self._num_images,
            "num_actor_images": self._num_actor_images,
            "valid_pixels": self._valid_pixels,
            "actor_pixels": self._actor_pixels,
            "psnr": psnr_value,
            "ssim": ssim_value,
            "lpips": lpips_value,
            "actor_psnr": (
                self._actor_psnr_sum / self._num_actor_images
                if self._num_actor_images > 0
                else None
            ),
        }


StreamingMetricAccumulator = EvaluationAccumulator


__all__ = [
    "EvaluationAccumulator",
    "ImageMetrics",
    "LPIPSMetric",
    "LPIPSUnavailableError",
    "StreamingMetricAccumulator",
    "evaluate_image_pair",
    "peak_signal_noise_ratio",
    "project_actor_boxes_to_mask",
    "psnr",
]
