"""Image-quality evaluation utilities for ArmGS reconstructions.

The paper reports PSNR, SSIM, LPIPS, and PSNR inside projected actor boxes.  This
module keeps those evaluation choices independent from the training objective and
supports streaming over scenes that do not fit in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import torch
from torch import Tensor, nn

from .losses import structural_similarity


class LPIPSUnavailableError(RuntimeError):
    """Raised when LPIPS was requested but its optional package is unavailable."""


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
    if not images.is_floating_point():
        images = images.to(torch.float32)
    elif images.device.type == "cpu" and images.dtype in (torch.float16, torch.bfloat16):
        # CPU conv2d support for low precision depends on the installed Torch build.
        images = images.to(torch.float32)
    return images.contiguous()


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
) -> tuple[Tensor, Tensor, Tensor]:
    prediction = _to_nchw(prediction, name="prediction")
    target = _to_nchw(target, name="target").to(
        device=prediction.device, dtype=prediction.dtype
    )
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have matching shapes")
    valid = _to_mask(valid_mask, reference=prediction, name="valid_mask")
    if not valid.any():
        raise ValueError("valid_mask selects no pixels")
    selected = valid.expand_as(prediction)
    if not torch.isfinite(prediction.masked_select(selected)).all():
        raise ValueError("prediction contains non-finite values in the valid region")
    if not torch.isfinite(target.masked_select(selected)).all():
        raise ValueError("target contains non-finite values in the valid region")
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

    if data_range <= 0:
        raise ValueError("data_range must be positive")
    prediction, target, valid = _validated_inputs(prediction, target, valid_mask)
    squared_error, count = _masked_squared_error(prediction, target, valid)
    mse = squared_error / count
    return 10.0 * torch.log10(
        torch.as_tensor(data_range**2, dtype=mse.dtype, device=mse.device) / mse
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
        if data_range <= 0:
            raise ValueError("data_range must be positive")
        prediction, target, valid = _validated_inputs(prediction, target, valid_mask)
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
            pred_crop = pred_crop.to(self.device, torch.float32) / data_range * 2.0 - 1.0
            target_crop = target_crop.to(self.device, torch.float32) / data_range * 2.0 - 1.0
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

    if data_range <= 0:
        raise ValueError("data_range must be positive")
    if ssim_window_size <= 0 or ssim_window_size % 2 == 0:
        raise ValueError("ssim_window_size must be a positive odd integer")
    if ssim_sigma <= 0:
        raise ValueError("ssim_sigma must be positive")
    prediction, target, valid = _validated_inputs(prediction, target, valid_mask)
    squared_error, value_count = _masked_squared_error(prediction, target, valid)
    mse = squared_error / value_count
    psnr_value = 10.0 * torch.log10(
        torch.as_tensor(data_range**2, dtype=mse.dtype, device=mse.device) / mse
    )
    ssim_value = _masked_structural_similarity(
        prediction,
        target,
        valid,
        window_size=ssim_window_size,
        sigma=ssim_sigma,
        data_range=data_range,
    )
    lpips_value = (
        float(
            lpips_metric(
                prediction, target, valid_mask=valid, data_range=data_range
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
                    data_range**2, dtype=actor_mse.dtype, device=actor_mse.device
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
        if data_range <= 0:
            raise ValueError("data_range must be positive")
        if ssim_window_size <= 0 or ssim_window_size % 2 == 0:
            raise ValueError("ssim_window_size must be a positive odd integer")
        if ssim_sigma <= 0:
            raise ValueError("ssim_sigma must be positive")
        if compute_lpips and lpips_metric is not None:
            raise ValueError("supply compute_lpips or lpips_metric, not both")
        self.data_range = float(data_range)
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
            prediction, target, valid_mask
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
                data_range=self.data_range,
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
    "psnr",
]
