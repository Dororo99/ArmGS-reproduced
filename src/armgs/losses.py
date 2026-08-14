"""ArmGS training objective from equation (9)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _to_nchw(images: Tensor) -> Tensor:
    if images.ndim == 3:
        if images.shape[0] == 3:
            return images.unsqueeze(0)
        if images.shape[-1] == 3:
            return images.permute(2, 0, 1).unsqueeze(0)
    if images.ndim == 4:
        if images.shape[1] == 3:
            return images
        if images.shape[-1] == 3:
            return images.permute(0, 3, 1, 2)
    raise ValueError("RGB images must have shape [3,H,W], [H,W,3], [B,3,H,W], or [B,H,W,3]")


def _safe_masked_mean(values: Tensor, mask: Tensor | None = None) -> Tensor:
    if mask is None:
        return values.mean()
    mask = mask.to(device=values.device, dtype=values.dtype)
    try:
        expanded = torch.broadcast_to(mask, values.shape)
    except RuntimeError as error:
        raise ValueError("mask is not broadcastable to values") from error
    denominator = expanded.sum()
    if denominator.detach().item() == 0:
        return values.sum() * 0.0
    return (values * expanded).sum() / denominator


def rgb_l1_loss(prediction: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("RGB prediction and target must have matching shapes")
    return _safe_masked_mean(torch.abs(prediction - target), mask)


def _gaussian_window(
    channels: int, window_size: int, sigma: float, reference: Tensor
) -> Tensor:
    coordinates = torch.arange(window_size, device=reference.device, dtype=reference.dtype)
    coordinates = coordinates - (window_size - 1) / 2.0
    kernel_1d = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def structural_similarity(
    prediction: Tensor,
    target: Tensor,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> Tensor:
    """Differentiable SSIM matching the conventional 3DGS objective."""

    prediction = _to_nchw(prediction)
    target = _to_nchw(target)
    if prediction.shape != target.shape:
        raise ValueError("SSIM inputs must have matching shapes")
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    channels = prediction.shape[1]
    window = _gaussian_window(channels, window_size, sigma, prediction)
    padding = window_size // 2

    mean_prediction = F.conv2d(
        prediction, window, padding=padding, groups=channels
    )
    mean_target = F.conv2d(target, window, padding=padding, groups=channels)
    mean_prediction_sq = mean_prediction.square()
    mean_target_sq = mean_target.square()
    mean_product = mean_prediction * mean_target

    variance_prediction = (
        F.conv2d(prediction * prediction, window, padding=padding, groups=channels)
        - mean_prediction_sq
    )
    variance_target = (
        F.conv2d(target * target, window, padding=padding, groups=channels)
        - mean_target_sq
    )
    covariance = (
        F.conv2d(prediction * target, window, padding=padding, groups=channels)
        - mean_product
    )
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = (
        (2.0 * mean_product + c1) * (2.0 * covariance + c2)
    ) / (
        (mean_prediction_sq + mean_target_sq + c1)
        * (variance_prediction + variance_target + c2)
    )
    return ssim_map.mean()


def depth_l1_loss(
    rendered_depth: Tensor, lidar_depth: Tensor, valid_mask: Tensor | None = None
) -> Tensor:
    if rendered_depth.shape != lidar_depth.shape:
        raise ValueError("rendered and LiDAR depth tensors must have matching shapes")
    finite = torch.isfinite(rendered_depth) & torch.isfinite(lidar_depth) & (lidar_depth > 0)
    if valid_mask is not None:
        finite = finite & valid_mask.to(dtype=torch.bool, device=finite.device)
    absolute_error = torch.abs(rendered_depth - lidar_depth)
    absolute_error = torch.where(finite, absolute_error, torch.zeros_like(absolute_error))
    return _safe_masked_mean(absolute_error, finite)


def sky_mask_loss(
    non_sky_accumulated_alpha: Tensor,
    target_sky_mask: Tensor,
    *,
    valid_mask: Tensor | None = None,
    eps: float = 1.0e-6,
) -> Tensor:
    if non_sky_accumulated_alpha.shape != target_sky_mask.shape:
        raise ValueError("alpha and sky mask must have matching shapes")
    sky_probability = (1.0 - non_sky_accumulated_alpha).clamp(eps, 1.0 - eps)
    per_pixel = F.binary_cross_entropy(
        sky_probability,
        target_sky_mask.to(sky_probability),
        reduction="none",
    )
    return _safe_masked_mean(per_pixel, valid_mask)


def foreground_entropy_loss(
    actor_alpha: Tensor,
    actor_bbox_mask: Tensor | None = None,
    *,
    eps: float = 1.0e-6,
) -> Tensor:
    """Entropy regularizer, optionally with StreetGS-style outside-box penalty."""

    probability = actor_alpha.clamp(eps, 1.0 - eps)
    entropy = -probability * probability.log() - (1.0 - probability) * (
        1.0 - probability
    ).log()
    if actor_bbox_mask is None:
        return entropy.mean()
    if actor_bbox_mask.shape != actor_alpha.shape:
        raise ValueError("actor_bbox_mask must match actor_alpha shape")
    inside = actor_bbox_mask.to(dtype=torch.bool, device=actor_alpha.device)
    outside_penalty = -(1.0 - probability).log()
    return torch.where(inside, entropy, outside_penalty).mean()


@dataclass(frozen=True)
class LossBreakdown:
    total: Tensor
    rgb: Tensor
    ssim: Tensor
    depth: Tensor
    sky: Tensor
    foreground: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        return {
            "loss": self.total,
            "rgb_loss": self.rgb,
            "ssim_loss": self.ssim,
            "depth_loss": self.depth,
            "sky_loss": self.sky,
            "foreground_loss": self.foreground,
        }


class ArmGSLoss(nn.Module):
    """Weighted combination in equation (9), with safe optional auxiliaries."""

    def __init__(
        self,
        *,
        lambda_ssim: float = 0.2,
        lambda_depth: float = 0.01,
        lambda_sky: float = 0.05,
        lambda_foreground: float = 0.1,
        require_auxiliary: bool = False,
        ssim_window_size: int = 11,
        ssim_sigma: float = 1.5,
        ssim_data_range: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= lambda_ssim <= 1.0:
            raise ValueError("lambda_ssim must be in [0,1]")
        if min(lambda_depth, lambda_sky, lambda_foreground) < 0.0:
            raise ValueError("auxiliary loss weights cannot be negative")
        self.lambda_ssim = lambda_ssim
        self.lambda_depth = lambda_depth
        self.lambda_sky = lambda_sky
        if ssim_window_size <= 0 or ssim_window_size % 2 == 0:
            raise ValueError("ssim_window_size must be a positive odd integer")
        if ssim_sigma <= 0 or ssim_data_range <= 0:
            raise ValueError("SSIM sigma and data range must be positive")
        self.lambda_foreground = lambda_foreground
        self.require_auxiliary = require_auxiliary
        self.ssim_window_size = int(ssim_window_size)
        self.ssim_sigma = float(ssim_sigma)
        self.ssim_data_range = float(ssim_data_range)

    def forward(
        self,
        prediction_rgb: Tensor,
        target_rgb: Tensor,
        *,
        rendered_depth: Tensor | None = None,
        lidar_depth: Tensor | None = None,
        depth_valid_mask: Tensor | None = None,
        non_sky_accumulated_alpha: Tensor | None = None,
        target_sky_mask: Tensor | None = None,
        sky_valid_mask: Tensor | None = None,
        actor_alpha: Tensor | None = None,
        actor_bbox_mask: Tensor | None = None,
        foreground_active: bool = True,
    ) -> LossBreakdown:
        if not isinstance(foreground_active, bool):
            raise TypeError("foreground_active must be a boolean")
        rgb = rgb_l1_loss(prediction_rgb, target_rgb)
        ssim = 1.0 - structural_similarity(
            prediction_rgb,
            target_rgb,
            window_size=self.ssim_window_size,
            sigma=self.ssim_sigma,
            data_range=self.ssim_data_range,
        )
        zero = prediction_rgb.sum() * 0.0
        if self.require_auxiliary:
            missing: list[str] = []
            if self.lambda_depth > 0.0 and (
                rendered_depth is None or lidar_depth is None
            ):
                missing.append("depth")
            if self.lambda_sky > 0.0 and (
                non_sky_accumulated_alpha is None or target_sky_mask is None
            ):
                missing.append("sky")
            if (
                foreground_active
                and self.lambda_foreground > 0.0
                and actor_alpha is None
            ):
                missing.append("foreground")
            if missing:
                raise ValueError("missing required auxiliary loss inputs: " + ", ".join(missing))

        if (rendered_depth is None) != (lidar_depth is None):
            raise ValueError("rendered_depth and lidar_depth must be supplied together")
        depth = (
            depth_l1_loss(rendered_depth, lidar_depth, depth_valid_mask)
            if rendered_depth is not None and lidar_depth is not None
            else zero
        )

        if (non_sky_accumulated_alpha is None) != (target_sky_mask is None):
            raise ValueError("non-sky alpha and target sky mask must be supplied together")
        if sky_valid_mask is not None and target_sky_mask is None:
            raise ValueError("sky_valid_mask requires a target sky mask")
        sky = (
            sky_mask_loss(
                non_sky_accumulated_alpha,
                target_sky_mask,
                valid_mask=sky_valid_mask,
            )
            if non_sky_accumulated_alpha is not None and target_sky_mask is not None
            else zero
        )
        if foreground_active:
            if actor_bbox_mask is not None and actor_alpha is None:
                raise ValueError("actor_bbox_mask requires actor_alpha")
            foreground = (
                foreground_entropy_loss(actor_alpha, actor_bbox_mask)
                if actor_alpha is not None
                else zero
            )
        else:
            foreground = zero

        total = (
            (1.0 - self.lambda_ssim) * rgb
            + self.lambda_ssim * ssim
            + self.lambda_depth * depth
            + self.lambda_sky * sky
            + self.lambda_foreground * foreground
        )
        return LossBreakdown(
            total=total,
            rgb=rgb,
            ssim=ssim,
            depth=depth,
            sky=sky,
            foreground=foreground,
        )
