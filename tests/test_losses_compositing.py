from __future__ import annotations

import pytest
import torch

from armgs.compositing import composite_sky, front_to_back_composite
from armgs.losses import (
    ArmGSLoss,
    depth_l1_loss,
    foreground_entropy_loss,
    sky_mask_loss,
)


def test_reference_alpha_compositing_and_sky() -> None:
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    alphas = torch.tensor([[0.5], [0.5]])
    result = front_to_back_composite(colors, alphas)

    torch.testing.assert_close(result.rgb, torch.tensor([0.5, 0.25, 0.0]))
    torch.testing.assert_close(result.accumulated_alpha, torch.tensor([0.75]))
    with_sky = composite_sky(
        result.rgb, result.accumulated_alpha, torch.tensor([0.0, 0.0, 1.0])
    )
    torch.testing.assert_close(with_sky, torch.tensor([0.5, 0.25, 0.25]))


def test_equation_nine_is_zero_for_perfect_rgb_without_auxiliaries() -> None:
    image = torch.rand(12, 13, 3)
    losses = ArmGSLoss()(image, image)

    torch.testing.assert_close(losses.rgb, torch.tensor(0.0))
    torch.testing.assert_close(losses.ssim, torch.tensor(0.0), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(losses.total, torch.tensor(0.0), atol=1.0e-6, rtol=0.0)


def test_strict_paper_loss_rejects_missing_auxiliaries() -> None:
    image = torch.rand(8, 9, 3)
    with pytest.raises(ValueError, match="depth, sky, foreground"):
        ArmGSLoss(require_auxiliary=True)(image, image)


def test_inactive_foreground_is_not_required_or_differentiated() -> None:
    prediction = torch.zeros(1, 4, 5, 3, requires_grad=True)
    target = torch.zeros_like(prediction)
    actor_alpha = torch.full(
        (1, 4, 5, 1), 0.2, requires_grad=True
    )
    loss = ArmGSLoss(
        lambda_ssim=0.0,
        lambda_depth=0.0,
        lambda_sky=0.0,
        lambda_foreground=1.0,
        require_auxiliary=True,
    )

    inactive = loss(
        prediction,
        target,
        actor_alpha=actor_alpha,
        actor_bbox_mask=torch.ones_like(actor_alpha, dtype=torch.bool),
        foreground_active=False,
    )
    inactive.total.backward()

    torch.testing.assert_close(inactive.foreground, torch.tensor(0.0))
    assert actor_alpha.grad is None

    active_alpha = actor_alpha.detach().requires_grad_()
    active = loss(
        prediction.detach(),
        target,
        actor_alpha=active_alpha,
        actor_bbox_mask=torch.ones_like(active_alpha, dtype=torch.bool),
        foreground_active=True,
    )
    active.total.backward()

    assert active.foreground > 0.0
    assert active_alpha.grad is not None
    assert torch.count_nonzero(active_alpha.grad) > 0


def test_inactive_foreground_keeps_other_strict_auxiliaries_required() -> None:
    image = torch.rand(8, 9, 3)
    with pytest.raises(ValueError, match=r"depth, sky$"):
        ArmGSLoss(require_auxiliary=True)(
            image,
            image,
            foreground_active=False,
        )


def test_empty_invalid_depth_mask_is_finite_zero() -> None:
    rendered = torch.tensor([float("nan")])
    target = torch.tensor([float("nan")])
    loss = depth_l1_loss(rendered, target, torch.tensor([False]))

    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_zero_lidar_depth_is_treated_as_invalid() -> None:
    rendered = torch.tensor([100.0, 3.0])
    target = torch.tensor([0.0, 2.0])
    loss = depth_l1_loss(rendered, target)
    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_sky_and_foreground_losses_have_expected_preferences() -> None:
    target_sky = torch.ones(2, 2, 1)
    good_sky = sky_mask_loss(torch.zeros_like(target_sky), target_sky)
    bad_sky = sky_mask_loss(torch.ones_like(target_sky), target_sky)
    assert good_sky < bad_sky

    crisp = foreground_entropy_loss(torch.tensor([1.0e-6, 1.0 - 1.0e-6]))
    uncertain = foreground_entropy_loss(torch.tensor([0.5, 0.5]))
    assert crisp < uncertain


def test_rejected_sky_validity_zeros_only_sky_loss_under_strict_mode() -> None:
    image = torch.zeros(1, 4, 5, 3)
    alpha = torch.full((1, 4, 5, 1), 0.8, requires_grad=True)
    target_sky = torch.ones_like(alpha)
    loss = ArmGSLoss(
        lambda_ssim=0.0,
        lambda_depth=0.0,
        lambda_sky=1.0,
        lambda_foreground=0.0,
        require_auxiliary=True,
    )

    rejected = loss(
        image,
        image,
        non_sky_accumulated_alpha=alpha,
        target_sky_mask=target_sky,
        sky_valid_mask=torch.tensor(False).reshape(1, 1, 1, 1),
    )
    rejected.total.backward()

    torch.testing.assert_close(rejected.sky, torch.tensor(0.0))
    assert alpha.grad is not None
    torch.testing.assert_close(alpha.grad, torch.zeros_like(alpha))

    valid = loss(
        image,
        image,
        non_sky_accumulated_alpha=alpha.detach(),
        target_sky_mask=target_sky,
        sky_valid_mask=torch.tensor(True).reshape(1, 1, 1, 1),
    )
    assert valid.sky > 0


def test_full_loss_backpropagates_and_stays_finite() -> None:
    prediction = torch.rand(1, 8, 9, 3, requires_grad=True)
    target = torch.zeros_like(prediction)
    rendered_depth = torch.ones(1, 8, 9, 1, requires_grad=True)
    lidar_depth = torch.full_like(rendered_depth, 2.0)
    alpha = torch.full((1, 8, 9, 1), 0.4, requires_grad=True)
    sky_mask = torch.zeros_like(alpha)
    actor_alpha = torch.full((1, 8, 9, 1), 0.2, requires_grad=True)

    losses = ArmGSLoss()(
        prediction,
        target,
        rendered_depth=rendered_depth,
        lidar_depth=lidar_depth,
        non_sky_accumulated_alpha=alpha,
        target_sky_mask=sky_mask,
        actor_alpha=actor_alpha,
    )
    losses.total.backward()

    assert torch.isfinite(losses.total)
    for tensor in (prediction, rendered_depth, alpha, actor_alpha):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
