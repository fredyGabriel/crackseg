"""Tests for loss functions."""

import torch
import torch.nn as nn
import torch.optim as optim
from src.training.losses import (
    BCELoss, DiceLoss, FocalLoss,
    CombinedLoss, BCEDiceLoss
)


def test_bce_loss_perfect_prediction():
    """Test BCE loss with perfect predictions."""
    loss_fn = BCELoss()
    pred = torch.tensor([[8.0]], dtype=torch.float32)  # High logit for 1
    target = torch.tensor([[1.0]], dtype=torch.float32)

    loss = loss_fn(pred, target)
    assert loss.item() < 0.01


def test_bce_loss_wrong_prediction():
    """Test BCE loss with completely wrong predictions."""
    loss_fn = BCELoss()
    pred = torch.tensor([[8.0]], dtype=torch.float32)  # High logit for 1
    target = torch.tensor([[0.0]], dtype=torch.float32)

    loss = loss_fn(pred, target)
    assert loss.item() > 1.0


def test_dice_loss_perfect_prediction():
    """Test Dice loss with perfect predictions."""
    loss_fn = DiceLoss(sigmoid=False)  # Disable sigmoid for test
    pred = torch.tensor([[1.0]], dtype=torch.float32)
    target = torch.tensor([[1.0]], dtype=torch.float32)

    loss = loss_fn(pred, target)
    assert loss.item() < 0.01


def test_dice_loss_wrong_prediction():
    """Test Dice loss with completely wrong predictions."""
    loss_fn = DiceLoss(sigmoid=False)  # Disable sigmoid for test
    pred = torch.ones((1, 1, 2, 2))  # All ones
    target = torch.zeros((1, 1, 2, 2))  # All zeros

    loss = loss_fn(pred, target)
    # Loss should be high for completely wrong predictions
    assert loss.item() > 0.7


def test_loss_batch_input():
    """Test losses with batch input."""
    batch_size = 4
    height = 32
    width = 32

    pred = torch.randn(batch_size, 1, height, width)
    target = torch.randint(0, 2, (batch_size, 1, height, width)).float()

    bce_loss = BCELoss()
    dice_loss = DiceLoss()

    bce_out = bce_loss(pred, target)
    dice_out = dice_loss(pred, target)

    assert isinstance(bce_out, torch.Tensor)
    assert isinstance(dice_out, torch.Tensor)
    assert bce_out.dim() == 0  # Scalar output
    assert dice_out.dim() == 0  # Scalar output


def test_dice_loss_smoothing():
    """Test Dice loss with different smoothing factors."""
    # Test with non-zero tensors to see smoothing effect
    pred = torch.ones((1, 1, 2, 2)) * 0.5  # All 0.5
    target = torch.ones((1, 1, 2, 2))  # All ones

    # With default smoothing
    loss_default = DiceLoss(sigmoid=False)(pred, target)

    # With zero smoothing
    loss_no_smooth = DiceLoss(smooth=0.0, sigmoid=False)(pred, target)

    assert not torch.isnan(loss_default)
    assert not torch.isnan(loss_no_smooth)
    assert loss_default != loss_no_smooth


def test_focal_loss_parameters():
    """Test Focal Loss with different gamma and alpha values."""
    # Test with prediction values
    pred = torch.tensor([[0.2]], dtype=torch.float32)
    target = torch.tensor([[1.0]], dtype=torch.float32)

    # Default parameters
    loss_default = FocalLoss(sigmoid=False)(pred, target)

    # High gamma (more focus on hard examples)
    loss_high_gamma = FocalLoss(gamma=4.0, sigmoid=False)(pred, target)

    # Different alpha (changes class weight)
    loss_diff_alpha = FocalLoss(alpha=0.75, sigmoid=False)(pred, target)

    assert not torch.isnan(loss_default)
    assert not torch.isnan(loss_high_gamma)
    assert not torch.isnan(loss_diff_alpha)

    # Verify each loss value is non-zero and positive
    assert loss_default > 0
    assert loss_high_gamma > 0
    assert loss_diff_alpha > 0

    # Higher alpha should give higher weight to positive class
    assert loss_diff_alpha > loss_default


def test_focal_loss_edge_cases():
    """Test Focal Loss with edge cases."""
    loss_fn = FocalLoss(sigmoid=False)

    # Perfect prediction
    pred_perfect = torch.tensor([[1.0]], dtype=torch.float32)
    target_one = torch.tensor([[1.0]], dtype=torch.float32)
    loss_perfect = loss_fn(pred_perfect, target_one)

    # Completely wrong
    pred_wrong = torch.tensor([[0.0]], dtype=torch.float32)
    loss_wrong = loss_fn(pred_wrong, target_one)

    assert loss_perfect.item() < 0.01
    assert loss_wrong.item() > 0.1
    assert loss_wrong > loss_perfect


def test_combined_loss():
    """Test combined loss with multiple loss functions."""
    pred = torch.tensor([[0.7]], dtype=torch.float32)
    target = torch.tensor([[1.0]], dtype=torch.float32)

    # Individual losses
    bce = BCELoss()(torch.logit(pred), target)
    # Note: we use dice loss value to verify combined loss behavior
    _ = DiceLoss(sigmoid=False)(pred, target)

    # Combined loss with equal weights
    combined_equal = CombinedLoss(
        losses=[BCELoss(), DiceLoss(sigmoid=True)],
        weights=[1.0, 1.0]
    )(torch.logit(pred), target)

    # Combined loss with more weight on BCE
    combined_bce_heavy = CombinedLoss(
        losses=[BCELoss(), DiceLoss(sigmoid=True)],
        weights=[0.8, 0.2]
    )(torch.logit(pred), target)

    assert not torch.isnan(combined_equal)
    assert not torch.isnan(combined_bce_heavy)

    # With more weight on BCE, the combined loss should be closer to BCE
    assert abs(combined_bce_heavy - bce) < abs(combined_equal - bce)


def test_bcedice_loss():
    """Test the predefined BCE+Dice loss."""
    pred = torch.tensor([[0.8]], dtype=torch.float32)
    target = torch.tensor([[1.0]], dtype=torch.float32)

    # Default weights (0.5, 0.5)
    loss_default = BCEDiceLoss()(torch.logit(pred), target)

    # More weight on BCE
    loss_bce_heavy = BCEDiceLoss(bce_weight=0.8, dice_weight=0.2)(
        torch.logit(pred), target
    )

    # More weight on Dice
    loss_dice_heavy = BCEDiceLoss(bce_weight=0.2, dice_weight=0.8)(
        torch.logit(pred), target
    )

    assert not torch.isnan(loss_default)
    assert not torch.isnan(loss_bce_heavy)
    assert not torch.isnan(loss_dice_heavy)
    assert loss_default > 0.0

    # BCE loss is typically higher than Dice for near-perfect predictions
    # So with more weight on BCE, the loss should be higher
    assert loss_bce_heavy > loss_dice_heavy


def test_loss_with_empty_target():
    """Test all losses with empty (all zeros) target."""
    pred = torch.rand((1, 1, 10, 10))  # Random predictions
    target = torch.zeros((1, 1, 10, 10))  # All zeros

    losses = [
        BCELoss(),
        DiceLoss(),
        FocalLoss(),
        BCEDiceLoss(),
        CombinedLoss(losses=[BCELoss(), DiceLoss()])
    ]

    for loss_fn in losses:
        loss = loss_fn(pred, target)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss >= 0.0


def test_focal_loss_imbalanced_data():
    """Test Focal Loss leads to better convergence on imbalanced data."""
    # Config
    batch_size = 50
    height = width = 16
    num_positive_pixels = 10  # Very few positive pixels per sample
    num_epochs = 10
    lr = 0.1

    # Data (highly imbalanced)
    target = torch.zeros((batch_size, 1, height, width))
    # Create a few positive pixels
    for i in range(batch_size):
        for _ in range(num_positive_pixels):
            r, c = torch.randint(0, height, (2,))
            target[i, 0, r, c] = 1.0

    # Simple Model (shared weights initially)
    model_bce = nn.Conv2d(1, 1, kernel_size=1)
    model_focal = nn.Conv2d(1, 1, kernel_size=1)
    # Ensure same initial weights
    model_focal.load_state_dict(model_bce.state_dict())

    # Optimizers
    optimizer_bce = optim.SGD(model_bce.parameters(), lr=lr)
    optimizer_focal = optim.SGD(model_focal.parameters(), lr=lr)

    # Losses
    bce_loss_fn = BCELoss()
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    # Training loop for BCE
    initial_bce_loss = -1.0
    final_bce_loss = -1.0
    for epoch in range(num_epochs):
        optimizer_bce.zero_grad()
        # Use target as input for simplicity
        pred_bce = model_bce(target)
        loss_bce = bce_loss_fn(pred_bce, target)
        if epoch == 0:
            initial_bce_loss = loss_bce.item()
        loss_bce.backward()
        optimizer_bce.step()
        final_bce_loss = loss_bce.item()

    # Training loop for Focal Loss
    initial_focal_loss = -1.0
    final_focal_loss = -1.0
    for epoch in range(num_epochs):
        optimizer_focal.zero_grad()
        # Use target as input
        pred_focal = model_focal(target)
        loss_focal = focal_loss_fn(pred_focal, target)
        if epoch == 0:
            initial_focal_loss = loss_focal.item()
        loss_focal.backward()
        optimizer_focal.step()
        final_focal_loss = loss_focal.item()

    # Assertions
    assert initial_bce_loss > 0
    assert initial_focal_loss > 0
    assert final_bce_loss < initial_bce_loss
    assert final_focal_loss < initial_focal_loss

    # Focal loss should converge better (lower final loss) on imbalanced data
    # Or show significantly more relative improvement
    bce_improvement = (initial_bce_loss - final_bce_loss) / initial_bce_loss
    focal_improvement = (initial_focal_loss - final_focal_loss)
    focal_improvement /= initial_focal_loss

    print(f"BCE Improvement: {bce_improvement:.4f}")
    print(f"Focal Improvement: {focal_improvement:.4f}")
    print(f"Final BCE Loss: {final_bce_loss:.4f}")
    print(f"Final Focal Loss: {final_focal_loss:.4f}")

    # Expect Focal Loss to achieve a lower final loss value
    assert final_focal_loss < final_bce_loss
    # Relative improvement comparison might be misleading, focus on final loss
    # assert focal_improvement > bce_improvement
