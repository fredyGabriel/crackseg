"""Unit tests for FocalDiceLoss implementation."""

import pytest
import torch

from crackseg.training.losses.focal_dice_loss import (
    FocalDiceLoss,
    FocalDiceLossConfig,
)


class TestFocalDiceLoss:
    """Test cases for FocalDiceLoss."""

    def test_default_initialization(self):
        """Test FocalDiceLoss initialization with default config."""
        loss_fn = FocalDiceLoss()
        assert loss_fn is not None
        # Type assertion to help the type checker
        config: FocalDiceLossConfig = loss_fn.config  # type: ignore
        assert config.focal_weight == 0.6
        assert config.dice_weight == 0.4
        assert config.focal_alpha == 0.25
        assert config.focal_gamma == 2.0

    def test_custom_config_initialization(self):
        """Test FocalDiceLoss initialization with custom config."""
        config = FocalDiceLossConfig(
            focal_weight=0.7, dice_weight=0.3, focal_alpha=0.3, focal_gamma=2.5
        )
        loss_fn = FocalDiceLoss(config)
        # Type assertion to help the type checker
        test_config: FocalDiceLossConfig = loss_fn.config  # type: ignore
        assert test_config.focal_weight == 0.7
        assert test_config.dice_weight == 0.3
        assert test_config.focal_alpha == 0.3
        assert test_config.focal_gamma == 2.5

    def test_invalid_weights(self):
        """Test that invalid weights raise ValueError."""
        # Negative weights
        config = FocalDiceLossConfig(focal_weight=-0.1, dice_weight=0.4)
        with pytest.raises(
            ValueError, match="Loss weights must be non-negative"
        ):
            FocalDiceLoss(config)

        # Zero sum weights
        config = FocalDiceLossConfig(focal_weight=0.0, dice_weight=0.0)
        with pytest.raises(
            ValueError, match="Sum of loss weights must be positive"
        ):
            FocalDiceLoss(config)

    def test_forward_pass(self):
        """Test forward pass with valid inputs."""
        loss_fn = FocalDiceLoss()

        # Create dummy data
        batch_size, channels, height, width = 2, 1, 64, 64
        pred = torch.randn(batch_size, channels, height, width)
        target = torch.randint(
            0, 2, (batch_size, channels, height, width), dtype=torch.float32
        )

        # Forward pass
        loss = loss_fn(pred, target)

        # Check output
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar tensor
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.item() >= 0  # Loss should be non-negative

    def test_loss_registry_registration(self):
        """Test that FocalDiceLoss is properly registered."""
        from crackseg.training.losses.loss_registry_setup import loss_registry

        # Check if registered by trying to get the entry
        try:
            entry = loss_registry.get("focal_dice_loss")
            assert entry is not None
        except KeyError:
            # If not found, this is a test failure
            raise AssertionError(
                "focal_dice_loss not found in registry"
            ) from None

    def test_crack_segmentation_scenario(self):
        """Test with realistic crack segmentation scenario."""
        loss_fn = FocalDiceLoss()

        # Simulate crack segmentation data
        batch_size, channels, height, width = 4, 1, 256, 256

        # Create predictions (logits)
        pred = torch.randn(batch_size, channels, height, width)

        # Create targets with very few positive pixels (<5% as typical for
        # cracks)
        target = torch.zeros(batch_size, channels, height, width)

        # Add some crack-like patterns (thin lines)
        for b in range(batch_size):
            # Add horizontal crack
            target[b, 0, 128, 50:200] = 1.0
            # Add vertical crack
            target[b, 0, 50:200, 128] = 1.0

        # Forward pass
        loss = loss_fn(pred, target)

        # Check output
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.item() >= 0

    def test_gradient_flow(self):
        """Test that gradients flow properly through the loss."""
        loss_fn = FocalDiceLoss()

        # Create dummy data that requires gradients
        batch_size, channels, height, width = 2, 1, 64, 64
        pred = torch.randn(
            batch_size, channels, height, width, requires_grad=True
        )
        target = torch.randint(
            0, 2, (batch_size, channels, height, width), dtype=torch.float32
        )

        # Forward pass
        loss = loss_fn(pred, target)

        # Backward pass
        loss.backward()

        # Check gradients
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()
        assert not torch.isinf(pred.grad).any()

    def test_different_batch_sizes(self):
        """Test that loss works with different batch sizes."""
        loss_fn = FocalDiceLoss()

        for batch_size in [1, 2, 4, 8]:
            pred = torch.randn(batch_size, 1, 64, 64)
            target = torch.randint(
                0, 2, (batch_size, 1, 64, 64), dtype=torch.float32
            )

            loss = loss_fn(pred, target)
            assert isinstance(loss, torch.Tensor)
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)

    def test_different_image_sizes(self):
        """Test that loss works with different image sizes."""
        loss_fn = FocalDiceLoss()

        for size in [(32, 32), (64, 64), (128, 128), (256, 256)]:
            pred = torch.randn(2, 1, *size)
            target = torch.randint(0, 2, (2, 1, *size), dtype=torch.float32)

            loss = loss_fn(pred, target)
            assert isinstance(loss, torch.Tensor)
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
