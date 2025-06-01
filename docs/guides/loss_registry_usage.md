# Registering and Using Loss Functions

This guide explains how to register custom loss functions with the project's registry system and how to instantiate them, typically via a loss factory.

The project uses a generic, type-safe, and thread-safe `Registry` system (defined in `src.model.factory.registry.py`). A specific instance of this registry is configured for loss functions in `src.training.losses.loss_registry_setup.py`.

## Key Principles for Loss Functions

- **Must be `torch.nn.Module` classes**: To be registered, your loss function must be a class that inherits from `torch.nn.Module`.
- **Registry Instance**: Import the dedicated loss registry instance: `from src.training.losses.loss_registry_setup import loss_registry`.
- **Mandatory Type Annotations**: Follow our [code standards](../../.cursor/rules/coding-preferences.mdc) with complete type hints.

## Registering a New Loss Function (Module)

To make your custom loss function (as an `nn.Module`) available, use the `register` decorator from the `loss_registry` instance.

### Steps

1. **Define your loss class**, inheriting from `torch.nn.Module`.
2. **Apply the decorator**:
   - Place `@loss_registry.register()` directly above your class definition.
   - By default, the class's `__name__` will be used as its registration name.
   - You can provide a custom name: `@loss_registry.register(name="my_custom_loss")`.
   - You can also provide `tags` for categorization: `@loss_registry.register(name="dice_loss", tags=["segmentation", "binary"])`.

### Naming Convention for Registration

- Use `snake_case` for registration names if providing a custom name (e.g., `dice_loss`). If not providing a name, the class name (e.g., `DiceLoss`) will be used.
- Tags should also be `snake_case`.

### Example: Registering a Class-based Loss

```python
# src/training/losses/custom_losses.py
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the specific loss_registry instance
from src.training.losses.loss_registry_setup import loss_registry

@loss_registry.register(name="weighted_mse_loss", tags=["regression"])
class WeightedMSELoss(nn.Module):
    """Weighted MSE Loss for regression with adjustable weights.

    Args:
        weight: Weighting factor for the loss
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the weighted MSE loss.

        Args:
            predictions: Prediction tensor of shape (N, *)
            targets: Target tensor of shape (N, *)
        Returns:
            Scalar tensor with the computed loss
        """
        loss = F.mse_loss(predictions, targets, reduction='none')
        return (loss * self.weight).mean()

@loss_registry.register(tags=["segmentation", "dice"])
class DiceLoss(nn.Module):
    """Dice Loss implementation for binary segmentation.

    Optimized for pavement crack segmentation.

    Args:
        smooth: Smoothing factor to avoid division by zero
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the Dice coefficient and returns 1 - dice as loss.

        Args:
            predictions: Prediction tensor of shape (N, C, H, W)
            targets: Mask tensor of shape (N, C, H, W) or (N, H, W)
        Returns:
            Scalar tensor with the Dice loss
        Raises:
            ValueError: If input shapes are not compatible
        """
        if predictions.dim() != targets.dim():
            if targets.dim() == 3 and predictions.dim() == 4:
                targets = targets.unsqueeze(1)  # Add channel dimension
            else:
                raise ValueError(
                    f"Incompatible shapes: predictions {predictions.shape}, "
                    f"targets {targets.shape}"
                )

        predictions_flat = predictions.view(-1)
        targets_flat = targets.view(-1)
        intersection = (predictions_flat * targets_flat).sum()

        dice_coeff = (2.0 * intersection + self.smooth) / (
            predictions_flat.sum() + targets_flat.sum() + self.smooth
        )

        return 1.0 - dice_coeff

@loss_registry.register(name="focal_loss", tags=["segmentation", "imbalanced"])
class FocalLoss(nn.Module):
    """Focal Loss to handle class imbalance in segmentation.

    Especially useful for pavement cracks where the background
    dominates over the cracks (minority class).

    Args:
        alpha: Class balance factor
        gamma: Focusing parameter for hard examples
        reduction: Reduction type ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes Focal Loss.

        Args:
            predictions: Logits tensor of shape (N, C, H, W)
            targets: Label tensor of shape (N, H, W)
        Returns:
            Tensor with the computed focal loss
        """
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
```

## Using Registered Losses

Registered losses are typically instantiated via a Loss Factory, which uses this registry internally. You generally will not call `loss_registry.get()` or `loss_registry.instantiate()` directly in training scripts, but rather define your desired loss in configuration files.

The factory will look up loss functions by their registered names and instantiate them with parameters from the configuration files.

### YAML Configuration Example

```yaml
# configs/training/loss/dice_focal.yaml
defaults:
  - base_loss

loss:
  _target_: src.training.losses.combined_loss.CombinedLoss
  losses:
    dice:
      _target_: DiceLoss  # Registered name
      smooth: 1.0
    focal:
      _target_: focal_loss  # Custom registered name
      alpha: 1.0
      gamma: 2.0
  weights: [0.5, 0.5]
```

### Integration with Quality Standards

All loss functions must comply with our [code standards](../../.cursor/rules/coding-preferences.mdc):

```python
# ✅ Correct: Complete type hints, validation, documentation
@loss_registry.register(name="combined_dice_bce", tags=["segmentation", "combined"])
class CombinedDiceBCE(nn.Module):
    """Combination of Dice Loss and Binary Cross Entropy for segmentation.

    This combination is especially effective for crack segmentation
    where both edge precision (Dice) and class separation (BCE) are needed.
    """

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5) -> None:
        super().__init__()
        if dice_weight + bce_weight != 1.0:
            raise ValueError("Weights must sum to 1.0")

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Combines Dice and BCE loss with specified weights."""
        dice = self.dice_loss(predictions, targets)
        bce = F.binary_cross_entropy_with_logits(predictions, targets)

        return self.dice_weight * dice + self.bce_weight * bce

# ❌ Incorrect: No type hints, no validation, no documentation
@loss_registry.register()
class BadLoss(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param

    def forward(self, pred, target):
        return F.mse_loss(pred, target) * self.param
```

### Listing Available Losses

You can inspect available losses programmatically:

```python
from src.training.losses.loss_registry_setup import loss_registry
from typing import List, Dict

# List all registered losses
available_losses: List[str] = loss_registry.list()
print("Available losses:", available_losses)

# List with tags
losses_with_tags: Dict[str, List[str]] = loss_registry.list_with_tags()
print("Losses with tags:", losses_with_tags)

# Filter by specific tag
segmentation_losses: List[str] = loss_registry.filter_by_tag("segmentation")
print("Segmentation losses:", segmentation_losses)

# Get detailed information
for loss_name in segmentation_losses:
    loss_class = loss_registry.get(loss_name)
    print(f"{loss_name}: {loss_class.__doc__}")
```

## Testing Loss Functions

Following our [testing standards](../../.cursor/rules/testing-standards.mdc), all loss functions must include tests:

```python
# tests/unit/training/losses/test_custom_losses.py
import torch
import pytest
from src.training.losses.custom_losses import DiceLoss, FocalLoss

class TestDiceLoss:
    """Tests for DiceLoss following project standards."""

    def test_dice_loss_perfect_prediction(self) -> None:
        """Test with perfect prediction should yield loss ≈ 0."""
        loss_fn = DiceLoss(smooth=1.0)
        predictions = torch.ones(2, 1, 4, 4)
        targets = torch.ones(2, 1, 4, 4)

        loss = loss_fn(predictions, targets)

        assert loss.item() < 0.1  # Almost zero with smooth=1.0

    def test_dice_loss_worst_prediction(self) -> None:
        """Test with opposite prediction should yield loss ≈ 1."""
        loss_fn = DiceLoss(smooth=1.0)
        predictions = torch.ones(2, 1, 4, 4)
        targets = torch.zeros(2, 1, 4, 4)

        loss = loss_fn(predictions, targets)

        assert loss.item() > 0.9  # Almost one

    def test_dice_loss_shape_compatibility(self) -> None:
        """Test automatic shape compatibility."""
        loss_fn = DiceLoss()
        predictions = torch.rand(2, 1, 4, 4)  # (N, C, H, W)
        targets = torch.randint(0, 2, (2, 4, 4)).float()  # (N, H, W)

        # Should not raise exception
        loss = loss_fn(predictions, targets)
        assert isinstance(loss, torch.Tensor)

    def test_dice_loss_invalid_shapes(self) -> None:
        """Test with incompatible shapes should fail."""
        loss_fn = DiceLoss()
        predictions = torch.rand(2, 1, 4, 4)
        targets = torch.rand(3, 4, 4)  # Different batch size

        with pytest.raises(ValueError, match="Incompatible shapes"):
            loss_fn(predictions, targets)
```

## Integration with ML Research Standards

For research-specific loss functions, see our [ML standards](../../.cursor/rules/ml-research-standards.mdc):

- **Reproducibility**: Document hyperparameters and behavior
- **Evaluation**: Include performance metrics and comparisons
- **VRAM Optimization**: Consider memory usage for RTX 3070 Ti

---

**Note**: This guide is kept in sync with our professional development standards. For updates or specific questions, see the rules in `.cursor/rules/` or open an issue in the repository.
