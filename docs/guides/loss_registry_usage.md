# Registering and Using Loss Functions

This guide explains how to register custom loss functions with the project's loss registry and how to instantiate them, typically via a loss factory.

The project uses a generic, type-safe, and thread-safe `Registry` system (defined in `src.model.factory.registry.py`). A specific instance of this registry is configured for loss functions in `src.training.losses.loss_registry_setup.py`.

## Key Principles for Loss Functions

- **Must be `torch.nn.Module` classes**: To be registered, your loss function must be a class that inherits from `torch.nn.Module`.
- **Registry Instance**: Import the dedicated loss registry instance: `from src.training.losses.loss_registry_setup import loss_registry`.

## Registering a New Loss Function (Module)

To make your custom loss function (as an `nn.Module`) available, use the `register` decorator from the `loss_registry` instance.

### Steps

1. **Define your loss class**, inheriting from `torch.nn.Module`.
2. **Apply the decorator**:
    *Place `@loss_registry.register()` directly above your class definition.
    *By default, the class's `__name__` will be used as its registration name.
    *You can provide a custom name: `@loss_registry.register(name="my_custom_loss")`.
    *You can also provide `tags` for categorization: `@loss_registry.register(name="dice_loss", tags=["segmentation", "binary"])`.

### Naming Convention for Registration

*Use `snake_case` for registration names if providing a custom name (e.g., `dice_loss`). If not providing a name, the class name (e.g., `DiceLoss`) will be used.
*Tags should also be `snake_case`.

### Example: Registering a Class-based Loss

```python
# src/training/losses/custom_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the specific loss_registry instance
from src.training.losses.loss_registry_setup import loss_registry

@loss_registry.register(name="weighted_mse_loss", tags=["regression"])
class WeightedMSELoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(predictions, targets, reduction='none')
        return (loss * self.weight).mean()

@loss_registry.register(tags=["segmentation", "dice"])
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions_flat = predictions.view(-1)
        targets_flat = targets.view(-1)
        intersection = (predictions_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (predictions_flat.sum() + targets_flat.sum() + self.smooth)
        return 1 - dice
```

## Using Registered Losses

Registered losses are typically instantiated via a Loss Factory (to be detailed in Task 8 documentation), which uses this registry internally. You generally won't call `loss_registry.get()` or `loss_registry.instantiate()` directly in training scripts, but rather define your desired loss in a configuration file.

The factory will look up loss functions by their registered names and instantiate them with parameters from the configuration files.

### Listing Available Losses

You can inspect available losses programmatically:

```python
from src.training.losses.loss_registry_setup import loss_registry

print(loss_registry.list()) # Prints all registered loss names
print(loss_registry.list_with_tags()) # Prints names and their tags
print(loss_registry.filter_by_tag("segmentation")) # Prints losses tagged with "segmentation"
```
