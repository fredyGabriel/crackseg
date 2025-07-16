"""
Setup and registration of all available loss functions.
This module registers all loss implementations with the clean registry
using lazy loading to prevent circular dependencies.
"""

from .clean_registry import CleanLossRegistry


def setup_standard_losses(registry: CleanLossRegistry) -> None:
    """
    Register all standard loss implementations with the registry.

    Args:
        registry: Registry instance to populate
    """
    # Register Dice Loss
    registry.register_class(
        name="dice_loss",
        module_path="src.training.losses.dice_loss",
        class_name="DiceLoss",
        tags=["segmentation", "dice"],
        description="Dice loss for segmentation tasks",
    )

    # Register Focal Loss
    registry.register_class(
        name="focal_loss",
        module_path="src.training.losses.focal_loss",
        class_name="FocalLoss",
        tags=["segmentation", "focal", "class_imbalance"],
        description="Focal loss for handling class imbalance",
    )

    # Register BCE Loss
    registry.register_class(
        name="bce_loss",
        module_path="src.training.losses.bce_loss",
        class_name="BCELoss",
        tags=["segmentation", "binary"],
        description="Binary cross-entropy loss",
    )

    # Register BCE Dice Loss (Combined)
    registry.register_class(
        name="bce_dice_loss",
        module_path="src.training.losses.bce_dice_loss",
        class_name="BCEDiceLoss",
        tags=["segmentation", "combined", "binary"],
        description="Combined BCE and Dice loss",
    )

    # Register Combined Loss (Meta-loss)
    registry.register_class(
        name="combined_loss",
        module_path="src.training.losses.combined_loss",
        class_name="CombinedLoss",
        tags=["meta", "utility"],
        description="Meta-loss for combining multiple losses",
    )


def get_configured_registry() -> CleanLossRegistry:
    """
    Get a registry pre-configured with all standard losses.

    Returns:
        Configured registry instance
    """
    from . import registry

    # Only setup if not already configured
    if not registry.list_available():
        setup_standard_losses(registry)

    return registry
