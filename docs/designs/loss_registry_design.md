# Loss Function Registry Design

This document outlines the design for the loss function registry system in the `crackseg` project.

## 1. Overview

The project utilizes a generic, type-safe, and thread-safe `Registry` system, primarily defined in
`src.model.factory.registry.py`. For managing loss functions, a dedicated instance of this generic
`Registry` is configured.

This approach ensures consistency in how components are registered and managed across the project,
leveraging a robust and feature-rich registry system.

## 2. Core Registry Instance for Losses

- **Location**: `src.training.losses.loss_registry_setup.py`
- **Initialization**:

```python
    from src.model.factory.registry import Registry
    import torch.nn as nn

    loss_registry = Registry(base_class=nn.Module, name="LossFunctions")
    ```
-   **Base Class Requirement**: All loss functions intended for registration **must** be classes that inherit from `torch.nn.Module`.

## 3. Key Functionality (Provided by Generic `Registry`)

The `loss_registry` instance inherits all functionalities from the generic `Registry` class (`src.model.factory.registry.Registry`), including:

-   **Registration**: Via the `@loss_registry.register()` decorator.
    -   `@loss_registry.register(name: Optional[str] = None, tags: Optional[List[str]] = None)`
    -   `name`: Optional custom name for registration. Defaults to class `__name__`.
    -   `tags`: Optional list of strings for categorization (e.g., `["segmentation", "focal_loss"]`).
-   **Retrieval**: `loss_registry.get(name: str) -> Type[nn.Module]`
-   **Instantiation**: `loss_registry.instantiate(name: str, *args, **kwargs) -> nn.Module`
-   **Listing**: `loss_registry.list() -> List[str]`
-   **Listing with Tags**: `loss_registry.list_with_tags() -> Dict[str, List[str]]`
-   **Filtering by Tag**: `loss_registry.filter_by_tag(tag: str) -> List[str]`
-   **Error Handling**: Raises `TypeError` if a class doesn't inherit from `nn.Module` during registration, and `ValueError` or `KeyError` for registration conflicts or lookup failures.
-   **Thread Safety**: All operations are thread-safe, inherited from the generic `Registry`.

## 4. Naming Convention

-   **Registered Names**: If a custom `name` is provided to the decorator, it should be in `snake_case` (e.g., `"dice_loss"`). If no name is provided, the class name (e.g., `DiceLoss`) is used.
-   **Tags**: Should be `snake_case` strings (e.g., `"segmentation"`, `"focal_loss"`).

## 5. Usage Workflow

1.  **Define Loss Class**: Create a class inheriting from `torch.nn.Module`.
    ```python
    # src/training/losses/custom_losses.py
    import torch
    import torch.nn as nn
    from src.training.losses.loss_registry_setup import loss_registry

    @loss_registry.register(name="my_dice_loss", tags=["segmentation"])
    class MyDiceLoss(nn.Module):
        def __init__(self, smooth: float = 1.0):
            super().__init__()
            self.smooth = smooth

        def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            # ... implementation ...
            return loss
    ```
2.  **Configuration**: In configuration files (e.g., Hydra YAML), specify the loss by its registered name and parameters.
    ```yaml
    # Example Hydra config
    training:
      loss:
        name: my_dice_loss # Or the class name if no custom name was given
        params:
          smooth: 1.0e-6
    ```
3.  **Instantiation (via Factory)**: A Loss Factory (to be developed in Task 8) will use the `loss_registry` to look up and instantiate the loss module based on the configuration.

## 6. Backward Compatibility

-   This system replaces the previously drafted simpler registry.
-   Existing loss function *definitions* will need to be refactored into `nn.Module` classes if they are currently simple functions.
-   The instantiation point in the training pipeline will change to use the new Loss Factory.

## 7. Justification for Using Generic Registry

-   **Consistency**: Uses the same robust registry mechanism as other model components (encoders, decoders).
-   **Reusability**: Leverages existing, well-tested code, avoiding duplication of registry logic.
-   **Features**: Immediately gains features like thread-safety, tagging, and type-checking against a base class (`nn.Module`).
-   **Maintainability**: Reduces the overall amount of custom registry code in the project.

This design promotes a standardized and powerful way to manage loss functions, aligning with best practices already established in other parts of the `crackseg` codebase.
