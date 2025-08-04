"""Transform configuration utilities for crack segmentation.

This module provides utilities for creating transform pipelines from
configuration dictionaries and YAML files.
"""

from typing import Any

import albumentations as A
from omegaconf import DictConfig, ListConfig, OmegaConf


def get_transforms_from_config(
    config_list: (
        list[dict[str, Any] | DictConfig] | dict[str, Any] | DictConfig
    ),
    mode: str,
) -> A.Compose:
    """Create Albumentations transform pipeline from configuration.

    This function converts configuration dictionaries or lists into
    Albumentations Compose objects for use in datasets and data loaders.

    Args:
        config_list: Transform configuration as list of dicts, single dict,
            or OmegaConf DictConfig/ListConfig
        mode: Transform mode ('train', 'val', 'test') for validation

    Returns:
        Albumentations Compose object with configured transforms

    Examples:
        From list of transform configurations:
        ```python
        config = [
            {"name": "Resize", "params": {"height": 512, "width": 512}},
            {"name": "HorizontalFlip", "params": {"p": 0.5}},
            {"name": "Normalize", "params": {"mean": [0.485, 0.456, 0.406]}}
        ]
        transforms = get_transforms_from_config(config, "train")
        ```

        From single transform configuration:
        ```python
        config = {
            "name": "Resize",
            "params": {"height": 512, "width": 512}
        }
        transforms = get_transforms_from_config(config, "val")
        ```

        From OmegaConf configuration:
        ```python
        config = OmegaConf.create({
            "transforms": [
                {"name": "Resize", "params": {"height": 512, "width": 512}}
            ]
        })
        transforms = get_transforms_from_config(config.transforms, "test")
        ```

    Error Handling:
        - Invalid transform names raise ValueError
        - Missing required parameters raise ValueError
        - Invalid parameter values raise ValueError
        - Configuration format errors raise TypeError
    """
    # Normalize input to list format
    transform_specs = _get_transform_specs(config_list)

    # Convert to Albumentations transforms
    transforms = []
    for spec in transform_specs:
        transform = _create_transform_from_spec(spec)
        transforms.append(transform)

    return A.Compose(transforms)


def _get_transform_specs(
    config_list: (
        list[dict[str, Any] | DictConfig] | dict[str, Any] | DictConfig
    ),
) -> list[dict[str, Any] | DictConfig]:
    """Extract transform specifications from configuration input.

    Args:
        config_list: Transform configuration in various formats

    Returns:
        List of transform specifications

    Raises:
        TypeError: If configuration format is invalid
    """
    if isinstance(config_list, DictConfig | dict):
        # Single transform configuration
        return [config_list]
    elif isinstance(config_list, ListConfig | list):
        # List of transform configurations
        return list(config_list)
    else:
        raise TypeError(f"Unsupported config type: {type(config_list)}")


def _create_transform_from_spec(spec: Any) -> A.BasicTransform:
    """Create Albumentations transform from specification.

    Args:
        spec: Transform specification with 'name' and 'params' keys

    Returns:
        Albumentations transform object

    Raises:
        ValueError: If transform name is invalid or parameters are missing
    """
    # Convert to dict if needed
    if isinstance(spec, DictConfig):
        spec_dict = OmegaConf.to_container(spec, resolve=True)
        if not isinstance(spec_dict, dict):
            raise ValueError(
                f"Transform spec must be dict, got {type(spec_dict)}"
            )
        spec = spec_dict

    if not isinstance(spec, dict):
        raise ValueError(f"Transform spec must be dict, got {type(spec)}")

    name = spec.get("name")
    params = spec.get("params", {})

    if not name:
        raise ValueError("Transform spec must have 'name' key")

    # Get transform class from Albumentations
    transform_class = getattr(A, name, None)
    if transform_class is None:
        raise ValueError(f"Unknown transform: {name}")

    # Create transform instance
    try:
        return transform_class(**params)
    except Exception as e:
        raise ValueError(f"Failed to create transform {name}: {e}") from e
