"""Validation utilities for model factory configurations."""

from __future__ import annotations

from collections.abc import Mapping


def validate_component_config(
    config: Mapping[str, object], component_type: str
) -> None:
    """Validate a single component configuration.

    Args:
        config: Component configuration dictionary.
        component_type: One of "encoder", "bottleneck", or "decoder".

    Raises:
        ValueError: If required keys are missing or invalid.
    """
    if component_type == "encoder":
        if "in_channels" not in config:
            raise ValueError("Encoder config must specify 'in_channels'")
    elif component_type == "bottleneck":
        if "in_channels" not in config:
            raise ValueError("Bottleneck config must specify 'in_channels'")
    elif component_type == "decoder":
        required = ["in_channels"]
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(
                f"Decoder config missing required keys: {', '.join(missing)}"
            )


def validate_architecture_config(config: Mapping[str, object]) -> None:
    """Validate complete architecture configuration structure."""
    required_components = ["encoder", "bottleneck", "decoder"]
    missing = [comp for comp in required_components if comp not in config]
    if missing:
        raise ValueError(
            f"Architecture config missing components: {', '.join(missing)}"
        )

    # Validate each component
    enc_cfg = config["encoder"]
    bot_cfg = config["bottleneck"]
    dec_cfg = config["decoder"]
    if (
        not isinstance(enc_cfg, Mapping)
        or not isinstance(bot_cfg, Mapping)
        or not isinstance(dec_cfg, Mapping)
    ):
        raise ValueError("All component configs must be mappings")
    validate_component_config(enc_cfg, "encoder")
    validate_component_config(bot_cfg, "bottleneck")
    validate_component_config(dec_cfg, "decoder")


__all__ = [
    "validate_component_config",
    "validate_architecture_config",
]
