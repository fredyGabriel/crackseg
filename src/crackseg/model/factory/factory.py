"""Shim re-exporting factory helpers after split (component/architecture)."""

from __future__ import annotations

from .architecture_factory import (
    add_final_activation,
    create_unet,
    create_unet_from_config,
    get_unet_class,
)
from .component_factory import (
    CBAMPostProcessor,
    FinalCBAMDecoder,
    apply_cbam_to_model,
    create_cbam_module,
    insert_cbam_if_enabled,
    instantiate_unet_components,
)
from .factory_utils import ConfigurationError

__all__ = [
    # Architecture
    "get_unet_class",
    "add_final_activation",
    "create_unet_from_config",
    "create_unet",
    # Components
    "instantiate_unet_components",
    "CBAMPostProcessor",
    "FinalCBAMDecoder",
    "create_cbam_module",
    "apply_cbam_to_model",
    "insert_cbam_if_enabled",
    # Errors
    "ConfigurationError",
]
