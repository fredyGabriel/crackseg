"""Factory package public API with lazy imports.

Defers heavy imports until attributes are accessed to avoid importing
optional dependencies when only registries/utilities are needed.
"""

from __future__ import annotations

from typing import Any

from .factory_utils import validate_config
from .registry import Registry
from .registry_setup import component_registries

__all__: list[str] = [
    # Exposed lazily via __getattr__
    # Architecture helpers
    # "get_unet_class",
    # "add_final_activation",
    # "create_unet_from_config",
    # "create_unet",
    # Component helpers
    # "instantiate_unet_components",
    # "CBAMPostProcessor",
    # "FinalCBAMDecoder",
    # "create_cbam_module",
    # "apply_cbam_to_model",
    # "insert_cbam_if_enabled",
    # Errors
    # "ConfigurationError",
    # Utilities/registries
    "validate_config",
    "Registry",
    "component_registries",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - simple dispatch
    if name == "ConfigurationError":
        from .factory_utils import ConfigurationError as _ConfigurationError

        return _ConfigurationError

    # Architecture-level helpers (lazy)
    if name in {
        "get_unet_class",
        "add_final_activation",
        "create_unet_from_config",
        "create_unet",
    }:
        from . import architecture_factory as _arch

        return getattr(_arch, name)

    # Component-level helpers (lazy)
    if name in {
        "instantiate_unet_components",
        "CBAMPostProcessor",
        "FinalCBAMDecoder",
        "create_cbam_module",
        "apply_cbam_to_model",
        "insert_cbam_if_enabled",
    }:
        from . import component_factory as _comp

        return getattr(_comp, name)

    raise AttributeError(
        f"module 'crackseg.model.factory' has no attribute {name!r}"
    )
