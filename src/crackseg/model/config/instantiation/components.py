"""
Component instantiation functions for model components.

This module contains the core instantiation logic for individual
model components (encoder, bottleneck, decoder, etc.).
"""

import logging
from typing import Any

from torch import nn

from crackseg.model.factory.registry import Registry
from crackseg.utils.component_cache import (
    cache_component,
    generate_cache_key,
    get_cached_component,
)

log = logging.getLogger(__name__)


class InstantiationError(Exception):
    """Exception raised when component instantiation fails."""

    pass


def _prepare_component_config(
    config: dict[str, Any],
    component_category: str,
    runtime_params: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    """Prepares and cleans the component configuration."""
    component_type = config.get("type")
    target_path = config.get("_target_")

    if not component_type and target_path:
        try:
            component_type = target_path.split(".")[-1]
            log.debug(
                f"Inferred component type '{component_type}' from "
                f"_target_ '{target_path}'"
            )
        except IndexError as exc:
            raise InstantiationError(
                f"Could not infer component type from _target_: {target_path}"
            ) from exc

    if not component_type:
        raise InstantiationError(
            f"{component_category.capitalize()} config must specify 'type' "
            f"or '_target_'"
        )

    config_copy = config.copy()
    config_copy.pop("type", None)
    config_copy.pop("_target_", None)

    if component_category == "decoder":
        if config_copy.pop("out_channels", None) is not None:
            log.debug(
                "Removed 'out_channels' from decoder config before "
                "instantiation."
            )

    if runtime_params:
        config_copy.update(runtime_params)

    return component_type, config_copy


def _instantiate_component(
    config: dict[str, Any],
    registry: Registry[nn.Module],
    component_category: str,
    runtime_params: dict[str, Any] | None = None,
    use_cache: bool = True,
) -> nn.Module:
    """
    Instantiate a component from its configuration.

    Args:
        config: Component configuration dictionary
        registry: Component registry to use for instantiation
        component_category: Category of the component (for logging/errors)
        runtime_params: Additional runtime parameters to merge
        use_cache: Whether to use component caching

    Returns:
        Instantiated component

    Raises:
        InstantiationError: If instantiation fails
    """
    try:
        component_type, clean_config = _prepare_component_config(
            config, component_category, runtime_params
        )

        if use_cache:
            cache_key = generate_cache_key(component_type, clean_config)
            cached_component = get_cached_component(cache_key)
            if cached_component is not None:
                log.debug(
                    f"Using cached {component_category} component: {component_type}"
                )
                return cached_component

        # Get component class from registry
        component_class = registry.get(component_type)
        if component_class is None:
            available_components = list(registry.keys())
            raise InstantiationError(
                f"Unknown {component_category} type '{component_type}'. "
                f"Available types: {available_components}"
            )

        # Instantiate component
        component = component_class(**clean_config)

        # Cache the component if caching is enabled
        if use_cache:
            cache_key = generate_cache_key(component_type, clean_config)
            cache_component(cache_key, component)
            log.debug(
                f"Cached {component_category} component: {component_type}"
            )

        log.debug(
            f"Successfully instantiated {component_category}: {component_type}"
        )
        return component

    except Exception as e:
        log.error(
            f"Failed to instantiate {component_category} from config: {config}"
        )
        if isinstance(e, InstantiationError):
            raise
        raise InstantiationError(
            f"Failed to instantiate {component_category}: {str(e)}"
        ) from e


def instantiate_encoder(
    config: dict[str, Any], use_cache: bool = True
) -> nn.Module:
    """
    Instantiate an encoder component.

    Args:
        config: Encoder configuration
        use_cache: Whether to use component caching

    Returns:
        Instantiated encoder
    """
    from crackseg.model.factory.registry_setup import encoder_registry

    return _instantiate_component(
        config, encoder_registry, "encoder", use_cache=use_cache
    )


def instantiate_bottleneck(
    config: dict[str, Any],
    runtime_params: dict[str, Any] | None = None,
    use_cache: bool = True,
) -> nn.Module:
    """
    Instantiate a bottleneck component.

    Args:
        config: Bottleneck configuration
        runtime_params: Runtime parameters (e.g., input channels)
        use_cache: Whether to use component caching

    Returns:
        Instantiated bottleneck
    """
    from crackseg.model.factory.registry_setup import bottleneck_registry

    return _instantiate_component(
        config, bottleneck_registry, "bottleneck", runtime_params, use_cache
    )


def instantiate_decoder(
    config: dict[str, Any],
    runtime_params: dict[str, Any] | None = None,
    use_cache: bool = True,
) -> nn.Module:
    """
    Instantiate a decoder component.

    Args:
        config: Decoder configuration
        runtime_params: Runtime parameters (e.g., input channels)
        use_cache: Whether to use component caching

    Returns:
        Instantiated decoder
    """
    from crackseg.model.factory.registry_setup import decoder_registry

    return _instantiate_component(
        config, decoder_registry, "decoder", runtime_params, use_cache
    )


def instantiate_additional_component(
    component_name: str,
    component_config: dict[str, Any],
    use_cache: bool = True,
) -> nn.Module:
    """
    Instantiate an additional component (not encoder/bottleneck/decoder).

    Args:
        component_name: Name of the component
        component_config: Component configuration
        use_cache: Whether to use component caching

    Returns:
        Instantiated component
    """
    from crackseg.model.factory.registry_setup import component_registries

    # Find the appropriate registry for this component
    for _registry_name, registry in component_registries.items():
        if component_name in registry:
            return _instantiate_component(
                component_config, registry, component_name, use_cache=use_cache
            )

    # If not found in any registry, try to instantiate directly
    log.warning(
        f"Component '{component_name}' not found in any registry. "
        "Attempting direct instantiation."
    )

    # This is a fallback - in practice, components should be in registries
    raise InstantiationError(
        f"Component '{component_name}' not found in any registry"
    )
