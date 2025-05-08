"""
Component Instantiation System.

This module provides specialized functions for instantiating model components
from validated configurations. It handles caching for component reuse,
specialized instantiation logic for different component types, and robust
error handling.
"""

import logging
from typing import Dict, Any, TypeVar, Optional

from torch import nn

from src.model.factory.registry import Registry
from src.model.factory.registry_setup import (
    encoder_registry,
    bottleneck_registry,
    decoder_registry,
    architecture_registry,
    component_registries
)
# Import cache utilities
from ...utils.component_cache import (
    get_cached_component,
    cache_component,
    generate_cache_key,
    # clear_component_cache # Keep clear_component_cache export if needed
    # externally - Removing unused import
)


# Create logger
log = logging.getLogger(__name__)

# Type variables
T = TypeVar('T', bound=nn.Module)
Component = TypeVar('Component', bound=nn.Module)

# Component cache system (using weak references to avoid memory leaks)
# REMOVED: _component_cache variable


class InstantiationError(Exception):
    """Exception raised when component instantiation fails."""
    pass


# REMOVED: clear_component_cache function

# REMOVED: get_cached_component function

# REMOVED: cache_component function

# REMOVED: generate_cache_key function


def _instantiate_component(
    config: Dict[str, Any],
    registry: Registry,
    component_category: str,  # e.g., 'encoder', 'bottleneck' for log/errors
    runtime_params: Optional[Dict[str, Any]] = None,
    use_cache: bool = True
) -> nn.Module:
    """
    Internal helper to instantiate a component from a registry.

    Handles config validation, caching, registry lookup, instantiation,
    and error handling.

    Args:
        config: Component configuration dictionary (must include 'type')
        registry: The specific component registry to use
        component_category: Name of the component category (for messages)
        runtime_params: Optional runtime parameters to override configuration
        use_cache: Whether to use/update the component cache

    Returns:
        Instantiated component

    Raises:
        InstantiationError: If instantiation fails
    """
    component_type = config.get("type")
    target_path = config.get("_target_")

    if not component_type and target_path:
        # Infer type from _target_ if type is missing
        try:
            component_type = target_path.split('.')[-1]
            log.debug(f"Inferred component type '{component_type}' from "
                      f"_target_ '{target_path}'")
        except IndexError:
            raise InstantiationError(
                f"Could not infer component type from _target_: {target_path}"
            )

    if not component_type:
        raise InstantiationError(
            f"{component_category.capitalize()} config must specify 'type' "
            f"or '_target_'"
        )

    # Create a copy to avoid modifying the original config
    config_copy = config.copy()
    # Remove keys used for lookup/instantiation logic, keep others for params
    config_copy.pop("type", None)       # Remove type if it existed
    config_copy.pop("_target_", None)  # Remove _target_ if it existed

    # Specific cleanup for decoders: remove out_channels if present
    if component_category == "decoder":
        removed_out = config_copy.pop("out_channels", None)
        if removed_out is not None:
            log.debug("Removed 'out_channels' from decoder config before "
                      "instantiation.")

        # Remove CBAM parameters from decoder config
        if "cbam_enabled" in config_copy:
            config_copy.pop("cbam_enabled")
            config_copy.pop("cbam_params", {})
            log.debug("Removed CBAM parameters from decoder config "
                      "before instantiation.")

    # Merge runtime parameters if provided
    if runtime_params:
        config_copy.update(runtime_params)
        log.debug(f"Merged runtime params for {component_type}: "
                  f"{runtime_params}")

    # --- Cache Check ---
    cache_key = None
    if use_cache:
        cache_key = generate_cache_key(component_type, config_copy)
        cached = get_cached_component(cache_key)
        if cached is not None:
            return cached
    # --- End Cache Check ---

    try:
        # Check if the component type exists in the registry
        if component_type not in registry:
            available = ", ".join(registry.list())
            raise InstantiationError(
                f"Unknown {component_category} type '{component_type}'. "
                f"Available types: {available}"
            )

        # Get the component class from the registry
        component_class = registry.get(component_type)

        # Instantiate the component
        component = component_class(**config_copy)
        log.info(
            f"Successfully instantiated {component_category}: {component_type}"
        )

        # --- Cache Update ---
        if use_cache and cache_key:
            cache_component(cache_key, component)
        # --- End Cache Update ---

        return component

    except Exception as e:
        # Preserve InstantiationError, wrap others
        if isinstance(e, InstantiationError):
            raise
        else:
            error_msg = (
                f"Failed to instantiate {component_category} "
                f"'{component_type}': {e}"
            )
            log.error(error_msg, exc_info=True)
            raise InstantiationError(error_msg) from e


# --- Public Instantiation Functions ---

def instantiate_encoder(
    config: Dict[str, Any], use_cache: bool = True
) -> nn.Module:
    """
    Instantiate an encoder component from configuration.

    Args:
        config: Encoder configuration dictionary (must include 'type')
        use_cache: Whether to use/update the component cache

    Returns:
        Instantiated encoder component

    Raises:
        InstantiationError: If instantiation fails
    """
    return _instantiate_component(
        config=config,
        registry=encoder_registry,
        component_category="encoder",
        use_cache=use_cache
    )


def instantiate_bottleneck(
    config: Dict[str, Any],
    runtime_params: Optional[Dict[str, Any]] = None,
    use_cache: bool = True
) -> nn.Module:
    """
    Instantiate a bottleneck component from configuration.

    Args:
        config: Bottleneck configuration dictionary (must include 'type')
        runtime_params: Optional runtime parameters to override configuration
        use_cache: Whether to use/update the component cache

    Returns:
        Instantiated bottleneck component

    Raises:
        InstantiationError: If instantiation fails
    """
    return _instantiate_component(
        config=config,
        registry=bottleneck_registry,
        component_category="bottleneck",
        runtime_params=runtime_params,
        use_cache=use_cache
    )


def instantiate_decoder(
    config: Dict[str, Any],
    runtime_params: Optional[Dict[str, Any]] = None,
    use_cache: bool = True
) -> nn.Module:
    """
    Instantiate a decoder component from configuration.

    Args:
        config: Decoder configuration dictionary (must include 'type')
        runtime_params: Optional runtime parameters to override configuration
        use_cache: Whether to use/update the component cache

    Returns:
        Instantiated decoder component

    Raises:
        InstantiationError: If instantiation fails
    """
    return _instantiate_component(
        config=config,
        registry=decoder_registry,
        component_category="decoder",
        runtime_params=runtime_params,
        use_cache=use_cache
    )


def instantiate_additional_component(
    component_name: str,
    component_config: Dict[str, Any],
    use_cache: bool = True
) -> nn.Module:
    """
    Instantiate an additional component (attention, etc.) from configuration.

    Args:
        component_name: Name to identify the component category
            (e.g., "attention")
        component_config: Component configuration dictionary
            (must include 'type')
        use_cache: Whether to use/update the component cache

    Returns:
        Instantiated component

    Raises:
        InstantiationError: If instantiation fails
    """
    if "type" not in component_config:
        raise InstantiationError(
            f"Component '{component_name}' config must specify 'type'"
        )

    component_type = component_config["type"]  # For logging/error later

    # Determine registry based on component_name primarily
    registry = component_registries.get(component_name.lower())

    if registry is None:
        # Fallback: Try common suffixes (could be expanded)
        if component_type.endswith("Encoder"):
            registry = encoder_registry
        elif component_type.endswith("Bottleneck") or \
                component_type.endswith("Module"):
            registry = bottleneck_registry
        elif component_type.endswith("Decoder"):
            registry = decoder_registry
        else:
            raise InstantiationError(
                f"Cannot determine registry for component '{component_name}' "
                f"of type '{component_type}'"
            )

    return _instantiate_component(
        config=component_config,
        registry=registry,
        component_category=component_name,  # Use name as category
        use_cache=use_cache
    )


# --- Helper Functions for Hybrid Instantiation ---

def _instantiate_main_components(
    config: Dict[str, Any], use_cache: bool
) -> Dict[str, nn.Module]:
    """
    Instantiate the core encoder, bottleneck, and decoder components.

    Handles passing runtime parameters (like inferred channel sizes)
    between components.
    """
    components = {}
    # Instantiate main components sequentially, calculating runtime params
    if "encoder" in config:
        components["encoder"] = instantiate_encoder(
            config["encoder"], use_cache
        )

    if "bottleneck" in config:
        runtime_params_bt = {}
        encoder_instance = components.get("encoder")
        if encoder_instance and hasattr(encoder_instance, 'out_channels'):
            enc_out_ch = getattr(encoder_instance, 'out_channels', None)
            if enc_out_ch is not None:
                runtime_params_bt["in_channels"] = enc_out_ch
                log.debug(f"Passing encoder out_channels ({enc_out_ch}) to "
                          "bottleneck.")
            else:
                log.warning("Encoder has 'out_channels' but value is None.")
        elif "in_channels" not in config["bottleneck"]:
            log.warning("Cannot determine bottleneck in_channels from encoder."
                        )

        components["bottleneck"] = instantiate_bottleneck(
            config["bottleneck"], runtime_params_bt, use_cache
        )

    if "decoder" in config:
        runtime_params_dec = {}
        bottleneck_instance = components.get("bottleneck")
        if bottleneck_instance and hasattr(bottleneck_instance, 'out_channels'
                                           ):
            bt_out_ch = getattr(bottleneck_instance, 'out_channels', None)
            if bt_out_ch is not None:
                runtime_params_dec["in_channels"] = bt_out_ch
                log.debug(f"Passing bottleneck out_channels ({bt_out_ch}) to "
                          "decoder.")
            else:
                log.warning("Bottleneck has 'out_channels' but value is None.")
        elif "in_channels" not in config["decoder"]:
            log.warning("Cannot determine decoder in_channels from bottleneck."
                        )

        encoder_instance = components.get("encoder")
        if encoder_instance and hasattr(encoder_instance, 'skip_channels'):
            encoder_skips = getattr(encoder_instance, 'skip_channels', None)
            if isinstance(encoder_skips, list):
                runtime_params_dec["skip_channels_list"] = list(
                    reversed(encoder_skips)
                )
                log.debug("Passing reversed encoder skip_channels_list to "
                          "decoder.")
            elif encoder_skips is not None:
                log.warning("Encoder 'skip_channels' found but is not a list.")

        components["decoder"] = instantiate_decoder(
            config["decoder"], runtime_params_dec, use_cache
        )
    return components


def _instantiate_additional_components(
    config: Dict[str, Any], use_cache: bool
) -> Dict[str, nn.Module]:
    """Instantiate additional components defined under the 'components' key."""
    additional_components = {}
    if "components" in config:
        for name, comp_config in config["components"].items():
            additional_components[name] = instantiate_additional_component(
                name, comp_config, use_cache
            )
    return additional_components


def _apply_final_activation(
    model: nn.Module, config: Dict[str, Any]
) -> nn.Module:
    """Apply a final activation layer to the model if configured."""
    if "final_activation" in config:
        activation_config = config["final_activation"]
        activation_instance = None

        try:
            if isinstance(activation_config, dict) and "type" in \
                    activation_config:
                act_type = activation_config["type"]
                act_params = {k: v for k, v in activation_config.items()
                              if k != "type"}
                # Try importing from torch.nn first
                if hasattr(nn, act_type):
                    activation_class = getattr(nn, act_type)
                else:  # Try dynamic import
                    module_name, class_name = act_type.rsplit(".", 1)
                    module = __import__(module_name, fromlist=[class_name])
                    activation_class = getattr(module, class_name)
                activation_instance = activation_class(**act_params)
                log.info(f"Added final activation: {act_type}")

            elif isinstance(activation_config, str):  # Simple case: just name
                activation_class = getattr(nn, activation_config)
                activation_instance = activation_class()
                log.info(f"Added final activation: {activation_config}")

            if activation_instance:
                return nn.Sequential(model, activation_instance)
            else:
                log.warning("Ignoring invalid 'final_activation' config: "
                            f"{activation_config}")
                return model  # Return original model if activation failed

        except (AttributeError, ImportError, TypeError, ValueError) as act_e:
            raise InstantiationError(
                "Failed to instantiate final_activation "
                f"'{activation_config}': {act_e}"
            ) from act_e
    else:
        return model  # No final activation specified

# --- End Helper Functions ---


def instantiate_hybrid_model(
    config: Dict[str, Any],
    use_cache: bool = True
) -> nn.Module:
    """Instantiate a hybrid model using helper functions."""
    if "type" not in config:
        raise InstantiationError(
            "Architecture configuration must specify 'type'"
        )

    arch_type = config["type"]
    if arch_type not in architecture_registry:
        available = ", ".join(architecture_registry.list())
        raise InstantiationError(
            f"Unknown architecture type '{arch_type}'. "
            f"Available types: {available}"
        )

    architecture_class = architecture_registry.get(arch_type)

    # Exclude component configs and activation from main params
    excluded_keys = ["type", "encoder", "bottleneck", "decoder",
                     "components", "final_activation"]
    params = {k: v for k, v in config.items() if k not in excluded_keys}

    try:
        # 1. Instantiate main components
        main_components = _instantiate_main_components(config, use_cache)

        # 2. Instantiate additional components
        additional_components = _instantiate_additional_components(config,
                                                                   use_cache)

        # 3. Combine components
        all_components = {**main_components, **additional_components}

        # 4. Instantiate the main architecture
        log.debug(
            f"Instantiating architecture '{arch_type}' with params: "
            f"{list(params.keys())} and components: "
            f"{list(all_components.keys())}"
        )
        model_base = architecture_class(**params, **all_components)

        # 5. Apply final activation
        model_final = _apply_final_activation(model_base, config)

        log.info(f"Successfully instantiated hybrid model: {arch_type}")
        return model_final

    except Exception as e:
        if isinstance(e, InstantiationError):
            raise
        else:
            error_msg = (
                f"Failed to instantiate hybrid model '{arch_type}': {e}"
            )
            log.error(error_msg, exc_info=True)
            raise InstantiationError(error_msg) from e


# Registry-based generic instantiation (kept for potential direct use)
