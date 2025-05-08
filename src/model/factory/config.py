"""
Component instantiation functions for model components.

This module provides functions for instantiating model components from
configuration dictionaries. It handles the creation of encoders, bottlenecks,
decoders, and hybrid models, with validation and runtime parameter support.
"""

import logging
from typing import Dict, Any, Optional, Type, Union

import torch.nn as nn
from omegaconf import DictConfig
import hydra.utils

from src.model.base.abstract import (
    EncoderBase,
    BottleneckBase,
    DecoderBase,
    UNetBase
)
from .factory_utils import (
    hydra_to_dict,
    merge_configs,
    log_component_creation
)
from .registry_setup import (
    encoder_registry,
    bottleneck_registry,
    decoder_registry
)

# Create logger
log = logging.getLogger(__name__)


class InstantiationError(Exception):
    """Exception raised for errors during component instantiation."""
    pass


def validate_component_config(
    config: Dict[str, Any],
    component_type: str
) -> None:
    """
    Validate that a component configuration is valid.

    Args:
        config: Component configuration dictionary
        component_type: Type of component ('encoder', 'bottleneck', 'decoder')

    Raises:
        ValueError: If configuration is invalid
    """
    if component_type == 'encoder':
        if 'in_channels' not in config:
            raise ValueError("Encoder config must specify 'in_channels'")
    elif component_type == 'bottleneck':
        if 'in_channels' not in config:
            raise ValueError("Bottleneck config must specify 'in_channels'")
    elif component_type == 'decoder':
        required = ['in_channels']
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(
                f"Decoder config missing required keys: {', '.join(missing)}"
            )
        # Skip validation can be done here, but we want to be flexible
        # with how skip connections are configured (skip_channels or
        # skip_channels_list)


def validate_architecture_config(config: Dict[str, Any]) -> None:
    """
    Validate that a complete architecture configuration is valid.

    Args:
        config: Complete architecture configuration

    Raises:
        ValueError: If configuration is invalid
    """
    required_components = ['encoder', 'bottleneck', 'decoder']
    missing = [comp for comp in required_components if comp not in config]
    if missing:
        raise ValueError(
            f"Architecture config missing components: {', '.join(missing)}"
        )

    # Validate each component
    validate_component_config(config['encoder'], 'encoder')
    validate_component_config(config['bottleneck'], 'bottleneck')
    validate_component_config(config['decoder'], 'decoder')


def normalize_config(
    config: Union[Dict[str, Any], DictConfig],
    defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Normalize a configuration dictionary for consistency.

    Args:
        config: Configuration dictionary or DictConfig
        defaults: Default values to merge with config

    Returns:
        Normalized dictionary
    """
    # Convert to regular dict if OmegaConf
    if isinstance(config, DictConfig):
        config_dict = hydra_to_dict(config)
    else:
        config_dict = dict(config)

    # Apply defaults if provided
    if defaults:
        for key, value in defaults.items():
            if key not in config_dict:
                config_dict[key] = value

    return config_dict


def parse_architecture_config(
    config: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Parse a complete architecture configuration into component configs.

    Args:
        config: Complete architecture configuration

    Returns:
        Dictionary mapping component names to their configs
    """
    validate_architecture_config(config)
    return {
        'encoder': config['encoder'],
        'bottleneck': config['bottleneck'],
        'decoder': config['decoder']
    }


def instantiate_encoder(
    config: Dict[str, Any],
    runtime_params: Optional[Dict[str, Any]] = None
) -> EncoderBase:
    """
    Instantiate an encoder component from configuration.

    Args:
        config: Encoder configuration dictionary
        runtime_params: Optional runtime parameters to override config

    Returns:
        Instantiated encoder component

    Raises:
        InstantiationError: If instantiation fails
    """
    try:
        # Validate and normalize config
        validate_component_config(config, 'encoder')
        full_config = normalize_config(config)

        # Apply runtime parameters if provided
        if runtime_params:
            full_config = merge_configs(full_config, runtime_params)

        # Try instantiation methods in order
        return _try_instantiation_methods(
            full_config, 'encoder', encoder_registry, EncoderBase
        )

    except Exception as e:
        log.error(f"Error instantiating encoder: {e}", exc_info=True)
        raise InstantiationError(f"Failed to instantiate encoder: {e}") from e


def instantiate_bottleneck(
    config: Dict[str, Any],
    runtime_params: Optional[Dict[str, Any]] = None
) -> BottleneckBase:
    """
    Instantiate a bottleneck component from configuration.

    Args:
        config: Bottleneck configuration dictionary
        runtime_params: Optional runtime parameters to override config

    Returns:
        Instantiated bottleneck component

    Raises:
        InstantiationError: If instantiation fails
    """
    try:
        # Validate and normalize config
        validate_component_config(config, 'bottleneck')
        full_config = normalize_config(config)

        # Apply runtime parameters if provided
        if runtime_params:
            full_config = merge_configs(full_config, runtime_params)

        # Try instantiation methods in order
        return _try_instantiation_methods(
            full_config, 'bottleneck', bottleneck_registry, BottleneckBase
        )

    except Exception as e:
        log.error(f"Error instantiating bottleneck: {e}", exc_info=True)
        raise InstantiationError(
            f"Failed to instantiate bottleneck: {e}"
        ) from e


def instantiate_decoder(
    config: Dict[str, Any],
    runtime_params: Optional[Dict[str, Any]] = None
) -> DecoderBase:
    """
    Instantiate a decoder component from configuration.

    Args:
        config: Decoder configuration dictionary
        runtime_params: Optional runtime parameters to override config

    Returns:
        Instantiated decoder component

    Raises:
        InstantiationError: If instantiation fails
    """
    try:
        # Validate and normalize config
        validate_component_config(config, 'decoder')
        full_config = normalize_config(config)

        # Apply runtime parameters if provided
        if runtime_params:
            full_config = merge_configs(full_config, runtime_params)

        # Try instantiation methods in order
        return _try_instantiation_methods(
            full_config, 'decoder', decoder_registry, DecoderBase
        )

    except Exception as e:
        log.error(f"Error instantiating decoder: {e}", exc_info=True)
        raise InstantiationError(f"Failed to instantiate decoder: {e}") from e


def instantiate_hybrid_model(
    encoder: EncoderBase,
    bottleneck: BottleneckBase,
    decoder: DecoderBase,
    model_type: str = "BaseUNet"
) -> UNetBase:
    """
    Instantiate a hybrid model from pre-created components.

    Args:
        encoder: Encoder component
        bottleneck: Bottleneck component
        decoder: Decoder component
        model_type: Type of model to instantiate

    Returns:
        Instantiated model

    Raises:
        InstantiationError: If model instantiation fails
    """
    try:
        # Attempt to get the model class from the registry
        from .registry_setup import architecture_registry

        if model_type in architecture_registry:
            model_cls = architecture_registry.get(model_type)
            model = model_cls(encoder, bottleneck, decoder)
            log_component_creation("Hybrid Model", model_type)
            return model

        # Fallback to direct import
        import importlib

        if '.' in model_type:
            # Assume it's a fully qualified class name
            module_name, class_name = model_type.rsplit('.', 1)
            module = importlib.import_module(module_name)
            model_cls = getattr(module, class_name)
        else:
            # Try a few common locations
            locations = [
                'src.model.core.unet',
                'src.model.unet',
                'src.model'
            ]

            for location in locations:
                try:
                    module = importlib.import_module(location)
                    if hasattr(module, model_type):
                        model_cls = getattr(module, model_type)
                        break
                except (ImportError, AttributeError):
                    continue
            else:
                raise InstantiationError(
                    f"Could not find model class '{model_type}'"
                )

        model = model_cls(encoder, bottleneck, decoder)
        log_component_creation("Hybrid Model", model_type)
        return model

    except Exception as e:
        log.error(f"Error instantiating hybrid model: {e}", exc_info=True)
        raise InstantiationError(
            f"Failed to instantiate hybrid model: {e}"
        ) from e


def create_model_from_config(config: Dict[str, Any]) -> UNetBase:
    """
    Create a complete model from a comprehensive configuration.

    Args:
        config: Complete model configuration

    Returns:
        Instantiated model

    Raises:
        InstantiationError: If model creation fails
    """
    try:
        # Parse configuration into component configs
        component_configs = parse_architecture_config(config)

        # Instantiate components
        encoder = instantiate_encoder(component_configs['encoder'])

        # Add runtime parameters from encoder to bottleneck
        bottleneck_runtime = {'in_channels': encoder.out_channels}
        bottleneck = instantiate_bottleneck(
            component_configs['bottleneck'],
            runtime_params=bottleneck_runtime
        )

        # Add runtime parameters from encoder and bottleneck to decoder
        decoder_runtime = {
            'in_channels': bottleneck.out_channels,
            'skip_channels_list': list(reversed(encoder.skip_channels))
        }
        decoder = instantiate_decoder(
            component_configs['decoder'],
            runtime_params=decoder_runtime
        )

        # Get model type from config or use default
        model_type = config.get('type', 'BaseUNet')

        # Instantiate the model
        return instantiate_hybrid_model(
            encoder, bottleneck, decoder, model_type
        )

    except Exception as e:
        log.error(f"Error creating model from config: {e}", exc_info=True)
        raise InstantiationError(
            f"Failed to create model from config: {e}"
        ) from e


def _try_instantiation_methods(
    config: Dict[str, Any],
    component_type: str,
    registry: Any,
    base_class: Type
) -> nn.Module:
    """
    Try multiple methods to instantiate a component.

    Args:
        config: Component configuration
        component_type: Type of component
        registry: Registry for the component type
        base_class: Base class for the component type

    Returns:
        Instantiated component

    Raises:
        InstantiationError: If all instantiation methods fail
    """
    # Method 1: Use _target_ with Hydra (preferred)
    if '_target_' in config:
        try:
            component = hydra.utils.instantiate(config)
            if not isinstance(component, base_class):
                raise TypeError(
                    f"Instantiated component is not a {base_class.__name__}"
                )
            log_component_creation(
                component_type.capitalize(),
                type(component).__name__
            )
            return component
        except Exception as e:
            log.warning(
                f"Hydra instantiation failed for {component_type}: {e}"
            )
            # Fall through to other methods

    # Method 2: Use 'type' with registry
    if 'type' in config:
        try:
            component_name = config['type']
            params = {k: v for k, v in config.items()
                      if k not in ['type', '_target_']}

            if component_name in registry:
                component = registry.instantiate(component_name, **params)
                log_component_creation(
                    component_type.capitalize(),
                    component_name
                )
                return component
        except Exception as e:
            log.warning(
                f"Registry instantiation failed for {component_type}: {e}"
            )
            # Fall through to other methods

    # Method 3: Direct import (last resort)
    if '_target_' in config:
        try:
            import importlib

            target = config['_target_']
            module_name, class_name = target.rsplit('.', 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)

            params = {k: v for k, v in config.items()
                      if k not in ['_target_', 'type']}
            component = cls(**params)

            if not isinstance(component, base_class):
                raise TypeError(
                    f"Instantiated component is not a {base_class.__name__}"
                )

            log_component_creation(
                component_type.capitalize(),
                class_name
            )
            return component
        except Exception as e:
            log.warning(
                f"Direct import instantiation failed for {component_type}: {e}"
            )

    # If we get here, all methods failed
    raise InstantiationError(
        f"Failed to instantiate {component_type} component with config: "
        f"{config}"
    )
