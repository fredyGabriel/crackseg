"""
Factory functions for instantiating model components from configuration.

Provides functions to create encoders, bottlenecks, decoders, and complete
UNet models based on configuration dictionaries. Uses the registry system
for component lookup.
"""

import logging
from typing import Dict, Any, List, TypeVar

# Imports are kept if validate_config or create_component_from_config are used
from omegaconf import DictConfig, OmegaConf
import hydra.utils
import torch.nn as nn
from src.model.base import EncoderBase, BottleneckBase, DecoderBase, UNetBase
from src.model.registry import Registry  # Keep if generic creator is used
from src.model.components.cbam import attention_registry

# Create logger
log = logging.getLogger(__name__)

# Component type
T = TypeVar('T')

# Create registries for each component type
# encoder_registry = Registry(EncoderBase, "Encoder") # Keep registry
# definitions if used elsewhere
bottleneck_registry = Registry(BottleneckBase, "Bottleneck")
# decoder_registry = Registry(DecoderBase, "Decoder")


class ConfigurationError(Exception):
    """Exception raised for errors in the configuration."""
    pass


def validate_config(config: Dict[str, Any], required_keys: List[str],
                    component_type: str) -> None:
    """
    Validate that a configuration dictionary contains all required keys.

    Args:
        config (Dict[str, Any]): Configuration dictionary to validate.
        required_keys (List[str]): List of keys that must be present.
        component_type (str): Type of component for error messages.

    Raises:
        ConfigurationError: If any required key is missing.
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ConfigurationError(
            f"Missing required configuration for {component_type}: "
            f"{', '.join(missing_keys)}"
        )


# ======================================================================
# Restoring factory functions
# ======================================================================

def create_encoder(config: DictConfig) -> EncoderBase:
    """Factory function to create an encoder based on configuration."""
    # Directly instantiate using the provided config (which includes _target_)
    try:
        log.info(f"Attempting to instantiate encoder with config: {config}")
        encoder = hydra.utils.instantiate(config)
        if not isinstance(encoder, EncoderBase):
            raise TypeError(
                f"Instantiated object for config {config} is not an \
EncoderBase"
            )
        log.info(
            f"Successfully instantiated encoder: "
            f"{type(encoder).__name__}"
        )
        return encoder
    except Exception as e:
        log.error(
            f"Failed to create encoder from config {config}: {e}",
            exc_info=True
        )
        raise ConfigurationError(f"Failed to create encoder: {e}") from e


def create_bottleneck(config: DictConfig, **runtime_kwargs) -> BottleneckBase:
    """Factory function to create a bottleneck based on configuration."""
    # Directly instantiate using the provided config, merging runtime kwargs
    try:
        log.info(
            f"Attempting to instantiate bottleneck with config: {config} "
            f"and runtime_kwargs: {runtime_kwargs}"
        )
        # Merge runtime kwargs into the config node before instantiation
        merged_config = OmegaConf.merge(config, runtime_kwargs)
        log.info(f"Instantiating bottleneck with merged config: \
{merged_config}")
        bottleneck = hydra.utils.instantiate(merged_config)
        if not isinstance(bottleneck, BottleneckBase):
            raise TypeError(
                f"Instantiated object for config {config} with runtime args "
                f"{runtime_kwargs} is not a BottleneckBase"
            )
        log.info(
            f"Successfully instantiated bottleneck: "
            f"{type(bottleneck).__name__}"
        )
        return bottleneck
    except Exception as e:
        log.error(
            f"Failed to create bottleneck from config {config} and runtime "
            f"args {runtime_kwargs}: {e}",
            exc_info=True
        )
        raise ConfigurationError(f"Failed to create bottleneck: {e}") from e


def create_decoder(config: DictConfig, **runtime_kwargs) -> DecoderBase:
    """Factory function to create a decoder based on configuration."""
    # Directly instantiate using the provided config, merging runtime kwargs
    try:
        log.info(
            f"Attempting to instantiate decoder with config: {config} "
            f"and runtime_kwargs: {runtime_kwargs}"
        )
        merged_config = OmegaConf.merge(config, runtime_kwargs)
        log.info(f"Instantiating decoder with merged config: {merged_config}")
        decoder = hydra.utils.instantiate(merged_config)
        if not isinstance(decoder, DecoderBase):
            raise TypeError(
                f"Instantiated object for config {config} with runtime args "
                f"{runtime_kwargs} is not a DecoderBase"
            )
        log.info(
            f"Successfully instantiated decoder: "
            f"{type(decoder).__name__}"
        )
        return decoder
    except Exception as e:
        log.error(
            f"Failed to create decoder from config {config} and runtime "
            f"args {runtime_kwargs}: {e}",
            exc_info=True
        )
        raise ConfigurationError(f"Failed to create decoder: {e}") from e


def create_unet(config: DictConfig) -> UNetBase:
    """Factory function to create a U-Net model from configuration."""
    required_components = ["encoder", "bottleneck", "decoder"]
    validate_config(config, required_components, "unet")

    try:
        # 1. Instantiate Encoder using its factory
        encoder = create_encoder(config.encoder)
        log.info(f"Instantiated Encoder: {type(encoder).__name__}")

        # 2. Instantiate Bottleneck using its factory, passing derived
        # in_channels
        bottleneck_in_channels = encoder.out_channels
        bottleneck = create_bottleneck(config.bottleneck,
                                       in_channels=bottleneck_in_channels)
        log.info(f"Instantiated Bottleneck: {type(bottleneck).__name__}")

        # 3. Instantiate Decoder using its factory, passing derived
        # channels/skips
        decoder_in_channels = bottleneck.out_channels
        encoder_skips = getattr(encoder, 'skip_channels', [])
        if not isinstance(encoder_skips, list):
            encoder_skips = []

        # IMPORTANT: Reverse encoder skip_channels when passing to decoder
        # This is because encoder outputs skip connections from high to low
        # resolution, but decoder processes them from low to high resolution
        # during upsampling.
        # Reverse encoder skips for the decoder
        reversed_encoder_skips = list(reversed(encoder_skips))
        # Pass reversed skips to decoder initialization
        decoder = create_decoder(
            config.decoder,
            in_channels=decoder_in_channels,
            skip_channels_list=reversed_encoder_skips
        )
        log.info(f"Instantiated Decoder: {type(decoder).__name__}")

        # 4. Instanciar la clase concreta indicada por _target_ o BaseUNet
        unet_target = config.get('_target_', 'src.model.unet.BaseUNet')
        if isinstance(unet_target, str):
            from hydra.utils import get_class
            UnetClass = get_class(unet_target)
        else:
            UnetClass = unet_target
        unet_model = UnetClass(
            encoder=encoder, bottleneck=bottleneck, decoder=decoder
        )
        log.info(f"Instantiated {UnetClass.__name__}.")

        # 5. Handle optional final activation
        if "final_activation" in config:
            activation = hydra.utils.instantiate(config.final_activation)
            unet_model = nn.Sequential(unet_model, activation)
            log.info(
                f"Added final activation: "
                f"{type(activation).__name__}"
            )

        return unet_model

    except Exception as e:
        log.error(f"Error instantiating UNet model from config: {config}",
                  exc_info=True)
        raise ConfigurationError(
            f"Error instantiating UNet model: {str(e)}"
        ) from e


def create_component_from_config(
    config: Dict[str, Any], registry: Registry[T]
) -> T:
    """
    Generic function to create a component from configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        registry (Registry[T]): Component registry to use.

    Returns:
        T: Instantiated component.

    Raises:
        ConfigurationError: If required configuration is missing or invalid.
    """
    # Validate basic configuration
    validate_config(config, ["type"], f"{registry.name.lower()}")

    component_type = config["type"]

    try:
        # Get additional params by excluding known keys
        params = {k: v for k, v in config.items() if k != "type"}

        # Create component instance
        return registry.instantiate(component_type, **params)
    except KeyError:
        available = registry.list()
        raise ConfigurationError(
            f"{registry.name} type '{component_type}' not found in registry. "
            f"Available types: {', '.join(available)}"
        )
    except Exception as e:
        # Catch any other exceptions during instantiation
        raise ConfigurationError(
            f"Error creating {registry.name.lower()} of type "
            f"'{component_type}': {str(e)}"
        ) from e


def insert_cbam_if_enabled(module: nn.Module, config: dict) -> nn.Module:
    """
    Optionally insert a CBAM block into a CNN module if enabled in config.

    Args:
        module (nn.Module): The CNN block to wrap.
        config (dict): Configuration dictionary. Should contain:
            - 'cbam_enabled' (bool): Whether to insert CBAM.
            - 'cbam_params' (dict): Parameters for CBAM (optional).

    Returns:
        nn.Module: The original module or a sequential module with CBAM.

    Example:
        >>> block = nn.Conv2d(64, 64, 3, padding=1)
        >>> config = {'cbam_enabled': True, 'cbam_params': {'reduction': 8}}
        >>> block_with_cbam = insert_cbam_if_enabled(block, config)
    """
    if config.get('cbam_enabled', False):
        cbam_params = config.get('cbam_params', {})
        # Infer in_channels from the module if not provided
        in_channels = getattr(module, 'out_channels', None)
        if in_channels is None:
            raise ValueError(
                "Cannot infer in_channels for CBAM from module. "
                "Please specify 'in_channels' in cbam_params."
            )
        cbam = attention_registry.instantiate(
            'CBAM', in_channels=in_channels, **cbam_params
        )
        return nn.Sequential(module, cbam)
    return module
