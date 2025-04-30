"""
Factory functions for instantiating model components from configuration.

Provides functions to create encoders, bottlenecks, decoders, and complete
UNet models based on configuration dictionaries. Uses the registry system
for component lookup.
"""

from typing import Dict, Any, List, TypeVar
import logging
import hydra.utils

from src.model.base import EncoderBase, BottleneckBase, DecoderBase, UNetBase
from src.model.registry import Registry


# Create logger
logger = logging.getLogger(__name__)

# Component type
T = TypeVar('T')

# Create registries for each component type
encoder_registry = Registry(EncoderBase, "Encoder")
bottleneck_registry = Registry(BottleneckBase, "Bottleneck")
decoder_registry = Registry(DecoderBase, "Decoder")
# unet_registry = Registry(UNetBase, "UNet") # Removed, not needed for
# create_unet


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


def create_encoder(config: Dict[str, Any]) -> EncoderBase:
    """
    Create an encoder component from configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary with:
            - type: registered encoder name
            - in_channels: number of input channels
            - Additional encoder-specific parameters

    Returns:
        EncoderBase: Instantiated encoder component.

    Raises:
        ConfigurationError: If required configuration is missing or invalid.
        KeyError: If the specified encoder type is not registered.
    """
    # Validate basic configuration
    validate_config(config, ["type", "in_channels"], "encoder")

    encoder_type = config["type"]

    try:
        # Get additional params by excluding known keys
        params = {k: v for k, v in config.items()
                  if k not in ["type", "_target_"]}

        # Create encoder instance
        return encoder_registry.instantiate(encoder_type, **params)
    except KeyError:
        available = encoder_registry.list()
        raise ConfigurationError(
            f"Encoder type '{encoder_type}' not found in registry. "
            f"Available types: {', '.join(available)}"
        )
    except Exception as e:
        # Catch any other exceptions during instantiation
        raise ConfigurationError(
            f"Error creating encoder of type '{encoder_type}': {str(e)}"
        ) from e


def create_bottleneck(config: Dict[str, Any]) -> BottleneckBase:
    """
    Create a bottleneck component from configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary with:
            - type: registered bottleneck name
            - in_channels: number of input channels
            - Additional bottleneck-specific parameters

    Returns:
        BottleneckBase: Instantiated bottleneck component.

    Raises:
        ConfigurationError: If required configuration is missing or invalid.
        KeyError: If the specified bottleneck type is not registered.
    """
    # Validate basic configuration
    validate_config(config, ["type", "in_channels"], "bottleneck")

    bottleneck_type = config["type"]

    try:
        # Get additional params by excluding known keys
        params = {k: v for k, v in config.items()
                  if k not in ["type", "_target_"]}

        # Create bottleneck instance
        return bottleneck_registry.instantiate(bottleneck_type, **params)
    except KeyError:
        available = bottleneck_registry.list()
        raise ConfigurationError(
            f"Bottleneck type '{bottleneck_type}' not found in registry. "
            f"Available types: {', '.join(available)}"
        )
    except Exception as e:
        # Catch any other exceptions during instantiation
        raise ConfigurationError(
            f"Error creating bottleneck of type '{bottleneck_type}': {str(e)}"
        ) from e


def create_decoder(config: Dict[str, Any]) -> DecoderBase:
    """
    Create a decoder component from configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary with:
            - type: registered decoder name
            - in_channels: number of input channels
            - skip_channels OR skip_channels_list: skip info
            - Additional decoder-specific parameters

    Returns:
        DecoderBase: Instantiated decoder component.

    Raises:
        ConfigurationError: If required configuration is missing or invalid.
        KeyError: If the specified decoder type is not registered.
    """
    # Validate basic configuration
    if "skip_channels" not in config and "skip_channels_list" not in config:
        raise ConfigurationError(
            "Missing required config for decoder: 'skip_channels' or "
            "'skip_channels_list'"
        )
    validate_config(config, ["type", "in_channels"], "decoder")

    decoder_type = config["type"]

    try:
        # Prepare params: pass everything except 'type' and '_target_'
        params = {k: v for k, v in config.items()
                  if k not in ["type", "_target_"]}

        # Special handling for mocks that might not expect all args
        if decoder_type == "TestDecoder" and "out_channels" in params:
            # MockDecoder doesn't expect out_channels in init
            del params["out_channels"]

        # Let the specific decoder handle its expected arguments
        return decoder_registry.instantiate(decoder_type, **params)
    except KeyError:
        available = decoder_registry.list()
        raise ConfigurationError(
            f"Decoder type '{decoder_type}' not found in registry. "
            f"Available types: {', '.join(available)}"
        )
    except Exception as e:
        raise ConfigurationError(
            f"Error creating decoder of type '{decoder_type}': {str(e)}"
        ) from e


def create_unet(config: Dict[str, Any]) -> UNetBase:
    """
    Create a complete UNet model from configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary with:
            - _target_: Path to the UNet class (e.g., src.model.unet.BaseUNet)
            - encoder: encoder configuration
            - bottleneck: bottleneck configuration
            - decoder: decoder configuration
            - Additional UNet-specific parameters (like final_activation)

    Returns:
        UNetBase: Instantiated UNet model.

    Raises:
        ConfigurationError: If required configuration is missing or invalid.
        ImportError: If the target class cannot be imported.
    """
    # Validate basic configuration
    validate_config(
        config, ["_target_", "encoder", "bottleneck", "decoder"], "unet"
    )

    unet_cls_path = config["_target_"]

    try:
        # Dynamically import the UNet class
        unet_cls = hydra.utils.get_class(unet_cls_path)
        if not issubclass(unet_cls, UNetBase):
            raise TypeError(f"Target class {unet_cls_path} does not inherit \
from UNetBase")

        # Create components first
        encoder = create_encoder(config["encoder"])
        bottleneck = create_bottleneck(config["bottleneck"])
        decoder = create_decoder(config["decoder"])

        # Get additional UNet-specific parameters
        unet_params = {k: v for k, v in config.items()
                       # Also exclude 'type' just in case
                       if k not in ["_target_", "type",
                                    "encoder", "bottleneck", "decoder"]}

        # Instantiate the specific UNet class
        return unet_cls(
            encoder=encoder,
            bottleneck=bottleneck,
            decoder=decoder,
            **unet_params
        )
    except ImportError as e:
        raise ConfigurationError(
            f"Could not import UNet class '{unet_cls_path}': {str(e)}"
        ) from e
    except TypeError as e:
        # Catch potential TypeError from issubclass or __init__
        raise ConfigurationError(
            f"Error related to UNet class '{unet_cls_path}': {str(e)}"
        ) from e
    except ConfigurationError:
        # Re-raise component creation errors
        raise
    except Exception as e:
        # Catch any other exceptions during instantiation
        raise ConfigurationError(
            f"Error creating UNet from '{unet_cls_path}': {str(e)}"
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
            f"Error creating {registry.name.lower()} of type \
'{component_type}': {str(e)}"
        ) from e
