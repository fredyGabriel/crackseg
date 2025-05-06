"""
Factory functions for instantiating model components from configuration.

Provides functions to create encoders, bottlenecks, decoders, and complete
UNet models based on configuration dictionaries. Uses the registry system
for component lookup.
"""

import logging
from typing import Dict, Any, TypeVar, Tuple, Optional, Type

from omegaconf import DictConfig
import torch.nn as nn
from src.model.base import UNetBase, DecoderBase
# Import instantiation functions
from .config import (
    instantiate_encoder,
    instantiate_bottleneck,
    instantiate_decoder
    # We might need instantiate_hybrid_model if create_unet handles more
)
# Import global component registries
from .registry_setup import component_registries
# Import factory utilities
from .factory_utils import (
    ConfigurationError, validate_config, hydra_to_dict,
    log_component_creation, log_configuration_error,
    extract_runtime_params
)

# Create logger
log = logging.getLogger(__name__)

# Type variable for better type hinting
ComponentType = TypeVar('ComponentType', bound=nn.Module)
ConfigDict = Dict[str, Any]

# REMOVED Registry Definitions: Managed in registry_setup.py
# encoder_registry = Registry(EncoderBase, "Encoder")
# bottleneck_registry = Registry(BottleneckBase, "Bottleneck")
# decoder_registry = Registry(DecoderBase, "Decoder")


# ======================================================================
# Post-processing class for CBAM integration
# ======================================================================

class CBAMPostProcessor(nn.Module):
    """
    Applies CBAM as a post-processor at the end of the model.

    This approach avoids interfering with the internal logic of components
    and prevents channel compatibility issues that occur when trying to
    integrate CBAM at specific points in the data flow.

    Args:
        original_model: The complete U-Net model
        cbam: The CBAM attention module to apply
    """
    def __init__(self, original_model, cbam):
        super().__init__()
        self.model = original_model
        self.cbam = cbam

    def forward(self, x):
        # Standard model processing
        output = self.model(x)
        # Post-processing with CBAM
        return self.cbam(output)


# ======================================================================
# Factory functions
# ======================================================================

def create_unet(config: DictConfig) -> UNetBase:
    """
    Create a UNet model from configuration.

    Args:
        config: Configuration dictionary containing encoder, bottleneck,
               decoder, and optional parameters

    Returns:
        A UNet model instance

    Raises:
        ConfigurationError: If configuration is invalid or model creation fails
    """
    required_components = ["encoder", "bottleneck", "decoder"]
    validate_config(config, required_components, "unet")

    try:
        # Instantiate components
        encoder, bottleneck, decoder = instantiate_unet_components(config)

        # Get UNet class and create model
        UnetClass = get_unet_class(config)
        unet_model = UnetClass(
            encoder=encoder, bottleneck=bottleneck, decoder=decoder
        )
        log_component_creation("UNet", UnetClass.__name__)

        # Handle optional final activation
        if "final_activation" in config:
            unet_model = add_final_activation(
                unet_model, config.final_activation
            )

        # Apply CBAM if enabled
        global_cbam_enabled = config.get('cbam_enabled', False)
        if global_cbam_enabled:
            global_cbam_params = config.get('cbam_params', {})
            output_channels = getattr(decoder, 'out_channels', None)
            unet_model = apply_cbam_to_model(
                unet_model,
                global_cbam_enabled,
                global_cbam_params,
                output_channels
            )

        return unet_model

    except Exception as e:
        # Catch all errors, log them, and re-raise as ConfigurationError
        log_configuration_error(
            "UNet Creation", str(e),
            hydra_to_dict(config) if isinstance(config, DictConfig) else config
        )
        # Ensure ConfigurationError is raised for consistency
        if isinstance(e, ConfigurationError):
            raise
        else:
            raise ConfigurationError(
                f"Error instantiating UNet model: {str(e)}"
            ) from e


# Keep create_component_from_config for now, uses registry directly
# Needs review later if it should use instantiation module too.
# COMMENTING OUT FOR NOW TO RESOLVE IMPORT/LINTING ERRORS
# def create_component_from_config(
#     config: Dict[str, Any], registry: 'Registry[T]' # Use forward reference
# ) -> T:
#     """
#     Generic function to create a component from configuration.
#
#     Args:
#         config (Dict[str, Any]): Configuration dictionary.
#         registry (Registry[T]): Component registry to use.
#
#     Returns:
#         T: Instantiated component.
#
#     Raises:
#         ConfigurationError: If required configuration is missing or invalid.
#     """
#     # Validate basic configuration
#     validate_config(config, ["type"], f"{registry.name.lower()}")
#
#     component_type = config["type"]
#
#     try:
#         # Get additional params by excluding known keys
#         params = {k: v for k, v in config.items() if k != "type"}
#
#         # Create component instance
#         return registry.instantiate(component_type, **params)
#     except KeyError:
#         available = registry.list()
#         raise ConfigurationError(
#             f"{registry.name} type '{component_type}' not found in registry."
#             f" Available types: {', '.join(available)}"
#         )


# ======================================================================
# Utility functions (example: inserting CBAM)
# ======================================================================

class FinalCBAMDecoder(DecoderBase):
    """
    Decorator that applies CBAM at the end of the decoder processing.

    This class inherits from DecoderBase to ensure full compatibility
    with the interface and validation of UNetBase.

    Args:
        decoder: Original decoder component
        cbam: CBAM attention module
    """
    def __init__(self, decoder, cbam):
        # Initialize DecoderBase with the same parameters as the original
        # decoder
        super().__init__(
            in_channels=decoder.in_channels,
            skip_channels=decoder.skip_channels
        )
        # Register submodules
        self.add_module('decoder', decoder)
        self.add_module('cbam', cbam)

        # Copy out_channels attribute
        self._out_channels = decoder.out_channels

    def forward(self, x, skip_connections=None):
        # First apply the decoder with skip_connections
        decoded = self.decoder(x, skip_connections)
        # Then apply CBAM at the end
        return self.cbam(decoded)

    @property
    def out_channels(self):
        """
        Returns the number of output channels of the decorated decoder.
        Implementation required by DecoderBase.
        """
        return self._out_channels


def insert_cbam_if_enabled(component: ComponentType, cbam_enabled: bool,
                           cbam_params: Dict[str, Any] = None
                           ) -> ComponentType:
    """
    Optionally insert CBAM (Convolutional Block Attention Module) after a
    component.

    If CBAM is enabled, creates and attaches a CBAM module to the component
    using the FinalCBAMDecoder pattern for decoders or a simple sequential
    for other components.

    Args:
        component: The model component (encoder, decoder, etc.)
        cbam_enabled: Whether to insert CBAM
        cbam_params: Parameters for CBAM configuration

    Returns:
        The original component or a component wrapped with CBAM
    """
    if not cbam_enabled:
        return component

    # Get output channels from component if it has that attribute
    if hasattr(component, 'out_channels'):
        channels = component.out_channels
    else:
        log.warning("Component doesn't have 'out_channels' attribute. "
                    "Using 64 as default for CBAM.")
        channels = 64

    # Create CBAM module
    cbam = create_cbam_module(channels, cbam_params)

    # For decoders, use FinalCBAMDecoder to apply CBAM only at the end
    if hasattr(component, 'skip_channels'):
        log.debug("Using FinalCBAMDecoder for decoder with CBAM")
        return FinalCBAMDecoder(component, cbam)

    # For other components, simple sequential is fine
    log.debug("Adding CBAM to component using Sequential")
    return nn.Sequential(component, cbam)


def create_unet_from_config(config: DictConfig) -> UNetBase:
    """
    Create a UNet model from a more explicit configuration dictionary.

    This variant uses a different configuration structure where components
    are specified in encoder_config, bottleneck_config, and decoder_config.

    Args:
        config: Configuration dictionary containing encoder_config,
               bottleneck_config, and decoder_config

    Returns:
        A UNet model instance

    Raises:
        TypeError: If config is not a dictionary-like object
    """
    if not isinstance(config, (dict, DictConfig)):
        raise TypeError(
            f"Expected dictionary-like config, but got {type(config)}"
        )

    # Create components
    encoder_config = hydra_to_dict(config.encoder_config)
    encoder = instantiate_encoder(encoder_config)

    bottleneck_config = hydra_to_dict(config.bottleneck_config)
    bottleneck = instantiate_bottleneck(bottleneck_config)

    decoder_config = hydra_to_dict(config.decoder_config)
    decoder = instantiate_decoder(decoder_config)

    # Get UNet class from architecture registry
    arch_type = config.architecture_type
    architecture_registry = component_registries.get('architecture')
    unet_class = architecture_registry.get_class(arch_type)

    # Instantiate UNet
    unet = unet_class(encoder, bottleneck, decoder)
    log_component_creation("UNet", unet_class.__name__)

    # Apply CBAM if enabled
    global_cbam_enabled = config.get('cbam_enabled', False)
    if global_cbam_enabled:
        global_cbam_params = config.get('cbam_params', {})
        unet = apply_cbam_to_model(
            unet,
            global_cbam_enabled,
            global_cbam_params,
            decoder.out_channels
        )

    return unet


#
# Component Instantiation Helpers
#
def instantiate_unet_components(
    config: DictConfig
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """
    Instantiate the encoder, bottleneck, and decoder components for a UNet.

    Args:
        config: Configuration dictionary containing encoder, bottleneck,
            and decoder configs

    Returns:
        Tuple containing (encoder, bottleneck, decoder)
    """
    # Convert DictConfig to dict for instantiation functions
    encoder_cfg = hydra_to_dict(config.encoder)
    bottleneck_cfg = hydra_to_dict(config.bottleneck)
    decoder_cfg = hydra_to_dict(config.decoder)

    # 1. Instantiate Encoder
    encoder = instantiate_encoder(encoder_cfg)
    log_component_creation("Encoder", type(encoder).__name__)

    # 2. Instantiate Bottleneck with runtime parameters from encoder
    bottleneck_runtime_params = extract_runtime_params(
        encoder, {'out_channels': 'in_channels'}
    )

    bottleneck = instantiate_bottleneck(
        bottleneck_cfg, runtime_params=bottleneck_runtime_params
    )
    log_component_creation("Bottleneck", type(bottleneck).__name__)

    # 3. Instantiate Decoder with runtime parameters from bottleneck and
    # encoder
    decoder_runtime_params = extract_runtime_params(
        bottleneck, {'out_channels': 'in_channels'}
    )

    encoder_skips = getattr(encoder, 'skip_channels', [])
    if isinstance(encoder_skips, list):
        decoder_runtime_params["skip_channels_list"] = list(
            reversed(encoder_skips)
        )

    decoder = instantiate_decoder(
        decoder_cfg, runtime_params=decoder_runtime_params
    )
    log_component_creation("Decoder", type(decoder).__name__)

    return encoder, bottleneck, decoder


def get_unet_class(config: DictConfig) -> Type[UNetBase]:
    """
    Get the UNet class from configuration.

    Args:
        config: Configuration dictionary that may contain _target_

    Returns:
        UNet class to instantiate
    """
    unet_target = config.get('_target_', 'src.model.unet.BaseUNet')

    if isinstance(unet_target, str):
        # Use hydra only to get the class, not instantiate
        from hydra.utils import get_class
        return get_class(unet_target)
    else:
        # Fallback to BaseUNet if target is not a string
        log.warning(
            f"UNet target '{unet_target}' is not a string, "
            f"defaulting to BaseUNet."
        )
        return UNetBase


def add_final_activation(
    model: nn.Module, activation_config: Any
) -> nn.Module:
    """
    Add final activation layer to a model.

    Args:
        model: The model to add activation to
        activation_config: Configuration for the activation layer

    Returns:
        Model with activation added

    Raises:
        ConfigurationError: If activation instantiation fails
    """
    try:
        import hydra.utils
        activation = hydra.utils.instantiate(activation_config)
        model_with_activation = nn.Sequential(model, activation)
        log_component_creation("Final Activation", type(activation).__name__)
        return model_with_activation
    except Exception as act_e:
        log.error(
            f"Failed to instantiate final_activation: {act_e}",
            exc_info=True
        )
        raise ConfigurationError(
            f"Failed to instantiate final_activation: {act_e}"
        ) from act_e


def create_cbam_module(
    in_channels: int, cbam_params: Dict[str, Any] = None
) -> nn.Module:
    """
    Create a CBAM attention module with the specified parameters.

    Args:
        in_channels: Number of input channels for the CBAM module
        cbam_params: Additional parameters for CBAM configuration

    Returns:
        A CBAM attention module
    """
    # Default parameters if none provided
    cbam_params = cbam_params or {}

    # Get CBAM from attention registry
    attention_registry = component_registries.get('attention')
    return attention_registry.instantiate(
        "CBAM", in_channels=in_channels, **cbam_params
    )


def apply_cbam_to_model(
    model: nn.Module,
    cbam_enabled: bool,
    cbam_params: Dict[str, Any] = None,
    output_channels: Optional[int] = None
) -> nn.Module:
    """
    Apply CBAM to a model if enabled.

    Args:
        model: The model to apply CBAM to
        cbam_enabled: Whether to apply CBAM
        cbam_params: Parameters for CBAM configuration
        output_channels: Number of output channels (if None, will try to
                        determine from model)

    Returns:
        Original model or model with CBAM applied
    """
    if not cbam_enabled:
        return model

    # Determine output channels
    channels = output_channels
    if channels is None:
        if hasattr(model, 'out_channels'):
            channels = model.out_channels
        else:
            log.warning(
                "Model doesn't have 'out_channels' attribute. "
                "Using 1 as default for CBAM."
            )
            channels = 1

    # Create CBAM module
    cbam = create_cbam_module(channels, cbam_params)
    log.info("Applying CBAM as post-processor")

    # Apply CBAM
    return CBAMPostProcessor(model, cbam)
