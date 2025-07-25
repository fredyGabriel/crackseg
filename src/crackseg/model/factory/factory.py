"""
Factory functions for instantiating model components from configuration.

Provides functions to create encoders, bottlenecks, decoders, and complete
UNet models based on configuration dictionaries. Uses the registry system
for component lookup.
"""

import logging
from typing import Any, TypeVar, cast

from omegaconf import DictConfig
from torch import nn

# Actualizar importación para usar ruta correcta al paquete base
from crackseg.model.base.abstract import (
    BottleneckBase,
    DecoderBase,
    EncoderBase,
    UNetBase,
)

# Import instantiation functions usando referencias relativas
from .config import (
    instantiate_bottleneck,
    instantiate_decoder,
    # We might need instantiate_hybrid_model if create_unet handles more
    instantiate_encoder,
)

# Import factory utilities - usar referencia relativa
from .factory_utils import (
    ConfigurationError,
    extract_runtime_params,
    hydra_to_dict,
    log_component_creation,
    log_configuration_error,
    validate_config,
)

# Import global component registries - usar referencia relativa
from .registry_setup import component_registries

# Create logger
log = logging.getLogger(__name__)

# Type variable for better type hinting
ComponentType = TypeVar("ComponentType", bound=nn.Module)
ConfigDict = dict[str, Any]

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

    def __init__(self, original_model: nn.Module, cbam: nn.Module) -> None:
        super().__init__()
        self.model: nn.Module = original_model
        self.cbam: nn.Module = cbam

    def forward(self, x: Any) -> Any:
        output = self.model(x)
        return self.cbam(output)


# ======================================================================
# Factory functions
# ======================================================================


def create_unet(config: DictConfig) -> nn.Module:
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

    # Convert DictConfig to dict for validate_config
    def normalize_keys(d: Any) -> Any:
        if isinstance(d, dict):
            out: dict[str, Any] = {}
            for k, v in d.items():
                key: str = str(cast(str, k))
                value = normalize_keys(cast(Any, v))
                out[key] = value
            return out
        elif isinstance(d, list):
            return [normalize_keys(cast(Any, i)) for i in d]
        return d

    config_for_validation: dict[str, Any] = normalize_keys(dict(config))
    validate_config(config_for_validation, required_components, "unet")

    try:
        # Instantiate components
        encoder, bottleneck, decoder = instantiate_unet_components(config)

        # Cast a tipos base para satisfacer basedpyright
        encoder_base = cast(EncoderBase, encoder)
        bottleneck_base = cast(BottleneckBase, bottleneck)
        decoder_base = cast(DecoderBase, decoder)
        UnetClass: type[UNetBase] = get_unet_class(config)
        unet_model: UNetBase = UnetClass(
            encoder=encoder_base,
            bottleneck=bottleneck_base,
            decoder=decoder_base,
        )
        log_component_creation(
            "UNet", str(getattr(UnetClass, "__name__", "Unknown"))
        )
        # Si se le aplica un wrapper, el tipo es nn.Module
        model: nn.Module = unet_model
        if "final_activation" in config:
            model = add_final_activation(model, config.final_activation)
        global_cbam_enabled: bool = config.get("cbam_enabled", False)
        if global_cbam_enabled:
            global_cbam_params: dict[str, Any] = config.get("cbam_params", {})
            output_channels: int | None = getattr(
                decoder, "out_channels", None
            )
            model = apply_cbam_to_model(
                model,
                global_cbam_enabled,
                global_cbam_params,
                output_channels,
            )
        return model

    except Exception as e:
        # Catch all errors, log them, and re-raise as ConfigurationError
        log_configuration_error(
            "UNet Creation",
            str(e),
            hydra_to_dict(config),
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

    def __init__(self, decoder: nn.Module, cbam: nn.Module) -> None:
        # Initialize DecoderBase with the same parameters as the original
        # decoder
        in_channels: int = getattr(decoder, "in_channels", 1)
        skip_channels_raw = getattr(decoder, "skip_channels", [1])
        skip_channels: list[int] = (
            cast(list[int], skip_channels_raw)
            if isinstance(skip_channels_raw, list)
            else [cast(int, skip_channels_raw)]
        )
        super().__init__(
            in_channels=in_channels,
            skip_channels=skip_channels,
        )
        # Register submodules
        self.add_module("decoder", decoder)
        self.add_module("cbam", cbam)

        # Type annotations for registered modules
        self.decoder: nn.Module = decoder
        self.cbam: nn.Module = cbam

        # Copy out_channels attribute
        self._out_channels: int = getattr(decoder, "out_channels", 1)

    def forward(self, x: Any, skips: Any = None) -> Any:
        # First apply the decoder with skip_connections
        decoded = self.decoder(x, skips)
        # Then apply CBAM at the end
        return self.cbam(decoded)

    @property
    def out_channels(self) -> int:
        """
        Returns the number of output channels of the decorated decoder.
        Implementation required by DecoderBase.
        """
        return self._out_channels


def insert_cbam_if_enabled(
    component: nn.Module, config: dict[str, Any] | DictConfig
) -> nn.Module:
    """
    Optionally insert CBAM (Convolutional Block Attention Module) after a
    component, based on a config dict or OmegaConf.

    Args:
        component: The model component (encoder, decoder, etc.)
        config: Configuration dict or OmegaConf with keys 'cbam_enabled' y
        'cbam_params'

    Returns:
        The original component or a component wrapped with CBAM
    """
    cbam_enabled: bool = config.get("cbam_enabled", False)
    cbam_params: dict[str, Any] = config.get("cbam_params", {})
    if not cbam_enabled:
        return component

    # Determinar número de canales para CBAM
    channels: int
    if "in_channels" in cbam_params:
        channels = cbam_params["in_channels"]
    elif hasattr(component, "out_channels"):
        channels = getattr(component, "out_channels", 64)
    else:
        channels = 64  # valor por defecto

    # Pasar los parámetros correctos a CBAM
    cbam_args: dict[str, Any] = dict(cbam_params)
    cbam_args["in_channels"] = channels
    cbam = create_cbam_module(channels, cbam_args)

    # Para decoders, usar FinalCBAMDecoder
    if hasattr(component, "skip_channels"):
        return FinalCBAMDecoder(component, cbam)
    # Para otros, Sequential
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
    if not isinstance(config, dict):
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
    architecture_registry = component_registries.get("architecture")
    if architecture_registry is None:
        raise ConfigurationError(
            f"Architecture registry not found for type '{arch_type}'."
        )
    unet_class = architecture_registry.get(arch_type)

    # Instantiate UNet
    unet = unet_class(encoder, bottleneck, decoder)
    log_component_creation(
        "UNet",
        str(getattr(unet_class, "__name__", "Unknown")),
    )

    # Apply CBAM if enabled
    global_cbam_enabled = config.get("cbam_enabled", False)
    if global_cbam_enabled:
        global_cbam_params = config.get("cbam_params", {})
        unet = apply_cbam_to_model(
            unet,
            global_cbam_enabled,
            global_cbam_params,
            decoder.out_channels,
        )

    return cast(UNetBase, unet)


#
# Component Instantiation Helpers
#
def instantiate_unet_components(
    config: DictConfig,
) -> tuple[nn.Module, nn.Module, nn.Module]:
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
        encoder, {"out_channels": "in_channels"}
    )

    bottleneck = instantiate_bottleneck(
        bottleneck_cfg, runtime_params=bottleneck_runtime_params
    )
    log_component_creation("Bottleneck", type(bottleneck).__name__)

    # 3. Instantiate Decoder with runtime parameters from bottleneck and
    # encoder
    decoder_runtime_params = extract_runtime_params(
        bottleneck, {"out_channels": "in_channels"}
    )

    encoder_skips = getattr(encoder, "skip_channels", [])
    if isinstance(encoder_skips, list):
        list(reversed(encoder_skips))

    decoder = instantiate_decoder(
        decoder_cfg, runtime_params=decoder_runtime_params
    )
    log_component_creation("Decoder", type(decoder).__name__)

    return encoder, bottleneck, decoder


def get_unet_class(config: DictConfig) -> type[UNetBase]:
    """
    Get the UNet class from configuration.

    Args:
        config: Configuration dictionary that may contain _target_

    Returns:
        UNet class to instantiate
    """
    # Actualizar ruta predeterminada para reflejar la nueva estructura
    unet_target = config.get("_target_", "src.model.core.unet.BaseUNet")

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
            f"Failed to instantiate final_activation: {act_e}", exc_info=True
        )
        raise ConfigurationError(
            f"Failed to instantiate final_activation: {act_e}"
        ) from act_e


def create_cbam_module(
    in_channels: int, cbam_params: dict[str, Any] | None = None
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
    attention_registry = component_registries.get("attention")
    if attention_registry is None:
        raise ConfigurationError("Attention registry not found for CBAM.")

    # Default CBAM configuration
    attention_type = cbam_params.get("attention_type", "CBAM")
    attention_params = cbam_params.get(
        "attention_params", {"channels": in_channels}
    )

    return attention_registry.instantiate(attention_type, **attention_params)


def apply_cbam_to_model(
    model: nn.Module,
    cbam_enabled: bool,
    cbam_params: dict[str, Any] | None = None,
    output_channels: int | None = None,
) -> nn.Module:
    """
    Apply CBAM to a model if enabled.

    Args:
        model: The model to apply CBAM to
        cbam_enabled: Whether to apply CBAM
        cbam_params: Parameters for CBAM configuration
        output_channels: Number of output channels (if None, will try to
                        determine from crackseg.model)

    Returns:
        Original model or model with CBAM applied
    """
    if not cbam_enabled:
        return model

    # Determine output channels
    channels = output_channels
    if channels is None:
        if hasattr(model, "out_channels"):
            out_channels_attr = model.out_channels
            if isinstance(out_channels_attr, int):
                channels = out_channels_attr
            else:
                log.warning(
                    "Model out_channels is not int "
                    f"(got {type(out_channels_attr)}). "
                    "Using 1 as default for CBAM."
                )
                channels = 1
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
