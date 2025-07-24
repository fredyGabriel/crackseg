"""
Registration support for specialized model components.

Provides functions to register the various specialized components:
- ConvLSTM components
- SwinV2 encoder components
- ASPP (Atrous Spatial Pyramid Pooling) components
- CBAM (Convolutional Block Attention Module) components

These components are registered with the appropriate registries from
registry_setup.py, making them available to the factory system.
"""

import logging

from crackseg.model.core.unet import BaseUNet  # For standard UNet
from crackseg.model.factory.hybrid_registry import (
    register_complex_hybrid,
    register_standard_hybrid,
)
from crackseg.model.factory.registry_setup import (
    architecture_registry,
    bottleneck_registry,
    component_registries,
    encoder_registry,
)

# Create logger
log = logging.getLogger(__name__)


def register_standard_unet() -> None:
    """Register the standard U-Net architecture."""
    if "BaseUNet" not in architecture_registry:
        architecture_registry.register(
            name="BaseUNet", tags=["standard", "cnn"]
        )(BaseUNet)
        log.info("Registered standard BaseUNet architecture")


def register_convlstm_components() -> None:
    """
    Register all ConvLSTM-related components with the appropriate registries.

    This includes:
    - ConvLSTM cell
    - ConvLSTM layer
    - ConvLSTM bottleneck (if available)
    - CNN-ConvLSTM UNet (if available)
    """
    from crackseg.model.components.convlstm import ConvLSTM, ConvLSTMCell

    convlstm_registry = component_registries.get("convlstm")

    # Check if components are already registered
    if convlstm_registry is not None:
        if "ConvLSTMCell" not in convlstm_registry:
            convlstm_registry.register(name="ConvLSTMCell")(ConvLSTMCell)
            log.info("Registered ConvLSTMCell component")

        if "ConvLSTM" not in convlstm_registry:
            convlstm_registry.register(name="ConvLSTM")(ConvLSTM)
            log.info("Registered ConvLSTM component")
    else:
        log.warning(
            "convlstm_registry is None. Skipping ConvLSTMCell and "
            "ConvLSTM registration."
        )

    # If there's a specific bottleneck implementation using ConvLSTM
    try:
        from crackseg.model.architectures.cnn_convlstm_unet import (
            ConvLSTMBottleneck,
        )

        if "ConvLSTMBottleneck" not in bottleneck_registry:
            bottleneck_registry.register(
                name="ConvLSTMBottleneck", tags=["temporal", "convlstm"]
            )(ConvLSTMBottleneck)
            log.info("Registered ConvLSTMBottleneck with bottleneck registry")
    except ImportError:
        log.debug("ConvLSTMBottleneck not found, skipping registration")

    # If there's a CNN-ConvLSTM architecture
    try:
        from crackseg.model.architectures.cnn_convlstm_unet import (
            CNNConvLSTMUNet,
        )

        if "CNNConvLSTMUNet" not in architecture_registry:
            architecture_registry.register(
                name="CNNConvLSTMUNet", tags=["hybrid", "temporal", "convlstm"]
            )(CNNConvLSTMUNet)
            log.info("Registered CNN-ConvLSTM UNet architecture")

            register_standard_hybrid(
                name="CNNConvLSTMUNet",
                encoder_type="CNNEncoder",
                bottleneck_type="ConvLSTMBottleneck",
                decoder_type="CNNDecoder",
                tags=["temporal", "convlstm"],
            )
            log.info("Registered CNN-ConvLSTM as hybrid architecture")
    except (ImportError, ValueError):
        log.debug(
            "CNN-ConvLSTM UNet architecture not found or dependencies "
            "missing, skipping"
        )


def register_swinv2_components() -> None:
    """
    Register all SwinV2-related components with the appropriate registries.

    This includes:
    - SwinV2 encoder adapter
    - Hybrid architectures using SwinV2
    """
    try:
        from crackseg.model.encoder.swin_v2_adapter import SwinV2EncoderAdapter

        if "SwinV2" not in encoder_registry:
            encoder_registry.register(
                name="SwinV2", tags=["transformer", "attention", "pretrained"]
            )(SwinV2EncoderAdapter)
            log.info("Registered SwinV2 encoder adapter")
    except ImportError:
        log.debug("SwinV2 encoder adapter not found, skipping registration")

    # SwinV2CnnAsppUNet is now registered via decorator in the class definition
    # No manual registration needed here


def register_aspp_components() -> None:
    """
    Register all ASPP-related components with the appropriate registries.

    This includes:
    - ASPP module
    - ASPP bottleneck
    """
    try:
        from crackseg.model.components.aspp import ASPPModule

        if "ASPPModule" not in bottleneck_registry:
            bottleneck_registry.register(
                name="ASPPModule", tags=["atrous", "pyramid", "multiscale"]
            )(ASPPModule)
            log.info("Registered ASPP Module with bottleneck registry")
    except ImportError:
        log.debug("ASPP Module not found, skipping registration")


def register_cbam_components() -> None:
    """
    Register all CBAM-related components with the appropriate registries.

    This includes:
    - CBAM attention module
    - Channel Attention module
    - Spatial Attention module
    """
    try:
        from crackseg.model.components.cbam import (
            CBAM,
            ChannelAttention,
            SpatialAttention,
        )

        attention_registry = component_registries.get("attention")

        if attention_registry is not None:
            if "CBAM" not in attention_registry:
                attention_registry.register(
                    name="CBAM", tags=["attention", "channel", "spatial"]
                )(CBAM)
                log.info("Registered CBAM attention module")

            if "ChannelAttention" not in attention_registry:
                attention_registry.register(
                    name="ChannelAttention", tags=["attention", "channel"]
                )(ChannelAttention)
                log.info("Registered Channel Attention module")

            if "SpatialAttention" not in attention_registry:
                attention_registry.register(
                    name="SpatialAttention", tags=["attention", "spatial"]
                )(SpatialAttention)
                log.info("Registered Spatial Attention module")
        else:
            log.warning(
                "attention_registry is None. Skipping CBAM, "
                "ChannelAttention, and SpatialAttention registration."
            )

    except ImportError:
        log.debug("CBAM components not found, skipping registration")


def register_hybrid_architectures() -> None:
    """
    Register all hybrid architectures with the hybrid registry.

    This is a centralized function to register all hybrid architectures
    that combine multiple component types. This includes:
    - SwinV2 + ASPP + CNN U-Net
    - CNN + ConvLSTM U-Net
    - Any other hybrid architectures
    """
    log.info("Registering hybrid architectures...")

    try:
        register_complex_hybrid(
            name="SwinV2ASPPCNNWithCBAM",
            components={
                "encoder": ("encoder", "SwinV2"),
                "bottleneck": ("bottleneck", "ASPPModule"),
                "decoder": ("decoder", "CNNDecoder"),
                "attention": ("attention", "CBAM"),
            },
            tags=["hybrid", "transformer", "aspp", "attention"],
        )
    except Exception as e:
        log.warning(f"Failed to register SwinV2ASPPCNNWithCBAM: {e}")

    # SwinV2CnnAsppUNet is now registered via decorator in the class definition
    # No manual registration needed here

    # Example: CNN with a different type of attention in decoder
    # try:
    #     register_complex_hybrid(
    #         name="CNNWithCustomAttention",
    #         components={
    #             "encoder": ("encoder", "CNNEncoder"),
    #             "bottleneck": ("bottleneck", "BottleneckBlock"),
    #             "decoder": ("decoder", "CNNDecoder"),
    #             "attention": ("attention", "SomeOtherAttention"), # Fictional
    #         },
    #         tags=["hybrid", "cnn", "custom_attention"],
    #         config_overrides={"decoder_config":
    #           {"attention_module": "SomeOtherAttention"}}
    #     )
    # except Exception as e:
    #     log.warning(f"Failed to register CNNWithCustomAttention: {e}")


def register_all_components() -> None:
    """Call all registration functions to populate the registries."""
    log.info("Registering all components and architectures...")
    register_standard_unet()
    register_convlstm_components()
    register_swinv2_components()
    register_aspp_components()
    register_cbam_components()
    register_hybrid_architectures()  # Call after other components are reg.
    log.info("Component and architecture registration complete.")


# Manual registration - call register_all_components() when needed
# Auto-registration removed to prevent import circular dependencies

if __name__ == "__main__":
    # Basic logging setup for testing registration
    logging.basicConfig(level=logging.INFO)

    # Manual registration for testing
    register_all_components()

    # Example: You can try to retrieve a registered component or architecture
    # try:
    #     swin_encoder_cls = encoder_registry.get("SwinV2")
    #     log.info(f"Successfully retrieved SwinV2: {swin_encoder_cls}")
    #     aspp_bottleneck_cls = bottleneck_registry.get("ASPPModule")
    #     log.info(f"Successfully retrieved ASPPModule: {aspp_bottleneck_cls}")
    #     hybrid_arch = architecture_registry.get("SwinV2ASPPCNNWithCBAM")
    #     log.info(f"Successfully retrieved SwinV2ASPPCNNWithCBAM: {hybrid_arch}")  # noqa: E501
    # except KeyError as e:
    #     log.error(f"Failed to retrieve a registered item: {e}")
