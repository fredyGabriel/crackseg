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

from src.model.registry_setup import (
    encoder_registry,
    bottleneck_registry,
    architecture_registry,
    component_registries
)
from src.model.hybrid_registry import (
    register_standard_hybrid,
    register_complex_hybrid
)

# Create logger
log = logging.getLogger(__name__)


def register_convlstm_components() -> None:
    """
    Register all ConvLSTM-related components with the appropriate registries.

    This includes:
    - ConvLSTMCell
    - ConvLSTM
    - ConvLSTM bottleneck
    - CNN-ConvLSTM architecture
    """
    # Register the components if not already registered
    from src.model.components.convlstm import ConvLSTMCell, ConvLSTM

    # Use the dedicated ConvLSTM registry
    convlstm_registry = component_registries.get('convlstm')

    # Check if components are already registered
    if 'ConvLSTMCell' not in convlstm_registry:
        # Register with the component registry
        convlstm_registry.register(name='ConvLSTMCell')(ConvLSTMCell)
        log.info("Registered ConvLSTMCell component")

    if 'ConvLSTM' not in convlstm_registry:
        # Register with the component registry
        convlstm_registry.register(name='ConvLSTM')(ConvLSTM)
        log.info("Registered ConvLSTM component")

    # If there's a specific bottleneck implementation using ConvLSTM
    try:
        from src.model.bottleneck.convlstm_bottleneck import (  # type: ignore
            ConvLSTMBottleneck
        )
        if 'ConvLSTMBottleneck' not in bottleneck_registry:
            bottleneck_registry.register(
                name='ConvLSTMBottleneck', tags=['temporal', 'convlstm']
            )(ConvLSTMBottleneck)
            log.info("Registered ConvLSTMBottleneck with bottleneck registry")
    except ImportError:
        log.debug("ConvLSTMBottleneck not found, skipping registration")

    # If there's a CNN-ConvLSTM architecture
    try:
        from src.model.architectures.cnn_convlstm_unet import CNNConvLSTMUNet
        if 'CNNConvLSTMUNet' not in architecture_registry:
            architecture_registry.register(
                name='CNNConvLSTMUNet',
                tags=['hybrid', 'temporal', 'convlstm']
            )(CNNConvLSTMUNet)
            log.info("Registered CNN-ConvLSTM UNet architecture")

            # Register as a hybrid architecture
            register_standard_hybrid(
                name='CNNConvLSTMUNet',
                encoder_type='CNNEncoder',  # Assuming this exists
                bottleneck_type='ConvLSTMBottleneck',
                decoder_type='CNNDecoder',  # Assuming this exists
                tags=['temporal', 'convlstm']
            )
            log.info("Registered CNN-ConvLSTM as hybrid architecture")
    except (ImportError, ValueError):
        log.debug("CNN-ConvLSTM UNet architecture not found or dependencies "
                  "missing, skipping")


def register_swinv2_components() -> None:
    """
    Register all SwinV2-related components with the appropriate registries.

    This includes:
    - SwinV2 encoder adapter
    - Hybrid architectures using SwinV2
    """
    # Register the SwinV2 encoder if not already registered
    try:
        from src.model.encoder.swin_v2_adapter import SwinV2EncoderAdapter
        if 'SwinV2' not in encoder_registry:
            encoder_registry.register(
                name='SwinV2',
                tags=['transformer', 'attention', 'pretrained']
            )(SwinV2EncoderAdapter)
            log.info("Registered SwinV2 encoder adapter")
    except ImportError:
        log.debug("SwinV2 encoder adapter not found, skipping registration")

    # Register any hybrid architectures using SwinV2
    try:
        from src.model.architectures.swinv2_cnn_aspp_unet import (
            SwinV2CNNASPPUNet
        )
        if 'SwinV2CNNASPPUNet' not in architecture_registry:
            architecture_registry.register(
                name='SwinV2CNNASPPUNet',
                tags=['hybrid', 'transformer', 'aspp']
            )(SwinV2CNNASPPUNet)
            log.info("Registered SwinV2-CNN-ASPP hybrid architecture")

            # Register as a hybrid architecture
            register_standard_hybrid(
                name='SwinV2CNNASPPUNet',
                encoder_type='SwinV2',
                bottleneck_type='ASPPModule',
                decoder_type='CNNDecoder',  # Assuming this exists
                tags=['transformer', 'aspp']
            )
            log.info("Registered SwinV2-CNN-ASPP as hybrid architecture")
    except (ImportError, ValueError):
        log.debug(
            "SwinV2-CNN-ASPP hybrid architecture not found or dependencies "
            "missing, skipping"
        )


def register_aspp_components() -> None:
    """
    Register all ASPP-related components with the appropriate registries.

    This includes:
    - ASPP module
    - ASPP bottleneck
    """
    # Register the ASPP component if not already registered
    try:
        from src.model.components.aspp import ASPPModule
        if 'ASPPModule' not in bottleneck_registry:
            bottleneck_registry.register(
                name='ASPPModule',
                tags=['atrous', 'pyramid', 'multiscale']
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
    # Register the CBAM components if not already registered
    try:
        from src.model.components.cbam import (
            CBAM, ChannelAttention, SpatialAttention
        )

        attention_registry = component_registries.get('attention')

        # Register CBAM (if not already registered)
        if 'CBAM' not in attention_registry:
            attention_registry.register(
                name='CBAM',
                tags=['attention', 'channel', 'spatial']
            )(CBAM)
            log.info("Registered CBAM attention module")

        # Register channel attention (if not already registered)
        if 'ChannelAttention' not in attention_registry:
            attention_registry.register(
                name='ChannelAttention',
                tags=['attention', 'channel']
            )(ChannelAttention)
            log.info("Registered Channel Attention module")

        # Register spatial attention (if not already registered)
        if 'SpatialAttention' not in attention_registry:
            attention_registry.register(
                name='SpatialAttention',
                tags=['attention', 'spatial']
            )(SpatialAttention)
            log.info("Registered Spatial Attention module")

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

    # Hybrid: SwinV2 encoder + ASPP bottleneck + CNN decoder with CBAM
    try:
        register_complex_hybrid(
            name="SwinV2ASPPCNNWithCBAM",
            components={
                'encoder': ('encoder', 'SwinV2'),
                'bottleneck': ('bottleneck', 'ASPPModule'),
                'decoder': ('decoder', 'CNNDecoder'),  # Assuming this exists
                'attention': ('attention', 'CBAM')
            },
            tags=['transformer', 'aspp', 'attention', 'cnn']
        )
        log.info("Registered SwinV2ASPPCNNWithCBAM hybrid architecture")
    except ValueError as e:
        log.debug(f"Could not register SwinV2ASPPCNNWithCBAM: {e}")

    # Add other hybrid architectures as needed


def register_all_components() -> None:
    """
    Register all component types with their appropriate registries.

    This is a convenience function to initialize all component registrations
    at once.
    """
    log.info("Registering all model components...")

    # Register each component type
    register_convlstm_components()
    register_swinv2_components()
    register_aspp_components()
    register_cbam_components()
    register_hybrid_architectures()

    log.info("Component registration complete")


# Optional initialization on import
# register_all_components()
