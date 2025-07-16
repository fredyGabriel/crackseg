"""
Test suite to verify import compatibility of all main model components.

This test ensures that all base classes, implementations, utilities,
and factory components can be imported both from their specific locations
and from the top-level src.model module.
"""

# --- Test structure will be filled in subtareas siguientes ---


def test_import_base_classes_specific():
    """Test import of base classes from their specific module locations."""
    from crackseg.model.base.abstract import (
        BottleneckBase,
        DecoderBase,
        EncoderBase,
        UNetBase,
    )

    assert EncoderBase is not None
    assert DecoderBase is not None
    assert BottleneckBase is not None
    assert UNetBase is not None


def test_import_base_classes_global():
    """Test import of base classes from the top-level src.model module."""
    from crackseg.model import (
        BottleneckBase,
        DecoderBase,
        EncoderBase,
        UNetBase,
    )

    assert EncoderBase is not None
    assert DecoderBase is not None
    assert BottleneckBase is not None
    assert UNetBase is not None


def test_import_implementation_classes_specific():
    """
    Test import of main implementation classes from their specific modules.
    """
    from crackseg.model.architectures import (
        CNNConvLSTMUNet,
        CNNDecoder,
        CNNEncoder,
    )
    from crackseg.model.architectures.swinv2_cnn_aspp_unet import (
        SwinV2CnnAsppUNet,
    )
    from crackseg.model.bottleneck.cnn_bottleneck import BottleneckBlock
    from crackseg.model.components.aspp import ASPPModule
    from crackseg.model.encoder.swin_v2_adapter import SwinV2EncoderAdapter

    assert CNNEncoder is not None
    assert CNNDecoder is not None
    assert CNNConvLSTMUNet is not None
    assert SwinV2CnnAsppUNet is not None
    assert SwinV2EncoderAdapter is not None
    assert ASPPModule is not None
    assert BottleneckBlock is not None


def test_import_implementation_classes_global():
    """Test import of main implementation classes from the top-level src.model
    module."""
    from crackseg.model import (
        ASPPModule,
        BottleneckBlock,
        CNNConvLSTMUNet,
        CNNDecoder,
        CNNEncoder,
        SwinV2CnnAsppUNet,
        SwinV2EncoderAdapter,
    )

    assert CNNEncoder is not None
    assert CNNDecoder is not None
    assert CNNConvLSTMUNet is not None
    assert SwinV2CnnAsppUNet is not None
    assert SwinV2EncoderAdapter is not None
    assert ASPPModule is not None
    assert BottleneckBlock is not None


def test_import_utilities_specific():
    """Test import of utility functions and modules from their specific
    locations."""
    from crackseg.model.factory.factory_utils import merge_configs
    from crackseg.model.factory.registry import Registry

    assert merge_configs is not None
    assert Registry is not None


def test_import_utilities_global():
    """Test import of utility functions and modules from the top-level
    src.model module."""
    # No utilidades re-exportadas globalmente por ahora.
    # Si en el futuro se re-exportan, añadir aquí los imports y asserts.
    pass


def test_import_factory_components_specific():
    """Test import of main factory components from their specific modules."""
    from crackseg.model.factory.factory import create_unet
    from crackseg.model.factory.registry_setup import (
        architecture_registry,
        bottleneck_registry,
        component_registries,
        decoder_registry,
        encoder_registry,
    )

    assert create_unet is not None
    assert encoder_registry is not None
    assert bottleneck_registry is not None
    assert decoder_registry is not None
    assert architecture_registry is not None
    assert component_registries is not None


def test_import_factory_components_global():
    """Test import of main factory components from the top-level src.model
    module."""
    # No componentes de fábrica re-exportados globalmente por ahora.
    # Si en el futuro se re-exportan, añadir aquí los imports y asserts.
    pass
