"""Fixtures and setup for model tests."""

import pytest  # noqa E402

# Import base classes and registry instances needed for mocks
from src.model.base import EncoderBase, BottleneckBase, DecoderBase  # noqa E402
from src.model.factory import (
    encoder_registry, bottleneck_registry, decoder_registry
)

# Import mock base classes (assuming they are defined elsewhere, e.g.,
# test_registry or a common mocks file)
# If MockEncoder/Decoder/Bottleneck are only used here, define them here.
# For now, assume they come from test_registry for simplicity in this example.
from tests.model.test_registry import (
    MockEncoder, MockBottleneck, MockDecoder
)


# Define and register mock components centrally here
@encoder_registry.register("FactoryEncoder")
class FactoryEncoder(MockEncoder):
    # Override skip_channels if necessary, default is [16, 32]
    pass


# Keep original name if not conflicting
@bottleneck_registry.register("TestBottleneck")
class TestBottleneck(MockBottleneck):
    pass


@decoder_registry.register("FactoryDecoder")
class FactoryDecoder(MockDecoder):
    # MockDecoder already stores skip_channels passed in __init__
    # The validation expects the *reversed* order compared to the encoder.
    # The base class MockDecoder init already handles storing the list.
    # The issue is likely in how it's passed during UNet creation in tests.
    # No changes needed here directly, but ensure tests pass the expected
    # reversed list to the UNet which then passes it to the decoder.
    # Let's double-check MockDecoder in test_registry.py just in case.
    pass

# Add fixtures if needed, e.g.:
# @pytest.fixture
# def sample_encoder():
#     return FactoryEncoder(in_channels=3)
