import pytest
import torch
from typing import List, Tuple

# Import the base class to be tested
from src.model import EncoderBase

# Import the new base class
from src.model import BottleneckBase

# Import the new base class
from src.model import DecoderBase

# Import the new base class
from src.model import UNetBase


# 1. Test that EncoderBase itself cannot be instantiated
def test_encoder_base_cannot_be_instantiated():
    """Verify that TypeError is raised when trying to instantiate EncoderBase.
    """
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        EncoderBase(in_channels=3)


# 2. Define minimal concrete subclasses for testing purposes

class MinimalEncoderMissingForward(EncoderBase):
    """A minimal concrete encoder missing the 'forward' implementation."""
    def __init__(self, in_channels: int = 3):
        super().__init__(in_channels)

    @property
    def out_channels(self) -> int:
        return 64  # Dummy value

    @property
    def skip_channels(self) -> List[int]:
        return [16, 32]  # Dummy value


class MinimalEncoderMissingOutChannels(EncoderBase):
    """A minimal concrete encoder missing the 'out_channels' property."""
    def __init__(self, in_channels: int = 3):
        super().__init__(in_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor,
                                                List[torch.Tensor]]:
        # Dummy implementation for testing instantiation
        # Returns dummy tensors with plausible channel/spatial dimensions
        return torch.randn(x.size(0), 64, x.size(2)//4, x.size(3)//4), \
               [torch.randn(x.size(0), 16, x.size(2), x.size(3)),
                torch.randn(x.size(0), 32, x.size(2)//2, x.size(3)//2)]

    @property
    def skip_channels(self) -> List[int]:
        return [16, 32]  # Dummy value


class MinimalEncoderMissingSkipChannels(EncoderBase):
    """A minimal concrete encoder missing the 'skip_channels' property."""
    def __init__(self, in_channels: int = 3):
        super().__init__(in_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor,
                                                List[torch.Tensor]]:
        # Dummy implementation for testing instantiation
        # Returns dummy tensors with plausible channel/spatial dimensions
        return torch.randn(x.size(0), 64, x.size(2)//4, x.size(3)//4), \
               [torch.randn(x.size(0), 16, x.size(2), x.size(3)),
                torch.randn(x.size(0), 32, x.size(2)//2, x.size(3)//2)]

    @property
    def out_channels(self) -> int:
        return 64  # Dummy value


class MinimalValidEncoder(EncoderBase):
    """A minimal concrete encoder implementing all abstract members."""
    def __init__(self, in_channels: int = 3):
        super().__init__(in_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor,
                                                List[torch.Tensor]]:
        # Dummy implementation for testing instantiation
        # Returns dummy tensors with plausible channel/spatial dimensions
        skip1 = torch.randn(x.size(0), 16, x.size(2), x.size(3))
        skip2 = torch.randn(x.size(0), 32, x.size(2)//2, x.size(3)//2)
        output = torch.randn(x.size(0), 64, x.size(2)//4, x.size(3)//4)
        return output, [skip1, skip2]

    @property
    def out_channels(self) -> int:
        return 64  # Must match channel dim of 'output' tensor above

    @property
    def skip_channels(self) -> List[int]:
        # Must match channel dims of 'skip1', 'skip2' tensors above
        return [16, 32]


# 3. Tests for concrete subclasses

def test_incomplete_encoder_missing_forward_raises_type_error():
    """Verify TypeError if 'forward' is not implemented."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        MinimalEncoderMissingForward(in_channels=3)


def test_incomplete_encoder_missing_out_channels_raises_type_error():
    """Verify TypeError if 'out_channels' property is not implemented."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        MinimalEncoderMissingOutChannels(in_channels=3)


def test_incomplete_encoder_missing_skip_channels_raises_type_error():
    """Verify TypeError if 'skip_channels' property is not implemented."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        MinimalEncoderMissingSkipChannels(in_channels=3)


def test_valid_minimal_encoder_instantiates():
    """Verify a valid concrete subclass can be instantiated."""
    try:
        encoder = MinimalValidEncoder(in_channels=3)
        assert isinstance(encoder, EncoderBase)
        # Check if properties return expected dummy values
        assert encoder.out_channels == 64
        assert encoder.skip_channels == [16, 32]
    except TypeError:
        pytest.fail("Valid minimal encoder failed to instantiate.")


# Optional: Test the forward pass contract (basic shape check)
def test_valid_minimal_encoder_forward_contract():
    """Check if the forward pass returns the expected types and list length."""
    encoder = MinimalValidEncoder(in_channels=3)
    # Batch size 2, 3 channels, 128x128
    dummy_input = torch.randn(2, 3, 128, 128)
    try:
        output, skips = encoder.forward(dummy_input)
        assert isinstance(output, torch.Tensor)
        assert isinstance(skips, list)
        # Check if list length matches property
        assert len(skips) == len(encoder.skip_channels)
        for i, skip_tensor in enumerate(skips):
            assert isinstance(skip_tensor, torch.Tensor)
            # Basic check: Ensure skip channel dimensions match the property
            assert skip_tensor.shape[1] == encoder.skip_channels[i]

    except Exception as e:
        pytest.fail(f"MinimalValidEncoder forward pass failed: {e}")


# --- Tests for BottleneckBase ---

# 4. Test that BottleneckBase itself cannot be instantiated
def test_bottleneck_base_cannot_be_instantiated():
    """Verify TypeError raised when trying to instantiate BottleneckBase."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BottleneckBase(in_channels=64)  # Assuming 64 channels from encoder


# 5. Define minimal concrete subclasses for testing BottleneckBase

class MinimalBottleneckMissingForward(BottleneckBase):
    """A minimal bottleneck missing the 'forward' implementation."""
    def __init__(self, in_channels: int = 64):
        super().__init__(in_channels)

    @property
    def out_channels(self) -> int:
        return 128  # Dummy value


class MinimalBottleneckMissingOutChannels(BottleneckBase):
    """A minimal bottleneck missing the 'out_channels' property."""
    def __init__(self, in_channels: int = 64):
        super().__init__(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dummy implementation just returning input shape but different
        # channels
        return torch.randn(x.size(0), 128, x.size(2), x.size(3))


class MinimalValidBottleneck(BottleneckBase):
    """A minimal bottleneck implementing all abstract members."""
    _out_channels = 128  # Define as class attribute for consistency

    def __init__(self, in_channels: int = 64):
        super().__init__(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dummy implementation
        return torch.randn(x.size(0), self._out_channels, x.size(2), x.size(3))

    @property
    def out_channels(self) -> int:
        return self._out_channels


# 6. Tests for concrete Bottleneck subclasses

def test_incomplete_bottleneck_missing_forward_raises_type_error():
    """Verify TypeError if 'forward' is not implemented."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        MinimalBottleneckMissingForward(in_channels=64)


def test_incomplete_bottleneck_missing_out_channels_raises_type_error():
    """Verify TypeError if 'out_channels' property is not implemented."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        MinimalBottleneckMissingOutChannels(in_channels=64)


def test_valid_minimal_bottleneck_instantiates():
    """Verify a valid concrete bottleneck subclass can be instantiated."""
    try:
        bottleneck = MinimalValidBottleneck(in_channels=64)
        assert isinstance(bottleneck, BottleneckBase)
        assert bottleneck.out_channels == 128
    except TypeError:
        pytest.fail("Valid minimal bottleneck failed to instantiate.")


def test_valid_minimal_bottleneck_forward_contract():
    """Check forward pass returns expected type and channels."""
    bottleneck = MinimalValidBottleneck(in_channels=64)
    # Dummy input resembling encoder output (e.g., batch 2, 64 channels, 16x16)
    dummy_input = torch.randn(2, 64, 16, 16)
    try:
        output = bottleneck.forward(dummy_input)
        assert isinstance(output, torch.Tensor)
        # Check output channels match the property
        assert output.shape[1] == bottleneck.out_channels
        # Check spatial dimensions (typically preserved in bottleneck)
        assert output.shape[2] == dummy_input.shape[2]
        assert output.shape[3] == dummy_input.shape[3]

    except Exception as e:
        pytest.fail(f"MinimalValidBottleneck forward pass failed: {e}")


# --- Tests for DecoderBase ---

# 7. Test that DecoderBase itself cannot be instantiated
def test_decoder_base_cannot_be_instantiated():
    """Verify TypeError is raised when trying to instantiate DecoderBase."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        # Dummy skip channels expected from encoder
        skip_channels_list = [16, 32]
        # Assuming 128 from bottleneck
        DecoderBase(in_channels=128, skip_channels=skip_channels_list)


# 8. Define minimal concrete subclasses for testing DecoderBase

class MinimalDecoderMissingForward(DecoderBase):
    """A minimal decoder missing the 'forward' implementation."""
    def __init__(self, in_channels: int = 128,
                 skip_channels: List[int] = [16, 32]):
        super().__init__(in_channels, skip_channels)

    @property
    def out_channels(self) -> int:
        return 1  # Dummy value for binary segmentation


class MinimalDecoderMissingOutChannels(DecoderBase):
    """A minimal decoder missing the 'out_channels' property."""
    def __init__(self, in_channels: int = 128,
                 skip_channels: List[int] = [16, 32]):
        super().__init__(in_channels, skip_channels)

    def forward(self, x: torch.Tensor,
                skips: List[torch.Tensor]) -> torch.Tensor:
        # Dummy impl returning tensor with expected output channels
        # but potentially wrong spatial dims for simplicity here
        # Example upsampling
        return torch.randn(x.size(0), 1, x.size(2)*4, x.size(3)*4)


class MinimalValidDecoder(DecoderBase):
    """A minimal decoder implementing all abstract members."""
    _out_channels = 1  # Binary segmentation

    def __init__(self, in_channels: int = 128,
                 skip_channels: List[int] = [16, 32]):
        # Basic check on input types
        if not isinstance(skip_channels, list) or not all(
            isinstance(c, int) for c in skip_channels
        ):
            raise TypeError("skip_channels must be a list of integers")
        # Store skip_channels in the REVERSED order expected by base validation
        skips_to_store = list(reversed(skip_channels))
        super().__init__(in_channels, skips_to_store)

    def forward(self, x: torch.Tensor,
                skips: List[torch.Tensor]) -> torch.Tensor:
        # Basic check: ensure number of skips matches expected list length
        assert len(skips) == len(self.skip_channels)
        # Dummy implementation assuming upsampling back to original size
        # This doesn't actually use skips, just checks contract
        # Example, assumes 2 stages of 2x upsampling
        output_h = x.size(2) * 4
        output_w = x.size(3) * 4
        return torch.randn(x.size(0), self.out_channels, output_h, output_w)

    @property
    def out_channels(self) -> int:
        return self._out_channels


# 9. Tests for concrete Decoder subclasses

def test_incomplete_decoder_missing_forward_raises_type_error():
    """Verify TypeError if 'forward' is not implemented."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        MinimalDecoderMissingForward(in_channels=128, skip_channels=[16, 32])


def test_incomplete_decoder_missing_out_channels_raises_type_error():
    """Verify TypeError if 'out_channels' property is not implemented."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        MinimalDecoderMissingOutChannels(in_channels=128,
                                         skip_channels=[16, 32])


def test_valid_minimal_decoder_instantiates():
    """Verify a valid concrete decoder subclass can be instantiated."""
    try:
        decoder = MinimalValidDecoder(in_channels=128, skip_channels=[16, 32])
        assert isinstance(decoder, DecoderBase)
        assert decoder.out_channels == 1
    except TypeError:
        pytest.fail("Valid minimal decoder failed to instantiate.")
    except AssertionError as e:
        pytest.fail(f"Assertion failed during decoder init: {e}")


def test_valid_minimal_decoder_forward_contract():
    """Check forward pass accepts skips & returns expected type/channels."""
    skip_channels_list = [16, 32]
    decoder = MinimalValidDecoder(in_channels=128,
                                  skip_channels=skip_channels_list)

    # Dummy input resembling bottleneck output
    dummy_bottleneck_out = torch.randn(2, 128, 16, 16)
    # Dummy skip connections resembling encoder outputs
    # (matching skip_channels_list)
    dummy_skip1 = torch.randn(2, skip_channels_list[0], 64, 64)  # Higher res
    dummy_skip2 = torch.randn(2, skip_channels_list[1], 32, 32)  # Lower res
    dummy_skips = [dummy_skip1, dummy_skip2]

    try:
        output = decoder.forward(dummy_bottleneck_out, dummy_skips)
        assert isinstance(output, torch.Tensor)
        # Check output channels match the property
        assert output.shape[1] == decoder.out_channels

    except Exception as e:
        pytest.fail(f"MinimalValidDecoder forward pass failed: {e}")


# --- Tests for UNetBase ---

# 10. Test that UNetBase itself cannot be instantiated

def test_unet_base_cannot_be_instantiated():
    """Verify that TypeError is raised when trying to instantiate UNetBase."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        UNetBase(
            encoder=MinimalValidEncoder(),
            bottleneck=MinimalValidBottleneck(),
            decoder=MinimalValidDecoder()
        )


# 11. Minimal concrete subclass for UNetBase (missing forward)

class MinimalUNetMissingForward(UNetBase):
    def __init__(self, encoder, bottleneck, decoder):
        super().__init__(encoder, bottleneck, decoder)


# 12. Minimal valid UNet implementation

class MinimalValidUNet(UNetBase):
    def __init__(self, encoder, bottleneck, decoder):
        super().__init__(encoder, bottleneck, decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck_out, skips = self.encoder(x)
        bottleneck_out = self.bottleneck(bottleneck_out)
        return self.decoder(bottleneck_out, skips)


# 13. Test that missing forward raises TypeError

def test_incomplete_unet_missing_forward_raises_type_error():
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        MinimalUNetMissingForward(
            encoder=MinimalValidEncoder(),
            bottleneck=MinimalValidBottleneck(),
            decoder=MinimalValidDecoder()
        )


# 14. Test valid minimal UNet instantiates and forward works

def test_valid_minimal_unet_instantiates_and_forward():
    encoder = MinimalValidEncoder()
    bottleneck = MinimalValidBottleneck()
    decoder = MinimalValidDecoder()
    unet = MinimalValidUNet(encoder, bottleneck, decoder)
    assert isinstance(unet, UNetBase)
    # Check properties
    assert unet.in_channels == encoder.in_channels
    assert unet.out_channels == decoder.out_channels
    # Forward pass
    dummy_input = torch.randn(2, 3, 128, 128)
    output = unet(dummy_input)
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == 2
    assert output.shape[1] == decoder.out_channels


# 15. Test component compatibility validation

def test_unet_component_compatibility_checks():
    encoder = MinimalValidEncoder()
    bottleneck = MinimalValidBottleneck()
    decoder = MinimalValidDecoder()

    # Change encoder out_channels to break compatibility
    class BadEncoder(MinimalValidEncoder):
        @property
        def out_channels(self):
            return 999

    with pytest.raises(ValueError, match="output channels"):
        MinimalValidUNet(BadEncoder(), bottleneck, decoder)

    # Change bottleneck out_channels to break compatibility
    class BadBottleneck(MinimalValidBottleneck):
        @property
        def out_channels(self):
            return 999

    with pytest.raises(ValueError, match="output channels"):
        MinimalValidUNet(encoder, BadBottleneck(), decoder)

    # Change skip_channels to break compatibility
    class BadDecoder(DecoderBase):
        """A decoder with incompatible skip channels for testing."""
        def __init__(self, in_channels: int = 128,
                     skip_channels: List[int] = None):
            super().__init__(in_channels, [1, 2, 3])  # Incompatible channels

        def forward(self, x: torch.Tensor,
                    skips: List[torch.Tensor]) -> torch.Tensor:
            # Dummy implementation
            return torch.randn(x.size(0), self.out_channels,
                               x.size(2)*4, x.size(3)*4)

        @property
        def out_channels(self) -> int:
            return 1  # Binary segmentation

    with pytest.raises(ValueError, match="skip channel"):
        MinimalValidUNet(encoder, bottleneck, BadDecoder())


# --- Integration Tests ---

def test_full_unet_data_flow():
    """Test the complete data flow through a valid UNet implementation."""
    # Create components with compatible dimensions
    encoder = MinimalValidEncoder(in_channels=3)  # RGB input
    bottleneck = MinimalValidBottleneck(in_channels=encoder.out_channels)
    decoder = MinimalValidDecoder(
        in_channels=bottleneck.out_channels,
        skip_channels=encoder.skip_channels
    )
    unet = MinimalValidUNet(encoder, bottleneck, decoder)

    # Test with a batch of 2 RGB images of size 64x64
    x = torch.randn(2, 3, 64, 64)

    # Forward pass should complete without errors
    try:
        output = unet(x)

        # Verify output dimensions
        assert output.shape[0] == x.shape[0]  # Batch size preserved
        assert output.shape[1] == decoder.out_channels  # Classes
        assert output.shape[2:] == x.shape[2:]  # Spatial dims preserved

    except Exception as e:
        pytest.fail(f"Full UNet forward pass failed: {e}")


def test_skip_connection_dimensions():
    """Verify skip connection dimensions are preserved through the network."""
    encoder = MinimalValidEncoder(in_channels=3)
    bottleneck = MinimalValidBottleneck(in_channels=encoder.out_channels)
    decoder = MinimalValidDecoder(
        in_channels=bottleneck.out_channels,
        skip_channels=encoder.skip_channels
    )
    # Create UNet but don't store reference since we test components directly
    MinimalValidUNet(encoder, bottleneck, decoder)

    # Input tensor
    x = torch.randn(2, 3, 64, 64)

    # Get encoder output and skip connections
    encoder_output, skips = encoder(x)

    # Verify skip connection dimensions
    assert len(skips) == len(encoder.skip_channels)
    for skip, channels in zip(skips, encoder.skip_channels):
        assert skip.shape[1] == channels

    # Verify bottleneck dimensions
    bottleneck_output = bottleneck(encoder_output)
    assert bottleneck_output.shape[1] == bottleneck.out_channels

    # Verify decoder can process bottleneck output with skip connections
    try:
        decoder_output = decoder(bottleneck_output, skips)
        assert decoder_output.shape[1] == decoder.out_channels
        assert decoder_output.shape[2:] == x.shape[2:]
    except Exception as e:
        msg = "Decoder failed to process bottleneck output with skips: {}"
        pytest.fail(msg.format(e))


def test_component_validation_error_messages():
    """Test that component validation provides clear error messages."""
    encoder = MinimalValidEncoder(in_channels=3)
    bottleneck = MinimalValidBottleneck(in_channels=encoder.out_channels)
    decoder = MinimalValidDecoder(
        in_channels=bottleneck.out_channels,
        skip_channels=encoder.skip_channels
    )

    # Test invalid encoder type
    with pytest.raises(
            TypeError, match="encoder must be an instance of EncoderBase"):
        MinimalValidUNet(None, bottleneck, decoder)

    # Test invalid bottleneck type
    with pytest.raises(
            TypeError,
            match="bottleneck must be an instance of BottleneckBase"):
        MinimalValidUNet(encoder, None, decoder)

    # Test invalid decoder type
    with pytest.raises(
            (TypeError, AttributeError),  # Acepta TypeError o AttributeError
            match="(decoder must be an instance of DecoderBase|" +
                  "'NoneType' object has no attribute)"):
        MinimalValidUNet(encoder, bottleneck, None)

    # Test channel mismatch between encoder and bottleneck
    class MismatchedBottleneck(MinimalValidBottleneck):
        def __init__(self):
            super().__init__(in_channels=32)  # Wrong input channels

    err_msg = "Encoder output channels .* must match bottleneck input channels"
    with pytest.raises(ValueError, match=err_msg):
        MinimalValidUNet(encoder, MismatchedBottleneck(), decoder)

    # Test channel mismatch between bottleneck and decoder
    class MismatchedDecoder(MinimalValidDecoder):
        def __init__(self):
            super().__init__(
                in_channels=64,  # Wrong input channels
                skip_channels=encoder.skip_channels
            )

    err_msg = "Bottleneck output channels .* must match decoder input channels"
    with pytest.raises(ValueError, match=err_msg):
        MinimalValidUNet(encoder, bottleneck, MismatchedDecoder())


def test_skip_connection_compatibility():
    """Test that skip connection compatibility is properly enforced."""
    encoder = MinimalValidEncoder(in_channels=3)
    bottleneck = MinimalValidBottleneck(in_channels=encoder.out_channels)

    # Test with incompatible skip channels (wrong number of channels)
    class IncompatibleDecoder(DecoderBase):
        """Decoder con skip_channels incompatibles para test."""
        def __init__(self):
            # Valores incompatibles de skip_channels
            # diferentes a los del encoder (que son [16, 32])
            in_channels = bottleneck.out_channels
            skip_channels = [8, 16]  # Diferente del encoder
            super().__init__(in_channels, skip_channels)
            self._out_channels = 1

        def forward(self, x, skips):
            # Implementación mínima
            return torch.zeros(1, 1, 64, 64)

        @property
        def out_channels(self):
            return self._out_channels

    # Crear el decoder incompatible
    incompatible_decoder = IncompatibleDecoder()

    # Verify that a warning is emitted when creating the UNet
    # In UNetBase a warning is emitted,
    # while in BaseUNet a ValueError is raised
    import warnings
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered
        warnings.simplefilter("always")

        # Create the model - should emit a warning
        MinimalValidUNet(encoder, bottleneck, incompatible_decoder)

        # Verify that the correct warning was emitted
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "Encoder skip channels" in str(w[0].message)
        assert (
            "don't match reversed decoder skip channels" in str(w[0].message)
        )
