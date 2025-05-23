from typing import cast

import pytest
import torch

from src.model import BottleneckBase, DecoderBase, EncoderBase, UNetBase

# Constants for test values
IN_CHANNELS_RGB = 3
SKIP_CHANNELS = [16, 32]
OUT_CHANNELS_ENCODER = 64
IN_CHANNELS_BOTTLENECK = 64
OUT_CHANNELS_BOTTLENECK = 128
IN_CHANNELS_DECODER = 128
OUT_CHANNELS_DECODER = 1


# 1. Test that EncoderBase itself cannot be instantiated
def test_encoder_base_cannot_be_instantiated():
    """
    Verify that TypeError is raised when trying to instantiate EncoderBase.
    """
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        EncoderBase(in_channels=IN_CHANNELS_RGB)  # type: ignore[abstract]


# 2. Define minimal concrete subclasses for testing purposes


class MinimalEncoderMissingForward(EncoderBase):
    """A minimal concrete encoder missing the 'forward' implementation."""

    def __init__(self, in_channels: int = IN_CHANNELS_RGB):
        super().__init__(in_channels)

    @property
    def out_channels(self) -> int:
        return OUT_CHANNELS_ENCODER

    @property
    def skip_channels(self) -> list[int]:
        return SKIP_CHANNELS


class MinimalEncoderMissingOutChannels(EncoderBase):
    """A minimal concrete encoder missing the 'out_channels' property."""

    def __init__(self, in_channels: int = IN_CHANNELS_RGB):
        super().__init__(in_channels)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return torch.randn(
            x.size(0), OUT_CHANNELS_ENCODER, x.size(2) // 4, x.size(3) // 4
        ), [
            torch.randn(x.size(0), SKIP_CHANNELS[0], x.size(2), x.size(3)),
            torch.randn(
                x.size(0), SKIP_CHANNELS[1], x.size(2) // 2, x.size(3) // 2
            ),
        ]

    @property
    def skip_channels(self) -> list[int]:
        return SKIP_CHANNELS


class MinimalEncoderMissingSkipChannels(EncoderBase):
    """A minimal concrete encoder missing the 'skip_channels' property."""

    def __init__(self, in_channels: int = IN_CHANNELS_RGB):
        super().__init__(in_channels)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return torch.randn(
            x.size(0), OUT_CHANNELS_ENCODER, x.size(2) // 4, x.size(3) // 4
        ), [
            torch.randn(x.size(0), SKIP_CHANNELS[0], x.size(2), x.size(3)),
            torch.randn(
                x.size(0), SKIP_CHANNELS[1], x.size(2) // 2, x.size(3) // 2
            ),
        ]

    @property
    def out_channels(self) -> int:
        return OUT_CHANNELS_ENCODER


class MinimalValidEncoder(EncoderBase):
    """A minimal concrete encoder implementing all abstract members."""

    def __init__(self, in_channels: int = IN_CHANNELS_RGB):
        super().__init__(in_channels)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skip1 = torch.randn(x.size(0), SKIP_CHANNELS[0], x.size(2), x.size(3))
        skip2 = torch.randn(
            x.size(0), SKIP_CHANNELS[1], x.size(2) // 2, x.size(3) // 2
        )
        output = torch.randn(
            x.size(0), OUT_CHANNELS_ENCODER, x.size(2) // 4, x.size(3) // 4
        )
        return output, [skip1, skip2]

    @property
    def out_channels(self) -> int:
        return OUT_CHANNELS_ENCODER

    @property
    def skip_channels(self) -> list[int]:
        return SKIP_CHANNELS


# 3. Tests for concrete subclasses


def test_incomplete_encoder_missing_forward_raises_type_error():
    """Verify TypeError if 'forward' is not implemented."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        MinimalEncoderMissingForward(in_channels=IN_CHANNELS_RGB)  # type: ignore[abstract]


def test_incomplete_encoder_missing_out_channels_raises_type_error():
    """Verify TypeError if 'out_channels' property is not implemented."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        MinimalEncoderMissingOutChannels(in_channels=IN_CHANNELS_RGB)  # type: ignore[abstract]


def test_incomplete_encoder_missing_skip_channels_raises_type_error():
    """Verify TypeError if 'skip_channels' property is not implemented."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        MinimalEncoderMissingSkipChannels(in_channels=IN_CHANNELS_RGB)  # type: ignore[abstract]


def test_valid_minimal_encoder_instantiates():
    """Verify a valid concrete subclass can be instantiated."""
    try:
        encoder = MinimalValidEncoder(in_channels=IN_CHANNELS_RGB)
        assert isinstance(encoder, EncoderBase)
        assert encoder.out_channels == OUT_CHANNELS_ENCODER
        assert encoder.skip_channels == SKIP_CHANNELS
    except TypeError:
        pytest.fail("Valid minimal encoder failed to instantiate.")


# Optional: Test the forward pass contract (basic shape check)
def test_valid_minimal_encoder_forward_contract():
    """Check if the forward pass returns the expected types and list length."""
    encoder = MinimalValidEncoder(in_channels=IN_CHANNELS_RGB)
    dummy_input = torch.randn(2, IN_CHANNELS_RGB, 128, 128)
    try:
        output, skips = encoder.forward(dummy_input)
        assert isinstance(output, torch.Tensor)
        assert isinstance(skips, list)
        assert len(skips) == len(encoder.skip_channels)
        for i, skip_tensor in enumerate(skips):
            assert isinstance(skip_tensor, torch.Tensor)
            assert skip_tensor.shape[1] == encoder.skip_channels[i]
    except Exception as e:
        pytest.fail(f"MinimalValidEncoder forward pass failed: {e}")


# --- Tests for BottleneckBase ---


# 4. Test that BottleneckBase itself cannot be instantiated
def test_bottleneck_base_cannot_be_instantiated():
    """Verify TypeError raised when trying to instantiate BottleneckBase."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BottleneckBase(in_channels=IN_CHANNELS_BOTTLENECK)  # type: ignore[abstract]


# 5. Define minimal concrete subclasses for testing BottleneckBase


class MinimalBottleneckMissingForward(BottleneckBase):
    """A minimal bottleneck missing the 'forward' implementation."""

    def __init__(self, in_channels: int = IN_CHANNELS_BOTTLENECK):
        super().__init__(in_channels)

    @property
    def out_channels(self) -> int:
        return OUT_CHANNELS_BOTTLENECK


class MinimalBottleneckMissingOutChannels(BottleneckBase):
    """A minimal bottleneck missing the 'out_channels' property."""

    def __init__(self, in_channels: int = IN_CHANNELS_BOTTLENECK):
        super().__init__(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn(
            x.size(0), OUT_CHANNELS_BOTTLENECK, x.size(2), x.size(3)
        )


class MinimalValidBottleneck(BottleneckBase):
    """A minimal bottleneck implementing all abstract members."""

    _out_channels = OUT_CHANNELS_BOTTLENECK

    def __init__(self, in_channels: int = IN_CHANNELS_BOTTLENECK):
        super().__init__(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn(x.size(0), self._out_channels, x.size(2), x.size(3))

    @property
    def out_channels(self) -> int:
        return self._out_channels


# 6. Tests for concrete Bottleneck subclasses


def test_incomplete_bottleneck_missing_forward_raises_type_error():
    """Verify TypeError if 'forward' is not implemented."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        MinimalBottleneckMissingForward(in_channels=IN_CHANNELS_BOTTLENECK)  # type: ignore[abstract]


def test_incomplete_bottleneck_missing_out_channels_raises_type_error():
    """Verify TypeError if 'out_channels' property is not implemented."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        MinimalBottleneckMissingOutChannels(in_channels=IN_CHANNELS_BOTTLENECK)  # type: ignore[abstract]


def test_valid_minimal_bottleneck_instantiates():
    """Verify a valid concrete bottleneck subclass can be instantiated."""
    try:
        bottleneck = MinimalValidBottleneck(in_channels=IN_CHANNELS_BOTTLENECK)
        assert isinstance(bottleneck, BottleneckBase)
        assert bottleneck.out_channels == OUT_CHANNELS_BOTTLENECK
    except TypeError:
        pytest.fail("Valid minimal bottleneck failed to instantiate.")


def test_valid_minimal_bottleneck_forward_contract():
    """Check forward pass returns expected type and channels."""
    bottleneck = MinimalValidBottleneck(in_channels=IN_CHANNELS_BOTTLENECK)
    dummy_input = torch.randn(2, IN_CHANNELS_BOTTLENECK, 16, 16)
    try:
        output = bottleneck.forward(dummy_input)
        assert isinstance(output, torch.Tensor)
        assert output.shape[1] == bottleneck.out_channels
        assert output.shape[2] == dummy_input.shape[2]
        assert output.shape[3] == dummy_input.shape[3]
    except Exception as e:
        pytest.fail(f"MinimalValidBottleneck forward pass failed: {e}")


# --- Tests for DecoderBase ---


# 7. Test that DecoderBase itself cannot be instantiated
def test_decoder_base_cannot_be_instantiated():
    """Verify TypeError is raised when trying to instantiate DecoderBase."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        skip_channels_list = SKIP_CHANNELS
        DecoderBase(
            in_channels=IN_CHANNELS_DECODER, skip_channels=skip_channels_list
        )  # type: ignore[abstract]


# 8. Define minimal concrete subclasses for testing DecoderBase


class MinimalDecoderMissingForward(DecoderBase):
    """A minimal decoder missing the 'forward' implementation."""

    def __init__(
        self,
        in_channels: int = IN_CHANNELS_DECODER,
        skip_channels: list[int] = SKIP_CHANNELS,
    ):
        super().__init__(in_channels, skip_channels)

    @property
    def out_channels(self) -> int:
        return OUT_CHANNELS_DECODER


class MinimalDecoderMissingOutChannels(DecoderBase):
    """A minimal decoder missing the 'out_channels' property."""

    def __init__(
        self,
        in_channels: int = IN_CHANNELS_DECODER,
        skip_channels: list[int] = SKIP_CHANNELS,
    ):
        super().__init__(in_channels, skip_channels)

    def forward(
        self, x: torch.Tensor, skips: list[torch.Tensor]
    ) -> torch.Tensor:
        return torch.randn(
            x.size(0), OUT_CHANNELS_DECODER, x.size(2) * 4, x.size(3) * 4
        )


class MinimalValidDecoder(DecoderBase):
    """A minimal decoder implementing all abstract members."""

    _out_channels = OUT_CHANNELS_DECODER

    def __init__(
        self,
        in_channels: int = IN_CHANNELS_DECODER,
        skip_channels: list[int] = SKIP_CHANNELS,
    ):
        if not isinstance(skip_channels, list) or not all(
            isinstance(c, int) for c in skip_channels
        ):
            raise TypeError("skip_channels must be a list of integers")
        skips_to_store = list(reversed(skip_channels))
        super().__init__(in_channels, skips_to_store)

    def forward(
        self, x: torch.Tensor, skips: list[torch.Tensor]
    ) -> torch.Tensor:
        assert len(skips) == len(self.skip_channels)
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
        MinimalDecoderMissingForward(
            in_channels=IN_CHANNELS_DECODER, skip_channels=SKIP_CHANNELS
        )  # type: ignore[abstract]


def test_incomplete_decoder_missing_out_channels_raises_type_error():
    """Verify TypeError if 'out_channels' property is not implemented."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        MinimalDecoderMissingOutChannels(
            in_channels=IN_CHANNELS_DECODER, skip_channels=SKIP_CHANNELS
        )  # type: ignore[abstract]


def test_valid_minimal_decoder_instantiates():
    """Verify a valid concrete decoder subclass can be instantiated."""
    try:
        decoder = MinimalValidDecoder(
            in_channels=IN_CHANNELS_DECODER, skip_channels=SKIP_CHANNELS
        )
        assert isinstance(decoder, DecoderBase)
        assert decoder.out_channels == OUT_CHANNELS_DECODER
    except TypeError:
        pytest.fail("Valid minimal decoder failed to instantiate.")
    except AssertionError as e:
        pytest.fail(f"Assertion failed during decoder init: {e}")


def test_valid_minimal_decoder_forward_contract():
    """Check forward pass accepts skips & returns expected type/channels."""
    skip_channels_list = SKIP_CHANNELS
    decoder = MinimalValidDecoder(
        in_channels=IN_CHANNELS_DECODER, skip_channels=skip_channels_list
    )

    dummy_bottleneck_out = torch.randn(2, IN_CHANNELS_DECODER, 16, 16)
    dummy_skip1 = torch.randn(2, skip_channels_list[0], 64, 64)
    dummy_skip2 = torch.randn(2, skip_channels_list[1], 32, 32)
    dummy_skips = [dummy_skip1, dummy_skip2]

    try:
        output = decoder.forward(dummy_bottleneck_out, dummy_skips)
        assert isinstance(output, torch.Tensor)
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
            decoder=MinimalValidDecoder(),
        )  # type: ignore[abstract]


# 11. Minimal concrete subclass for UNetBase (missing forward)


class MinimalUNetMissingForward(UNetBase):
    def __init__(self, encoder, bottleneck, decoder):
        super().__init__(encoder, bottleneck, decoder)


# 12. Minimal valid UNet implementation


class MinimalValidUNet(UNetBase):
    def __init__(self, encoder, bottleneck, decoder):
        super().__init__(encoder, bottleneck, decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.encoder is None
            or self.bottleneck is None
            or self.decoder is None
        ):
            raise RuntimeError("UNet components must not be None")
        bottleneck_out, skips = self.encoder(x)
        bottleneck_out = self.bottleneck(bottleneck_out)
        return cast(torch.Tensor, self.decoder(bottleneck_out, skips))


# 13. Test that missing forward raises TypeError


def test_incomplete_unet_missing_forward_raises_type_error():
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        MinimalUNetMissingForward(
            encoder=MinimalValidEncoder(),
            bottleneck=MinimalValidBottleneck(),
            decoder=MinimalValidDecoder(),
        )  # type: ignore[abstract]


# 14. Test valid minimal UNet instantiates and forward works


def test_valid_minimal_unet_instantiates_and_forward():
    encoder = MinimalValidEncoder()
    bottleneck = MinimalValidBottleneck()
    decoder = MinimalValidDecoder()
    unet = MinimalValidUNet(encoder, bottleneck, decoder)
    assert isinstance(unet, UNetBase)
    assert unet.in_channels == encoder.in_channels
    assert unet.out_channels == decoder.out_channels
    dummy_input = torch.randn(2, IN_CHANNELS_RGB, 128, 128)
    output = unet(dummy_input)
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == 2  # noqa: PLR2004
    assert output.shape[1] == decoder.out_channels


# 15. Test component compatibility validation


def test_unet_component_compatibility_checks():
    encoder = MinimalValidEncoder()
    bottleneck = MinimalValidBottleneck()
    decoder = MinimalValidDecoder()

    class BadEncoder(MinimalValidEncoder):
        @property
        def out_channels(self):
            return 999

    with pytest.raises(ValueError, match="output channels"):
        MinimalValidUNet(BadEncoder(), bottleneck, decoder)

    class BadBottleneck(MinimalValidBottleneck):
        @property
        def out_channels(self):
            return 999

    with pytest.raises(ValueError, match="output channels"):
        MinimalValidUNet(encoder, BadBottleneck(), decoder)

    class BadDecoder(DecoderBase):
        def __init__(
            self,
            in_channels: int = IN_CHANNELS_DECODER,
            skip_channels: list[int] | None = None,
        ):
            super().__init__(in_channels, [1, 2, 3])

        def forward(
            self, x: torch.Tensor, skips: list[torch.Tensor]
        ) -> torch.Tensor:
            return torch.randn(
                x.size(0), self.out_channels, x.size(2) * 4, x.size(3) * 4
            )

        @property
        def out_channels(self) -> int:
            return 1

    with pytest.raises(ValueError, match="skip channel"):
        MinimalValidUNet(encoder, bottleneck, BadDecoder())


# --- Integration Tests ---


def test_full_unet_data_flow():
    """Test the complete data flow through a valid UNet implementation."""
    encoder = MinimalValidEncoder(in_channels=IN_CHANNELS_RGB)
    bottleneck = MinimalValidBottleneck(in_channels=encoder.out_channels)
    decoder = MinimalValidDecoder(
        in_channels=bottleneck.out_channels,
        skip_channels=encoder.skip_channels,
    )
    unet = MinimalValidUNet(encoder, bottleneck, decoder)

    x = torch.randn(2, IN_CHANNELS_RGB, 64, 64)

    try:
        output = unet(x)

        assert output.shape[0] == x.shape[0]
        assert output.shape[1] == decoder.out_channels
        assert output.shape[2:] == x.shape[2:]

    except Exception as e:
        pytest.fail(f"Full UNet forward pass failed: {e}")


def test_skip_connection_dimensions():
    """Verify skip connection dimensions are preserved through the network."""
    encoder = MinimalValidEncoder(in_channels=IN_CHANNELS_RGB)
    bottleneck = MinimalValidBottleneck(in_channels=encoder.out_channels)
    decoder = MinimalValidDecoder(
        in_channels=bottleneck.out_channels,
        skip_channels=encoder.skip_channels,
    )
    MinimalValidUNet(encoder, bottleneck, decoder)

    x = torch.randn(2, IN_CHANNELS_RGB, 64, 64)

    encoder_output, skips = encoder(x)

    assert len(skips) == len(encoder.skip_channels)
    for skip, channels in zip(skips, encoder.skip_channels, strict=False):
        assert skip.shape[1] == channels

    bottleneck_output = bottleneck(encoder_output)
    assert bottleneck_output.shape[1] == bottleneck.out_channels

    try:
        decoder_output = decoder(bottleneck_output, skips)
        assert decoder_output.shape[1] == decoder.out_channels
        assert decoder_output.shape[2:] == x.shape[2:]
    except Exception as e:
        msg = "Decoder failed to process bottleneck output with skips: {}"
        pytest.fail(msg.format(e))


def test_component_validation_error_messages():
    """Test that component validation provides clear error messages."""
    encoder = MinimalValidEncoder(in_channels=IN_CHANNELS_RGB)
    bottleneck = MinimalValidBottleneck(in_channels=encoder.out_channels)
    decoder = MinimalValidDecoder(
        in_channels=bottleneck.out_channels,
        skip_channels=encoder.skip_channels,
    )

    with pytest.raises(
        TypeError, match="encoder must be an instance of EncoderBase"
    ):
        MinimalValidUNet(None, bottleneck, decoder)

    with pytest.raises(
        TypeError, match="bottleneck must be an instance of BottleneckBase"
    ):
        MinimalValidUNet(encoder, None, decoder)

    with pytest.raises(
        (TypeError, AttributeError),
        match="(decoder must be an instance of DecoderBase|"
        + "'NoneType' object has no attribute)",
    ):
        MinimalValidUNet(encoder, bottleneck, None)

    class MismatchedBottleneck(MinimalValidBottleneck):
        def __init__(self):
            super().__init__(in_channels=32)

    err_msg = "Encoder output channels .* must match bottleneck input channels"
    with pytest.raises(ValueError, match=err_msg):
        MinimalValidUNet(encoder, MismatchedBottleneck(), decoder)

    class MismatchedDecoder(MinimalValidDecoder):
        def __init__(self):
            super().__init__(
                in_channels=64, skip_channels=encoder.skip_channels
            )

    err_msg = "Bottleneck output channels .* must match decoder input channels"
    with pytest.raises(ValueError, match=err_msg):
        MinimalValidUNet(encoder, bottleneck, MismatchedDecoder())


def test_skip_connection_compatibility():
    """Test that skip connection compatibility is properly enforced."""
    encoder = MinimalValidEncoder(in_channels=IN_CHANNELS_RGB)
    bottleneck = MinimalValidBottleneck(in_channels=encoder.out_channels)

    class IncompatibleDecoder(DecoderBase):
        def __init__(self):
            in_channels = bottleneck.out_channels
            skip_channels = [8, 16]
            super().__init__(in_channels, skip_channels)
            self._out_channels = 1

        def forward(self, x, skips):
            return torch.zeros(1, 1, 64, 64)

        @property
        def out_channels(self):
            return self._out_channels

    incompatible_decoder = IncompatibleDecoder()

    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        MinimalValidUNet(encoder, bottleneck, incompatible_decoder)

        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "Encoder skip channels" in str(w[0].message)
        assert "don't match reversed decoder skip channels" in str(
            w[0].message
        )
