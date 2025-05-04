import pytest
import torch
import torch.nn as nn
import os
import hydra
import logging

# Import the component to be tested
from src.model.architectures.cnn_convlstm_unet import CNNEncoder
from src.model.base import EncoderBase

# Import BottleneckBase and the new bottleneck implementation
from src.model.base import BottleneckBase
from src.model.architectures.cnn_convlstm_unet import ConvLSTMBottleneck

# Import DecoderBase and the new decoder implementation
from src.model.base import DecoderBase
from src.model.architectures.cnn_convlstm_unet import CNNDecoder

# Import CNNConvLSTMUNet and the new UNetBase
from src.model.architectures.cnn_convlstm_unet import CNNConvLSTMUNet
from src.model.base import UNetBase

log = logging.getLogger(__name__)


@pytest.fixture
def encoder_params():
    """Provides common parameters for CNNEncoder tests."""
    return {
        "in_channels": 3,
        "base_filters": 16,  # Use fewer filters for faster testing
        "depth": 4,          # Use shallower depth for faster testing
        "kernel_size": 3,
        "pool_size": 2,
        "use_batchnorm": True,  # This param is not used by CNNEncoder itself
        "batch_size": 2,
        "height": 64,        # Example input height
        "width": 64,         # Example input width
    }


def test_cnn_encoder_init(encoder_params):
    """Tests CNNEncoder initialization."""
    encoder = CNNEncoder(
        in_channels=encoder_params["in_channels"],
        base_filters=encoder_params["base_filters"],
        depth=encoder_params["depth"],
        kernel_size=encoder_params["kernel_size"],
        pool_size=encoder_params["pool_size"],
    )
    assert isinstance(encoder, EncoderBase)
    assert len(encoder.encoder_blocks) == encoder_params["depth"]

    # Check channel progression
    current_ch = encoder_params["in_channels"]
    expected_skip_channels = []
    for i, block in enumerate(encoder.encoder_blocks):
        expected_out_ch = encoder_params["base_filters"] * (2**i)
        assert block.conv1.in_channels == current_ch
        assert block.conv1.out_channels == expected_out_ch
        assert block.conv2.in_channels == expected_out_ch
        assert block.conv2.out_channels == expected_out_ch
        assert block.out_channels == expected_out_ch  # Check property
        assert block.skip_channels == [expected_out_ch]
        expected_skip_channels.append(expected_out_ch)
        current_ch = expected_out_ch

    # Check overall properties
    assert encoder.skip_channels == expected_skip_channels
    assert encoder.out_channels == current_ch  # Final output channels


def test_cnn_encoder_forward_shapes(encoder_params):
    """Tests the output shapes of the CNNEncoder forward pass."""
    encoder = CNNEncoder(
        in_channels=encoder_params["in_channels"],
        base_filters=encoder_params["base_filters"],
        depth=encoder_params["depth"],
        pool_size=encoder_params["pool_size"],
    )
    input_tensor = torch.randn(
        encoder_params["batch_size"],
        encoder_params["in_channels"],
        encoder_params["height"],
        encoder_params["width"],
    )

    output, skips = encoder(input_tensor)

    # Check final output shape
    pool_factor = encoder_params["pool_size"] ** encoder_params["depth"]
    expected_final_h = encoder_params["height"] // pool_factor
    expected_final_w = encoder_params["width"] // pool_factor
    expected_final_c = encoder.out_channels
    assert output.shape == (
        encoder_params["batch_size"],
        expected_final_c,
        expected_final_h,
        expected_final_w,
    )

    # Check skip connection shapes
    assert len(skips) == encoder_params["depth"]
    assert len(encoder.skip_channels) == encoder_params["depth"]
    for i in range(encoder_params["depth"]):
        pool_factor_i = encoder_params["pool_size"] ** i
        expected_skip_h = encoder_params["height"] // pool_factor_i
        expected_skip_w = encoder_params["width"] // pool_factor_i
        expected_skip_c = encoder.skip_channels[i]
        assert skips[i].shape == (
            encoder_params["batch_size"],
            expected_skip_c,
            expected_skip_h,
            expected_skip_w,
        )


def test_cnn_encoder_invalid_depth(encoder_params):
    """Tests that initializing with depth < 1 raises ValueError."""
    with pytest.raises(ValueError, match="Encoder depth must be at least 1."):
        CNNEncoder(
            in_channels=encoder_params["in_channels"],
            base_filters=encoder_params["base_filters"],
            depth=0,  # Invalid depth
        )


# --- Tests for ConvLSTMBottleneck ---

@pytest.fixture
def bottleneck_params(encoder_params):
    """Provides common parameters for ConvLSTMBottleneck tests."""
    # Calculate expected input channels from the encoder fixture
    encoder_depth = encoder_params["depth"]
    base_filters = encoder_params["base_filters"]
    in_channels_bottleneck = base_filters * (2**(encoder_depth - 1))

    return {
        "in_channels": in_channels_bottleneck,
        "hidden_dim": in_channels_bottleneck * 2,  # Example: double channels
        "kernel_size": (3, 3),
        "num_layers": 1,  # Test with single layer first
        "bias": True,
        "batch_size": encoder_params["batch_size"],
        # Calculate H/W after encoder pooling
        "height": (
            encoder_params["height"] //
            (encoder_params["pool_size"] ** encoder_depth)
        ),
        "width": (
            encoder_params["width"] //
            (encoder_params["pool_size"] ** encoder_depth)
        ),
    }


def test_convlstm_bottleneck_init(bottleneck_params):
    """Tests ConvLSTMBottleneck initialization."""
    bottleneck = ConvLSTMBottleneck(
        in_channels=bottleneck_params["in_channels"],
        hidden_dim=bottleneck_params["hidden_dim"],
        kernel_size=bottleneck_params["kernel_size"],
        num_layers=bottleneck_params["num_layers"],
        bias=bottleneck_params["bias"],
    )
    assert isinstance(bottleneck, BottleneckBase)
    assert bottleneck.in_channels == bottleneck_params["in_channels"]
    assert bottleneck.out_channels == bottleneck_params["hidden_dim"]
    assert bottleneck.bottleneck_convlstm.num_layers == bottleneck_params[
        "num_layers"]


@pytest.mark.parametrize("num_layers", [1, 2])
def test_convlstm_bottleneck_forward_shape(bottleneck_params, num_layers):
    """Tests the output shape of the ConvLSTMBottleneck forward pass."""
    hidden_dim = bottleneck_params["hidden_dim"]
    # Use list if multiple layers
    if num_layers > 1:
        # Create a list of hidden dimensions for multi-layer setup
        hidden_dim_list = [hidden_dim] * num_layers
    else:
        hidden_dim_list = hidden_dim

    bottleneck = ConvLSTMBottleneck(
        in_channels=bottleneck_params["in_channels"],
        hidden_dim=hidden_dim_list,
        kernel_size=bottleneck_params["kernel_size"],
        num_layers=num_layers,
        bias=bottleneck_params["bias"],
    )

    input_tensor = torch.randn(
        bottleneck_params["batch_size"],
        bottleneck_params["in_channels"],
        bottleneck_params["height"],
        bottleneck_params["width"],
    )

    output = bottleneck(input_tensor)

    expected_out_channels = bottleneck.out_channels
    assert output.shape == (
        bottleneck_params["batch_size"],
        expected_out_channels,
        bottleneck_params["height"],
        bottleneck_params["width"],
    )


def test_convlstm_bottleneck_init_invalid_hidden(bottleneck_params):
    """Tests that init fails with empty hidden_dim list."""
    with pytest.raises(ValueError, match="hidden_dim list cannot be empty"):
        ConvLSTMBottleneck(
            in_channels=bottleneck_params["in_channels"],
            hidden_dim=[],  # Empty list
            kernel_size=bottleneck_params["kernel_size"],
            num_layers=2,  # Must match len(hidden_dim) if list
            bias=bottleneck_params["bias"],
        )


# --- Tests for CNNDecoder ---

@pytest.fixture
def decoder_params(encoder_params, bottleneck_params):
    """Provides common parameters for CNNDecoder tests."""
    # Skip channels from encoder (high-res to low-res)
    encoder_depth = encoder_params["depth"]
    base_filters = encoder_params["base_filters"]
    skip_channels_list = [base_filters * (2**i) for i in range(encoder_depth)]

    # Get the bottleneck output channels (which is the last hidden_dim)
    bottleneck_out_channels = bottleneck_params["hidden_dim"]
    if isinstance(bottleneck_out_channels, list):
        bottleneck_out_channels = bottleneck_out_channels[-1]

    return {
        "in_channels": bottleneck_out_channels,  # Use derived value
        "skip_channels_list": skip_channels_list,
        "out_channels": 1,  # Example: binary segmentation
        "depth": encoder_depth,
        "kernel_size": 3,
        "upsample_mode": 'bilinear',
        "batch_size": encoder_params["batch_size"],
        # Input H/W to decoder is bottleneck H/W
        "height": bottleneck_params["height"],
        "width": bottleneck_params["width"],
        # Final expected H/W is original input H/W
        "final_height": encoder_params["height"],
        "final_width": encoder_params["width"],
    }


def test_cnn_decoder_init(decoder_params):
    """Tests CNNDecoder initialization."""
    decoder = CNNDecoder(
        in_channels=decoder_params["in_channels"],
        skip_channels_list=decoder_params["skip_channels_list"],
        out_channels=decoder_params["out_channels"],
        depth=decoder_params["depth"],
        kernel_size=decoder_params["kernel_size"],
        upsample_mode=decoder_params["upsample_mode"],
    )
    assert isinstance(decoder, DecoderBase)
    assert len(decoder.decoder_blocks) == decoder_params["depth"]
    assert decoder.out_channels == decoder_params["out_channels"]
    # Check internal skip channel order (low-res to high-res)
    assert decoder.skip_channels == list(reversed(
        decoder_params["skip_channels_list"]))


def test_cnn_decoder_forward_shape(decoder_params, encoder_params):
    """Tests the output shape of the CNNDecoder forward pass."""
    decoder = CNNDecoder(
        in_channels=decoder_params["in_channels"],
        skip_channels_list=decoder_params["skip_channels_list"],
        out_channels=decoder_params["out_channels"],
        depth=decoder_params["depth"],
    )

    # Mock bottleneck output tensor
    bottleneck_output = torch.randn(
        decoder_params["batch_size"],
        decoder_params["in_channels"],
        decoder_params["height"],
        decoder_params["width"],
    )

    # Mock skip connection tensors (high-res to low-res)
    skips = []
    final_h, final_w = decoder_params["final_height"], decoder_params[
        "final_width"]
    pool_factor = encoder_params["pool_size"]
    for i in range(decoder_params["depth"]):
        skip_c = decoder_params["skip_channels_list"][i]
        # Calculate expected H/W for this skip level
        skip_h = final_h // (pool_factor ** i)
        skip_w = final_w // (pool_factor ** i)
        # Create mock skip tensor
        skips.append(
            torch.randn(
                decoder_params["batch_size"], skip_c, skip_h, skip_w
            )
        )

    output = decoder(bottleneck_output, skips)

    assert output.shape == (
        decoder_params["batch_size"],
        decoder_params["out_channels"],
        decoder_params["final_height"],
        decoder_params["final_width"],
    )


def test_cnn_decoder_init_mismatch_depth(decoder_params):
    """Tests init failure when skip_channels_list length != depth."""
    with pytest.raises(ValueError, match="must match decoder depth"):
        CNNDecoder(
            in_channels=decoder_params["in_channels"],
            # Pass a list with incorrect length
            skip_channels_list=decoder_params["skip_channels_list"][:-1],
            out_channels=decoder_params["out_channels"],
            depth=decoder_params["depth"],
        )


def test_cnn_decoder_forward_mismatch_skips(decoder_params):
    """Tests forward failure when number of skips != depth."""
    decoder = CNNDecoder(
        in_channels=decoder_params["in_channels"],
        skip_channels_list=decoder_params["skip_channels_list"],
        depth=decoder_params["depth"],
    )
    bottleneck_output = torch.randn(
        decoder_params["batch_size"],
        decoder_params["in_channels"],
        decoder_params["height"],
        decoder_params["width"],
    )
    # Create incorrect number of skips
    skips_wrong_num = [
        torch.randn(1, 1, 1, 1)
    ] * (decoder_params["depth"] - 1)

    # Adjust regex to match the actual error message more closely
    expected_error_msg = (
        r"Number of skips \(\d+\) must match decoder depth \(\d+\)"
    )
    with pytest.raises(ValueError, match=expected_error_msg):
        decoder(bottleneck_output, skips_wrong_num)


# --- Tests for CNNConvLSTMUNet Assembly ---

@pytest.fixture
def assembled_unet(encoder_params, bottleneck_params, decoder_params):
    """Provides an assembled CNNConvLSTMUNet for testing."""
    # Use smaller depth/filters for faster assembly testing if needed
    # but ensure consistency across params

    encoder = CNNEncoder(
        in_channels=encoder_params["in_channels"],
        base_filters=encoder_params["base_filters"],
        depth=encoder_params["depth"],
        kernel_size=encoder_params["kernel_size"],
        pool_size=encoder_params["pool_size"],
    )

    # Derive bottleneck params based on encoder output
    bottleneck_in_channels = encoder.out_channels
    # Ensure hidden dim is appropriate, can reuse from decoder_params if
    # consistent
    bottleneck_hidden_dim = decoder_params["in_channels"]  # Input to decoder

    bottleneck = ConvLSTMBottleneck(
        in_channels=bottleneck_in_channels,
        hidden_dim=bottleneck_hidden_dim,
        # Use bottleneck fixture kernel
        kernel_size=bottleneck_params["kernel_size"],
        # Use bottleneck fixture layers
        num_layers=bottleneck_params["num_layers"],
        bias=bottleneck_params["bias"],
    )

    # Derive decoder params based on bottleneck and encoder
    decoder_in_channels = bottleneck.out_channels
    encoder_skips = getattr(encoder, 'skip_channels', [])
    if not isinstance(encoder_skips, list):
        encoder_skips = []
    # Don't reverse the list - CNNDecoder expects high-to-low resolution
    # and will do the reversing internally
    decoder_skip_channels_list = encoder_skips

    # Create the decoder using direct constructor
    decoder = CNNDecoder(
        in_channels=decoder_in_channels,
        skip_channels_list=decoder_skip_channels_list,
        out_channels=decoder_params["out_channels"],
        depth=encoder_params["depth"],
        kernel_size=decoder_params["kernel_size"],
        upsample_mode=decoder_params["upsample_mode"]
    )

    # Assemble the UNet
    unet = CNNConvLSTMUNet(encoder=encoder, bottleneck=bottleneck,
                           decoder=decoder)
    return unet


def test_cnn_convlstm_unet_init(assembled_unet):
    """Tests the initialization and component validation of CNNConvLSTMUNet."""
    assert isinstance(assembled_unet, UNetBase)
    assert isinstance(assembled_unet.encoder, CNNEncoder)
    assert isinstance(assembled_unet.bottleneck, ConvLSTMBottleneck)
    assert isinstance(assembled_unet.decoder, CNNDecoder)
    # Base class validation should have passed if we got here


def test_cnn_convlstm_unet_forward_shape(assembled_unet, encoder_params,
                                         decoder_params):
    """Tests the end-to-end forward pass shape of the assembled UNet."""
    input_tensor = torch.randn(
        encoder_params["batch_size"],
        encoder_params["in_channels"],
        encoder_params["height"],  # Original input height
        encoder_params["width"],  # Original input width
    )

    output = assembled_unet(input_tensor)

    assert output.shape == (
        encoder_params["batch_size"],
        decoder_params["out_channels"],  # Final output channels
        encoder_params["height"],        # Should match original input height
        encoder_params["width"],         # Should match original input width
    )


def test_cnn_convlstm_unet_init_type_mismatch():
    """Tests that initialization fails with incorrect component types."""
    # Create dummy components of wrong types (using basic nn.Module)
    dummy_encoder = nn.Module()
    dummy_bottleneck = nn.Module()
    dummy_decoder = nn.Module()

    # Mock necessary attributes for base class validation to pass temporarily
    # This focuses the test on the explicit type checks
    dummy_encoder.out_channels = 128
    dummy_encoder.skip_channels = [16, 32, 64]
    dummy_bottleneck.in_channels = 128
    dummy_bottleneck.out_channels = 256
    dummy_decoder.skip_channels = [64, 32, 16]  # Reversed
    dummy_decoder.in_channels = 256             # Match dummy bottleneck

    with pytest.raises(TypeError, match="Expected CNNEncoder"):
        CNNConvLSTMUNet(encoder=dummy_encoder, bottleneck=dummy_bottleneck,
                        decoder=dummy_decoder)

    # Need valid encoder for next check
    valid_encoder = CNNEncoder(in_channels=3, base_filters=16, depth=3)
    # Match encoder output
    dummy_bottleneck.in_channels = valid_encoder.out_channels

    with pytest.raises(TypeError, match="Expected ConvLSTMBottleneck"):
        CNNConvLSTMUNet(encoder=valid_encoder, bottleneck=dummy_bottleneck,
                        decoder=dummy_decoder)

    # Need valid bottleneck for next check
    valid_bottleneck = ConvLSTMBottleneck(
        in_channels=valid_encoder.out_channels,
        hidden_dim=256,
        kernel_size=(3, 3)
    )
    # Match bottleneck
    dummy_decoder.in_channels = valid_bottleneck.out_channels
    # Match skip channels (reversed)
    skips = valid_encoder.skip_channels
    dummy_decoder.skip_channels = list(reversed(skips))

    with pytest.raises(TypeError, match="Expected CNNDecoder"):
        CNNConvLSTMUNet(encoder=valid_encoder, bottleneck=valid_bottleneck,
                        decoder=dummy_decoder)


# --- Test Hydra Instantiation ---

# Mark this test to use hydra
@pytest.mark.hydra
def test_instantiate_cnn_convlstm_unet_with_hydra(hydra_config_dir):
    """Tests instantiation of the full model using Hydra config."""
    # Assuming hydra_config_dir is a fixture providing path to configs
    # Update path to reflect the moved YAML file location
    config_path = os.path.join(hydra_config_dir, "model")

    # Initialize Hydra
    # Use context manager for cleaner state management if possible,
    # otherwise use clear() in finally block.
    try:
        hydra.initialize_config_dir(
            config_dir=config_path, job_name="test_unet_instantiation"
        )
    except hydra.errors.HydraAlreadyInitializedError:
        # Already initialized in a previous test, clear first
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize_config_dir(
            config_dir=config_path, job_name="test_unet_instantiation"
        )

    # Compose config, overriding only essential parts for the test
    # The factory and UNetBase validation handle channel matching
    encoder_depth = 4  # Example depth for test
    bottleneck_hidden_dim = 256  # Example hidden dim for test

    cfg = hydra.compose(config_name="cnn_convlstm_unet", overrides=[
        # Set encoder depth for this test instance
        f"encoder.depth={encoder_depth}",
        # Set bottleneck hidden dimension for this test instance
        f"bottleneck.hidden_dim={bottleneck_hidden_dim}",
        # Added: Explicitly override kernel_size to a list of integers
        "bottleneck.kernel_size=[3,3]",
        # Decoder depth will be passed directly in instantiate call
    ])

    # Print resolved config for debugging
    from omegaconf import OmegaConf
    print("\nBottleneck config:")
    print(OmegaConf.to_yaml(cfg.bottleneck))
    print(f"kernel_size type: {type(cfg.bottleneck.kernel_size)}")
    print(f"kernel_size value: {cfg.bottleneck.kernel_size}")

    try:
        # Instantiate components individually
        log.info("Instantiating encoder...")
        encoder = hydra.utils.instantiate(cfg.encoder)
        log.info(f"Encoder instantiated: {type(encoder)}")

        bottleneck_in_channels = encoder.out_channels
        log.info(f"Instantiating bottleneck with in_channels=\
{bottleneck_in_channels}...")
        # Pass derived channels as explicit overrides to instantiate
        bottleneck = hydra.utils.instantiate(
            cfg.bottleneck, in_channels=bottleneck_in_channels
        )
        log.info(f"Bottleneck instantiated: {type(bottleneck)}")

        decoder_in_channels = bottleneck.out_channels
        encoder_skips = getattr(encoder, 'skip_channels', [])
        if not isinstance(encoder_skips, list):
            encoder_skips = []
        # Don't reverse the list - CNNDecoder expects high-to-low resolution
        # and will do the reversing internally
        decoder_skip_channels_list = encoder_skips
        log.info(
            f"Instantiating decoder with in_channels={decoder_in_channels}, "
            f"skip_channels_list={decoder_skip_channels_list}..."
        )
        decoder = hydra.utils.instantiate(
            cfg.decoder,
            in_channels=decoder_in_channels,
            skip_channels_list=decoder_skip_channels_list,
            depth=encoder_depth  # Explicitly pass encoder_depth here
        )
        log.info(f"Decoder instantiated: {type(decoder)}")

        # Instantiate the final U-Net model with the components
        log.info("Instantiating CNNConvLSTMUNet...")
        model = CNNConvLSTMUNet(
            encoder=encoder, bottleneck=bottleneck, decoder=decoder
        )
        log.info(f"Model instantiated: {type(model)}")

        assert isinstance(model, UNetBase)
        assert isinstance(model.encoder, CNNEncoder)
        assert isinstance(model.bottleneck, ConvLSTMBottleneck)
        assert isinstance(model.decoder, CNNDecoder)

        # Verify some basic properties post-instantiation
        # Verify encoder, bottleneck, and decoder connectivity
        assert model.bottleneck.in_channels == model.encoder.out_channels
        assert model.decoder.in_channels == model.bottleneck.out_channels
        # Skip channels test
        decoder_skips_reversed = list(reversed(model.decoder.skip_channels))
        assert model.encoder.skip_channels == decoder_skips_reversed
        # Verify output channels
        assert model.out_channels == cfg.decoder.out_channels

    except Exception as e:
        pytest.fail(f"Hydra instantiation failed: {e}")
    finally:
        # Clean up Hydra global state
        hydra.core.global_hydra.GlobalHydra.instance().clear()

# Need to add hydra_config_dir fixture, e.g., in conftest.py
# Example conftest.py content:
# import pytest
# import os
#
# @pytest.fixture(scope='session')
# def hydra_config_dir():
#     # Adjust path relative to your conftest.py location
#     return os.path.abspath(os.path.join(
#         os.path.dirname(__file__),
#         "..", # Assuming tests/ is one level down
#         "configs"
#     ))

# --- Tests for Real-world Usage Scenarios ---


def test_cnn_convlstm_unet_with_realistic_data(assembled_unet, encoder_params,
                                               decoder_params):
    """Tests the CNN-ConvLSTM U-Net architecture with more realistic data."""
    # Use a larger batch size to simulate a real-world scenario
    batch_size = 4

    # Create input tensor with RGB channels and larger dimensions
    input_tensor = torch.randn(
        batch_size,
        encoder_params["in_channels"],  # Usually 3 for RGB
        128,  # Larger height
        128,  # Larger width
    )

    # Forward pass
    output = assembled_unet(input_tensor)

    # Verify output dimensions match input dimensions (except for channels)
    assert output.shape == (
        batch_size,
        decoder_params["out_channels"],  # Output channels (classes)
        128,  # Should match input height
        128,  # Should match input width
    )

    # Verify values are in reasonable range for segmentation
    if decoder_params["out_channels"] == 1:  # Binary segmentation
        # Without activation, values can be any real number
        # With sigmoid, values would be in [0, 1]
        # Check they're not NaN or infinite
        assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "base_filters,depth,in_channels,out_channels",
    [
        (16, 3, 3, 1),    # Pequeña: RGB a binario, pocas capas
        (32, 4, 1, 2),    # Mediana: monocanal a 2 clases
        (64, 2, 3, 3),    # Baja profundidad: RGB a RGB
    ]
)
def test_cnn_convlstm_unet_configurations(base_filters, depth,
                                          in_channels, out_channels):
    """Tests CNN-ConvLSTM U-Net with different configurations."""
    # Pequeño tamaño de entrada para prueba rápida
    input_height, input_width = 64, 64

    # Crear encoder
    encoder = CNNEncoder(
        in_channels=in_channels,
        base_filters=base_filters,
        depth=depth,
    )

    # Crear bottleneck
    bottleneck = ConvLSTMBottleneck(
        in_channels=encoder.out_channels,
        hidden_dim=encoder.out_channels * 2,
        kernel_size=(3, 3),
        num_layers=1,
    )

    # Crear decoder
    decoder = CNNDecoder(
        in_channels=bottleneck.out_channels,
        skip_channels_list=encoder.skip_channels,
        out_channels=out_channels,
        depth=depth,
    )

    # Ensamblar U-Net
    unet = CNNConvLSTMUNet(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder
    )

    # Prueba de pase hacia adelante
    x = torch.randn(2, in_channels, input_height, input_width)
    output = unet(x)

    # Verificar forma de salida
    assert output.shape == (2, out_channels, input_height, input_width)

    # Verificar valores finitos
    assert torch.isfinite(output).all()
