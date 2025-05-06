# tests/model/architectures/test_swinv2_cnn_aspp_unet.py

import pytest
import torch
from omegaconf import OmegaConf
import hydra
import torch.nn as nn

# Assuming the main run script or a utility initializes Hydra context
# For isolated testing, we might need to load config manually

# Import the class to be tested
from src.model.architectures.swinv2_cnn_aspp_unet import SwinV2CnnAsppUNet
from src.model.base import UNetBase

# Default configuration path (relative to project root)
DEFAULT_CONFIG_PATH = "configs/model/swinv2_hybrid.yaml"


@pytest.fixture
def default_config() -> OmegaConf:
    """Loads the default configuration for the hybrid model."""
    return OmegaConf.load(DEFAULT_CONFIG_PATH)


def test_instantiation_with_default_config(default_config):
    """
    Tests if SwinV2CnnAsppUNet can be instantiated with the default config.
    """
    try:
        # Use hydra.utils.instantiate, similar to the factory
        # This requires the _target_ key in the config
        model = hydra.utils.instantiate(default_config)
        assert isinstance(model, SwinV2CnnAsppUNet)
        assert isinstance(model, UNetBase)
        # Check basic properties derived from config
        assert model.in_channels == default_config.encoder_cfg.in_channels
        assert model.out_channels == default_config.num_classes
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'bottleneck')
        assert hasattr(model, 'decoder')
        assert hasattr(model, 'final_activation_layer')
    except Exception as e:
        pytest.fail(f"Failed to instantiate SwinV2CnnAsppUNet from default \
config: {e}")


def test_forward_pass_default_config(default_config):
    """
    Tests the forward pass with default config and dummy data.
    Checks output shape.
    """
    # Use default config directly (which includes handle_input_size='resize')
    test_cfg = default_config
    img_size = test_cfg.encoder_cfg.img_size  # Use configured img_size

    batch_size = 2
    in_channels = test_cfg.encoder_cfg.in_channels
    num_classes = test_cfg.num_classes

    dummy_input = torch.randn(batch_size, in_channels, img_size, img_size)

    try:
        model = hydra.utils.instantiate(test_cfg)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == batch_size
        assert output.shape[1] == num_classes
        assert output.shape[2] == img_size, \
            f"Output H={output.shape[2]} != Input H={img_size}"
        assert output.shape[3] == img_size, \
            f"Output W={output.shape[3]} != Input W={img_size}"

    except Exception as e:
        pytest.fail(f"Forward pass failed for default config: {e}")


def test_instantiation_and_forward_with_cbam(default_config):
    """
    Tests instantiation and forward pass when CBAM is enabled in the decoder.
    Uses default input size handling.
    """
    # Modify the default config to enable CBAM
    test_cfg = default_config.copy()
    test_cfg.decoder_cfg.use_cbam = True
    # Keep default handle_input_size = 'resize'
    img_size = test_cfg.encoder_cfg.img_size  # Use configured img_size

    batch_size = 2
    in_channels = test_cfg.encoder_cfg.in_channels
    num_classes = test_cfg.num_classes

    dummy_input = torch.randn(batch_size, in_channels, img_size, img_size)

    try:
        model = hydra.utils.instantiate(test_cfg)
        cnn_decoder = None
        for module in model.modules():
            if isinstance(module, model.decoder.__class__):
                cnn_decoder = module
                break
        assert cnn_decoder is not None, "CNNDecoder not found"
        first_decoder_block = cnn_decoder.decoder_blocks[0]
        assert hasattr(first_decoder_block, 'cbam')
        assert not isinstance(first_decoder_block.cbam, nn.Identity)

        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, num_classes, img_size, img_size)

    except Exception as e:
        pytest.fail(f"Instantiation or forward pass failed with CBAM: {e}")


def test_backward_and_gradients(default_config):
    """
    Verifies that the model supports backward pass and parameters receive
    gradients.
    """
    test_cfg = default_config
    img_size = test_cfg.encoder_cfg.img_size
    batch_size = 2
    in_channels = test_cfg.encoder_cfg.in_channels
    dummy_input = torch.randn(batch_size, in_channels, img_size, img_size,
                              requires_grad=True)
    model = hydra.utils.instantiate(test_cfg)
    model.train()
    output = model(dummy_input)
    # Dummy loss: sum of outputs
    loss = output.sum()
    loss.backward()
    # Check that at least one parameter received gradient
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads), (
        "No gradients flowed through the model."
    )


def test_skip_connection_shapes_and_channels(default_config):
    """
    Checks that skip connections from encoder have expected shapes and
    channels.
    """
    test_cfg = default_config
    img_size = test_cfg.encoder_cfg.img_size
    batch_size = 2
    in_channels = test_cfg.encoder_cfg.in_channels
    model = hydra.utils.instantiate(test_cfg)
    model.eval()
    dummy_input = torch.randn(batch_size, in_channels, img_size, img_size)
    # Accede al encoder y ejecuta forward manual para obtener skips
    encoder = model.encoder
    with torch.no_grad():
        bottleneck, skips = encoder(dummy_input)
    # Verifica número de skips y shapes
    expected_skips = encoder.skip_channels
    assert len(skips) == len(expected_skips), \
        f"Expected {len(expected_skips)} skips, got {len(skips)}"
    for i, (skip, ch) in enumerate(zip(skips, expected_skips)):
        assert skip.shape[0] == batch_size
        assert skip.shape[1] == ch, (
            f"Skip {i} has {skip.shape[1]} channels, expected {ch}"
        )
        assert skip.shape[2] > 1 and skip.shape[3] > 1, (
            f"Skip {i} has invalid spatial dims: {skip.shape}"
        )


def test_advanced_configuration_variant(default_config):
    """
    Tests the model with an advanced configuration (different SwinV2 model,
    ASPP rates, decoder depth).
    """
    # Copia y modifica la config
    test_cfg = default_config.copy()
    test_cfg.encoder_cfg.model_name = "swinv2_small_window16_256"
    test_cfg.encoder_cfg.img_size = 256
    test_cfg.bottleneck_cfg.dilation_rates = [1, 12, 24]
    test_cfg.decoder_cfg.kernel_size = 5
    test_cfg.decoder_cfg.upsample_mode = "nearest"
    img_size = test_cfg.encoder_cfg.img_size
    batch_size = 1
    in_channels = test_cfg.encoder_cfg.in_channels
    num_classes = test_cfg.num_classes
    dummy_input = torch.randn(batch_size, in_channels, img_size, img_size)
    try:
        model = hydra.utils.instantiate(test_cfg)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (batch_size, num_classes, img_size, img_size)
    except Exception as e:
        pytest.fail(f"Advanced config variant failed: {e}")


def test_minimal_configuration_instantiation():
    """
    Tests instantiation and forward with a minimal configuration (only
    required fields).
    """
    # Construye una config mínima a mano
    minimal_cfg = OmegaConf.create({
        '_target_':
        'src.model.architectures.swinv2_cnn_aspp_unet.SwinV2CnnAsppUNet',
        'encoder_cfg': {
            'in_channels': 3,
            'model_name': 'swinv2_tiny_window16_256',
            'img_size': 256,
            'pretrained': False,
        },
        'bottleneck_cfg': {
            'output_channels': 64,
            'dilation_rates': [1, 6, 12],
        },
        'decoder_cfg': {
            'kernel_size': 3,
        },
        'num_classes': 2,
        'final_activation': None,
    })
    img_size = minimal_cfg.encoder_cfg.img_size
    batch_size = 1
    in_channels = minimal_cfg.encoder_cfg.in_channels
    # num_classes = minimal_cfg.num_classes  # No se usa, elimino
    dummy_input = torch.randn(batch_size, in_channels, img_size, img_size)
    try:
        model = hydra.utils.instantiate(minimal_cfg)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (
            batch_size,
            minimal_cfg.num_classes,
            img_size,
            img_size,
        )
    except Exception as e:
        pytest.fail(
            f"Minimal config instantiation failed: {e}"
        )

# TODO:
# - Add tests for different configurations (e.g., different model_name)
# - Add tests checking skip connection shapes/channels (more involved)
# - Add tests for gradient flow (requires training mode and loss)
