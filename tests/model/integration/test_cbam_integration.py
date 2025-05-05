import torch
from src.model.unet import BaseUNet
from src.model.factory import insert_cbam_if_enabled
from src.model.architectures.cnn_convlstm_unet import (
    CNNConvLSTMUNet, CNNEncoder, ConvLSTMBottleneck, CNNDecoder
)


def test_cbam_integration_unet_forward():
    """Test CBAM integration in UNet forward pass (out_channels > 1)."""
    in_channels = 4
    h, w = 32, 32
    x = torch.randn(2, in_channels, h, w)
    encoder = CNNEncoder(in_channels=in_channels, base_filters=8, depth=2)
    bottleneck = torch.nn.Identity()
    decoder = CNNDecoder(
        in_channels=8, skip_channels_list=[8, 16], out_channels=2, depth=2
    )
    decoder = insert_cbam_if_enabled(
        decoder,
        {
            "cbam_enabled": True,
            "cbam_params": {"reduction": 1, "kernel_size": 3},
        },
    )
    model = BaseUNet(encoder=encoder, bottleneck=bottleneck, decoder=decoder)
    out = model(x)
    assert out.shape[0] == 2
    assert out.shape[1] == 2
    assert out.shape[2:] == (h, w)


def test_cbam_integration_cnn_convlstm_unet_forward():
    """Test CBAM integration in CNNConvLSTMUNet forward pass
    (out_channels > 1)."""
    in_channels = 3
    h, w = 32, 32
    x = torch.randn(1, in_channels, h, w)
    encoder = CNNEncoder(in_channels=in_channels, base_filters=4, depth=2)
    bottleneck = ConvLSTMBottleneck(
        in_channels=16, hidden_dim=8, kernel_size=(3, 3), num_layers=1
    )
    decoder = CNNDecoder(
        in_channels=8, skip_channels_list=[4, 8], out_channels=2, depth=2
    )
    decoder = insert_cbam_if_enabled(
        decoder,
        {
            "cbam_enabled": True,
            "cbam_params": {"reduction": 1, "kernel_size": 3},
        },
    )
    model = CNNConvLSTMUNet(
        encoder=encoder, bottleneck=bottleneck, decoder=decoder
    )
    out = model(x)
    assert out.shape[0] == 1
    assert out.shape[1] == 2
    assert out.shape[2:] == (h, w)


def test_cbam_save_and_load(tmp_path):
    """Test saving and loading a model with CBAM (out_channels > 1)."""
    encoder = CNNEncoder(in_channels=2, base_filters=4, depth=2)
    bottleneck = torch.nn.Identity()
    decoder = CNNDecoder(
        in_channels=4, skip_channels_list=[4, 8], out_channels=2, depth=2
    )
    decoder = insert_cbam_if_enabled(
        decoder,
        {
            "cbam_enabled": True,
            "cbam_params": {"reduction": 1, "kernel_size": 3},
        },
    )
    model = BaseUNet(encoder=encoder, bottleneck=bottleneck, decoder=decoder)
    path = tmp_path / "cbam_model.pt"
    torch.save(model.state_dict(), path)
    model2 = BaseUNet(encoder=encoder, bottleneck=bottleneck, decoder=decoder)
    model2.load_state_dict(torch.load(path))
    x = torch.randn(1, 2, 16, 16)
    out = model2(x)
    assert out.shape == (1, 2, 16, 16)


def test_cbam_grad_integration():
    """Test gradient flow through UNet with CBAM (out_channels > 1)."""
    encoder = CNNEncoder(in_channels=2, base_filters=4, depth=2)
    bottleneck = torch.nn.Identity()
    decoder = CNNDecoder(
        in_channels=4, skip_channels_list=[4, 8], out_channels=2, depth=2
    )
    decoder = insert_cbam_if_enabled(
        decoder,
        {
            "cbam_enabled": True,
            "cbam_params": {"reduction": 1, "kernel_size": 3},
        },
    )
    model = BaseUNet(encoder=encoder, bottleneck=bottleneck, decoder=decoder)
    x = torch.randn(2, 2, 8, 8, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
