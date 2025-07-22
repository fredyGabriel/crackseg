from pathlib import Path

import torch

from crackseg.model.common.utils import (
    count_parameters,
    estimate_memory_usage,
    estimate_receptive_field,
    get_layer_hierarchy,
)
from crackseg.model.common.visualization import (
    render_unet_architecture_diagram,
)


class DummyBlock(torch.nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels


class DummyModule(torch.nn.Module):
    def __init__(self, requires_grad: bool = True) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(
            torch.ones(10), requires_grad=requires_grad
        )


class DummyEncoder(DummyModule):
    def __init__(self):
        super().__init__()
        self.in_channels = 3
        self.out_channels = 8
        self.skip_channels = [4, 4]
        self.depth = 2
        self.encoder_blocks = [DummyBlock(3, 4), DummyBlock(4, 8)]


class DummyBottleneck(DummyModule):
    def __init__(self):
        super().__init__()
        self.in_channels = 8
        self.out_channels = 16


class DummyDecoder(DummyModule):
    def __init__(self):
        super().__init__()
        self.in_channels = 16
        self.out_channels = 1
        self.skip_channels = [4, 4]
        self.decoder_blocks = [DummyBlock(16, 8), DummyBlock(8, 4)]
        self.final_conv = DummyBlock(4, 1)


class DummyActivation(DummyModule):
    pass


def test_count_parameters() -> None:
    model = DummyModule()
    trainable, non_trainable = count_parameters(model)
    assert trainable == 10  # noqa: PLR2004
    assert non_trainable == 0
    # Add a non-trainable param
    model.register_buffer("buf", torch.ones(5))
    trainable, non_trainable = count_parameters(model)
    assert trainable == 10  # noqa: PLR2004


def test_estimate_receptive_field() -> None:
    encoder = DummyEncoder()
    rf = estimate_receptive_field(encoder)
    assert "theoretical_rf_size" in rf
    assert "downsampling_factor" in rf
    # Encoder sin depth
    encoder2 = DummyEncoder()
    del encoder2.depth
    rf2 = estimate_receptive_field(encoder2)
    assert "note" in rf2


def test_estimate_memory_usage() -> None:
    model = DummyModule()
    encoder = DummyEncoder()

    def get_out_ch():
        return 1

    mem = estimate_memory_usage(model, encoder, get_out_ch, (1, 3, 32, 32))
    assert "model_size_mb" in mem
    assert "estimated_activation_mb" in mem
    mem2 = estimate_memory_usage(model, encoder, get_out_ch)
    assert "model_size_mb" in mem2
    assert "note" in mem2


def test_get_layer_hierarchy() -> None:
    encoder = DummyEncoder()
    bottleneck = DummyBottleneck()
    decoder = DummyDecoder()
    activation = DummyActivation()
    hierarchy = get_layer_hierarchy(encoder, bottleneck, decoder, activation)
    assert isinstance(hierarchy, list)
    assert any(layer["name"] == "Encoder" for layer in hierarchy)
    assert any(layer["name"] == "Bottleneck" for layer in hierarchy)
    assert any(layer["name"] == "Decoder" for layer in hierarchy)
    assert any(layer["name"] == "FinalActivation" for layer in hierarchy)


def test_render_unet_architecture_diagram(tmp_path: Path) -> None:
    encoder = DummyEncoder()
    bottleneck = DummyBottleneck()
    decoder = DummyDecoder()
    activation = DummyActivation()
    hierarchy = get_layer_hierarchy(encoder, bottleneck, decoder, activation)
    # Solo verificar que no lanza error y genera archivo
    out_file = tmp_path / "unet_architecture"
    render_unet_architecture_diagram(hierarchy, str(out_file), view=False)
    assert (tmp_path / "unet_architecture.png").exists()
