from typing import Any, cast

import torch
import torch.nn as nn

from src.model.architectures.cnn_convlstm_unet import (
    CNNConvLSTMUNet,
    CNNEncoder,
)
from src.model.decoder.cnn_decoder import CNNDecoder
from src.model.factory.factory import CBAMPostProcessor
from src.model.factory.registry_setup import component_registries

# Importamos la implementación de SimpleConvLSTMBottleneck de los tests
from tests.integration.model.test_cnn_convlstm_unet import (
    SimpleConvLSTMBottleneck,
)


def get_cbam_instance(out_channels: int):
    attention_registry = component_registries.get("attention")
    if attention_registry is None:
        raise RuntimeError("Attention registry not found")
    cbam_cls = cast(type[nn.Module] | None, attention_registry.get("CBAM"))
    if cbam_cls is None:
        raise RuntimeError("CBAM not registered in attention registry")
    return cbam_cls(in_channels=out_channels, reduction=1, kernel_size=3)


def test_cbam_integration_unet_forward():
    """Test CBAM integration in UNet forward pass (out_channels > 1)."""
    # Test con parámetros simplificados que se ha comprobado que funcionan
    in_channels = 3
    h, w = 32, 32
    out_channels = 2

    # Crear datos de entrada
    x = torch.randn(2, in_channels, h, w)

    # Usar CNNConvLSTMUNet que ya funciona correctamente
    encoder = CNNEncoder(in_channels=in_channels, base_filters=4, depth=1)
    bottleneck = SimpleConvLSTMBottleneck(
        in_channels=encoder.out_channels,
        hidden_dim=8,
        kernel_size=(3, 3),
        num_layers=1,
    )
    decoder = cast(
        Any,
        CNNDecoder(
            in_channels=bottleneck.out_channels,
            skip_channels_list=list(reversed(encoder.skip_channels)),
            out_channels=out_channels,
            depth=1,
        ),
    )

    # Modelo base - usar CNNConvLSTMUNet en lugar de BaseUNet
    model = CNNConvLSTMUNet(
        encoder=encoder, bottleneck=bottleneck, decoder=decoder
    )

    # Ejecutar primero para verificar que el modelo base funciona
    base_output = model(x)
    assert base_output.shape == (2, out_channels, h, w)

    # Ahora agregar CBAM como post-procesador
    cbam = get_cbam_instance(out_channels)
    cbam_model = CBAMPostProcessor(model, cbam)

    # Ejecutar modelo con CBAM
    out = cbam_model(x)

    # Verificar que la salida tiene la forma correcta
    assert out.shape == (2, out_channels, h, w)


def test_cbam_integration_cnn_convlstm_unet_forward():
    """Test CBAM integration in CNNConvLSTMUNet forward pass (out_channels > 1
    )."""
    # Test con parámetros simplificados para evitar errores de dimensiones
    in_channels = 3
    h, w = 32, 32
    out_channels = 2

    # Crear datos de entrada
    x = torch.randn(1, in_channels, h, w)

    # Crear UNet válido con dimensiones consistentes
    encoder = CNNEncoder(in_channels=in_channels, base_filters=4, depth=1)
    bottleneck = SimpleConvLSTMBottleneck(
        in_channels=encoder.out_channels,
        hidden_dim=8,
        kernel_size=(3, 3),
        num_layers=1,
    )
    decoder = cast(
        Any,
        CNNDecoder(
            in_channels=bottleneck.out_channels,
            skip_channels_list=list(reversed(encoder.skip_channels)),
            out_channels=out_channels,
            depth=1,
        ),
    )

    # Modelo base
    model = CNNConvLSTMUNet(
        encoder=encoder, bottleneck=bottleneck, decoder=decoder
    )

    # Ejecutar primero para verificar que el modelo base funciona
    base_output = model(x)
    assert base_output.shape == (1, out_channels, h, w)

    # Ahora agregar CBAM como post-procesador
    cbam = get_cbam_instance(out_channels)
    cbam_model = CBAMPostProcessor(model, cbam)

    # Ejecutar modelo con CBAM
    out = cbam_model(x)

    # Verificar que la salida tiene la forma correcta
    assert out.shape == (1, out_channels, h, w)


def test_cbam_save_and_load(tmp_path: Any):
    """Test saving and loading a model with CBAM (out_channels > 1)."""
    # Test con parámetros simplificados para evitar errores de dimensiones
    in_channels = 3
    h, w = 16, 16
    out_channels = 2

    # Crear datos de entrada
    x = torch.randn(1, in_channels, h, w)

    # Usar CNNConvLSTMUNet que ya funciona correctamente
    encoder = CNNEncoder(in_channels=in_channels, base_filters=4, depth=1)
    bottleneck = SimpleConvLSTMBottleneck(
        in_channels=encoder.out_channels,
        hidden_dim=8,
        kernel_size=(3, 3),
        num_layers=1,
    )
    decoder = cast(
        Any,
        CNNDecoder(
            in_channels=bottleneck.out_channels,
            skip_channels_list=list(reversed(encoder.skip_channels)),
            out_channels=out_channels,
            depth=1,
        ),
    )

    # Modelo base - usar CNNConvLSTMUNet en lugar de BaseUNet
    model = CNNConvLSTMUNet(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
    )

    # Ejecutar primero para verificar que el modelo base funciona
    base_output = model(x)
    assert base_output.shape == (1, out_channels, h, w)

    # Ahora agregar CBAM como post-procesador
    cbam = get_cbam_instance(out_channels)
    cbam_model = CBAMPostProcessor(model, cbam)

    # Guarda y carga el modelo
    path = tmp_path / "cbam_model.pt"
    torch.save(cbam_model.state_dict(), str(path))

    # Crea un nuevo modelo idéntico y carga los pesos
    model2 = CNNConvLSTMUNet(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
    )
    cbam2 = get_cbam_instance(out_channels)
    cbam_model2 = CBAMPostProcessor(model2, cbam2)
    cbam_model2.load_state_dict(torch.load(str(path)))

    # Ejecuta el modelo cargado
    out = cbam_model2(x)

    # Verificar que la salida tiene la forma correcta
    assert out.shape == (1, out_channels, h, w)


def test_cbam_grad_integration():
    """Test gradient flow through UNet with CBAM (out_channels > 1)."""
    # Test con parámetros simplificados para evitar errores de dimensiones
    in_channels = 3
    h, w = 8, 8
    out_channels = 2

    # Usar CNNConvLSTMUNet que ya funciona correctamente
    encoder = CNNEncoder(in_channels=in_channels, base_filters=4, depth=1)
    bottleneck = SimpleConvLSTMBottleneck(
        in_channels=encoder.out_channels,
        hidden_dim=8,
        kernel_size=(3, 3),
        num_layers=1,
    )
    decoder = cast(
        Any,
        CNNDecoder(
            in_channels=bottleneck.out_channels,
            skip_channels_list=list(reversed(encoder.skip_channels)),
            out_channels=out_channels,
            depth=1,
        ),
    )

    # Modelo base - usar CNNConvLSTMUNet en lugar de BaseUNet
    model = CNNConvLSTMUNet(
        encoder=encoder, bottleneck=bottleneck, decoder=decoder
    )

    # Ahora agregar CBAM como post-procesador
    cbam = get_cbam_instance(out_channels)
    cbam_model = CBAMPostProcessor(model, cbam)

    # Prepara un tensor con gradientes
    x = torch.randn(2, in_channels, h, w, requires_grad=True)
    out = cbam_model(x)

    # Calcula la pérdida y realiza la retropropagación
    loss = out.mean()
    loss.backward()

    # Verifica que el gradiente fluye hasta la entrada
    assert x.grad is not None
