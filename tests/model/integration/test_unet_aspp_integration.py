import pytest
import torch
import logging
from typing import List
import math

# Import components for direct instantiation
from src.model.components.aspp import ASPPModule
from src.model.encoder.cnn_encoder import CNNEncoder

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "base_filters, in_channels, out_channels, dilations",
    [
        (16, 3, 1, [1, 6, 12, 18]),
    ]
)
def test_simple_aspp(
    base_filters: int,
    in_channels: int,
    out_channels: int,
    dilations: List[int]
):
    """
    Test simplificado para verificar la funcionalidad básica de ASPP.

    Prueba que el módulo ASPP funciona correctamente de forma independiente,
    sin integrarlo con toda la UNet.
    """
    # Preparar dimensiones
    batch_size = 2
    input_height = 64
    input_width = 64

    # Crear una entrada simple
    x = torch.randn(batch_size, in_channels, input_height, input_width)

    # Crear el encoder
    encoder = CNNEncoder(
        in_channels=in_channels,
        init_features=base_filters,
        depth=4
    )

    # Extraer features y skip connections
    features, skips = encoder(x)

    # Crear ASPP module
    aspp = ASPPModule(
        in_channels=encoder.out_channels,
        output_channels=encoder.out_channels,
        dilation_rates=dilations,
        dropout_rate=0.1
    )

    # Aplicar ASPP
    aspp_output = aspp(features)

    # Verificar dimensiones de salida
    assert aspp_output.shape[0] == batch_size
    assert aspp_output.shape[1] == encoder.out_channels
    assert aspp_output.shape[2:] == features.shape[2:]

    # Verificar que no hay NaN o infinitos
    assert torch.isfinite(aspp_output).all()

    logger.info(
        f"ASPP test passed: Input shape {features.shape} -> "
        f"Output shape {aspp_output.shape}"
    )


@pytest.mark.parametrize(
    "base_filters, in_channels, out_channels, dilations",
    [
        (16, 3, 1, [1, 6, 12, 18]),
    ]
)
def test_aspp_simplified_unet(
    base_filters: int,
    in_channels: int,
    out_channels: int,
    dilations: List[int]
):
    """
    Test usando una estructura simplificada para la integración de ASPP.

    Esta prueba verifica que ASPP puede integrarse correctamente con el flujo
    de decodificación de UNet, simulando los pasos clave:
    1. Codificación (encoder)
    2. Bottleneck (ASPP)
    3. Primera etapa de decodificación
    """
    # Definir parámetros
    encoder_depth = 4
    input_height = 64
    input_width = 64
    batch_size = 2

    # Crear entrada
    x = torch.randn(batch_size, in_channels, input_height, input_width)

    # 1. Crear encoder
    encoder = CNNEncoder(
        in_channels=in_channels,
        init_features=base_filters,
        depth=encoder_depth
    )

    # Aplicar encoder
    features, skips = encoder(x)

    # 2. Crear bottleneck ASPP
    bottleneck = ASPPModule(
        in_channels=encoder.out_channels,
        output_channels=encoder.out_channels,
        dilation_rates=dilations,
        dropout_rate=0.1
    )

    # Aplicar bottleneck
    bottleneck_output = bottleneck(features)

    # 3. Crear un adaptador con el número correcto de canales para reducir
    # los canales a la mitad antes de entrar en el primer DecoderBlock
    adapter_output_channels = math.ceil(bottleneck.out_channels / 2) * 2
    adapter = torch.nn.Conv2d(
        bottleneck.out_channels,
        adapter_output_channels,
        kernel_size=1,
        bias=False
    )

    # Aplicar adaptador
    adapted_output = adapter(bottleneck_output)

    # 4. Aplicar la primera etapa de upsampling directamente
    skip_channels = list(reversed(encoder.skip_channels))
    logger.info(f"Encoder skip channels: {encoder.skip_channels}")
    logger.info(f"Reversed skip channels: {skip_channels}")

    # 5. Crear la capa de upsampling
    upsampler = torch.nn.Upsample(
        scale_factor=2,
        mode='bilinear',
        align_corners=True
    )
    # Aplicar upsampling
    upsampled = upsampler(adapted_output)

    # Reducir canales a la mitad después del upsampling
    first_up_conv = torch.nn.Conv2d(
        adapter_output_channels,
        adapter_output_channels // 2,
        kernel_size=1
    )
    upsampled_reduced = first_up_conv(upsampled)

    # Concatenar con el skip correspondiente
    first_skip = skips[-1]  # Tomar el último skip (más cercano al bottleneck)
    logger.info(f"Upsampled shape: {upsampled_reduced.shape}")
    logger.info(f"Skip shape: {first_skip.shape}")
    concat = torch.cat([upsampled_reduced, first_skip], dim=1)

    # Verificar dimensiones
    expected_concat_channels = (adapter_output_channels // 2
                                ) + skip_channels[0]
    assert concat.shape[1] == expected_concat_channels, (
        f"Expected {expected_concat_channels} channels, "
        f"got {concat.shape[1]}"
    )

    logger.info(f"Concatenated tensor shape: {concat.shape}")
    logger.info("Test passed for simplified decoder path")

    # Comentarios sobre la integración completa:
    # 1. Al integrar ASPP en UNet, debemos prestar atención a las dimensiones
    # 2. El DecoderBlock espera recibir un tensor con canales específicos
    # 3. Antes de hacer upsampling, es necesario adaptar los canales
    # 4. Los skip connections deben estar en el orden correcto
    # (baja a alta resolución)
