# Lista de nombres públicos que deben poder importarse desde src.model
PUBLIC_NAMES = [
    # Clases base
    "EncoderBase",
    "DecoderBase",
    "BottleneckBase",
    "UNetBase",
    # Implementaciones principales
    "CNNEncoder",
    "ConvLSTMBottleneck",
    "CNNDecoder",
    "CNNConvLSTMUNet",
    # Variantes avanzadas
    "SwinV2CnnAsppUNet",
    "SwinV2EncoderAdapter",
    "ASPPModule",
    "BottleneckBlock",
    # Otros relevantes
    "BaseUNet",
    "ModelBase",
]


def test_public_exports():
    from src import model

    missing = []
    for name in PUBLIC_NAMES:
        if not hasattr(model, name):
            missing.append(name)
    assert not missing, f"Missing public exports in src.model: {missing}"
