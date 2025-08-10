"""Model package public API with lazy imports.

Avoid importing heavy dependencies (e.g., `timm`, `hydra`) at package import
time. Symbols are provided lazily via module-level attribute access.
"""

from __future__ import annotations

import importlib
from typing import Any


def __getattr__(
    name: str,
) -> Any:  # pragma: no cover - straightforward dispatch
    # Factory/config
    if name in {
        "InstantiationError",
        "create_model_from_config",
        "instantiate_bottleneck",
        "instantiate_decoder",
        "instantiate_encoder",
        "instantiate_hybrid_model",
        "normalize_config",
        "parse_architecture_config",
        "validate_architecture_config",
        "validate_component_config",
    }:
        module = importlib.import_module("crackseg.model.factory.config")
        return getattr(module, name)

    if name in {
        "EncoderConfig",
        "BottleneckConfig",
        "DecoderConfig",
        "UNetConfig",
        "load_unet_config_from_yaml",
        "validate_unet_config",
    }:
        module = importlib.import_module(
            "crackseg.model.factory.config_schema"
        )
        return getattr(module, name)

    if name in {"ConfigurationError", "validate_config"}:
        module = importlib.import_module(
            "crackseg.model.factory.factory_utils"
        )
        return getattr(
            module, name if name == "validate_config" else "ConfigurationError"
        )

    if name == "Registry":
        module = importlib.import_module("crackseg.model.factory.registry")
        return module.Registry

    # Base classes
    if name in {"EncoderBase", "DecoderBase", "BottleneckBase", "UNetBase"}:
        from .base import (
            BottleneckBase as _BottleneckBase,
        )
        from .base import (
            DecoderBase as _DecoderBase,
        )
        from .base import (
            EncoderBase as _EncoderBase,
        )
        from .base import (
            UNetBase as _UNetBase,
        )

        return {
            "EncoderBase": _EncoderBase,
            "DecoderBase": _DecoderBase,
            "BottleneckBase": _BottleneckBase,
            "UNetBase": _UNetBase,
        }[name]

    # Optional heavy symbols
    if name in {"BaseUNet"}:
        from .core.unet import BaseUNet as _BaseUNet

        return _BaseUNet

    if name in {"CNNEncoder"}:
        from .encoder import CNNEncoder as _CNNEncoder

        return _CNNEncoder

    if name in {"CNNDecoder"}:
        from .decoder import CNNDecoder as _CNNDecoder

        return _CNNDecoder

    if name in {"SwinV2CnnAsppUNet"}:
        from .architectures.swinv2_cnn_aspp_unet import (
            SwinV2CnnAsppUNet as _SwinV2CnnAsppUNet,
        )

        return _SwinV2CnnAsppUNet

    if name in {"SwinV2EncoderAdapter"}:
        from .encoder.swin_v2_adapter import (
            SwinV2EncoderAdapter as _SwinV2EncoderAdapter,
        )

        return _SwinV2EncoderAdapter

    if name in {"ASPPModule"}:
        from .components.aspp import ASPPModule as _ASPPModule

        return _ASPPModule

    if name in {"BottleneckBlock"}:
        from .bottleneck.cnn_bottleneck import (
            BottleneckBlock as _BottleneckBlock,
        )

        return _BottleneckBlock

    raise AttributeError(f"module 'crackseg.model' has no attribute {name!r}")
