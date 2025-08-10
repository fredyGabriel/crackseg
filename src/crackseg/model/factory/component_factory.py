"""Component-level factory helpers (UNet parts, CBAM integration)."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig
from torch import nn

from .config import (
    instantiate_bottleneck,
    instantiate_decoder,
    instantiate_encoder,
)
from .factory_utils import (
    extract_runtime_params,
    hydra_to_dict,
    log_component_creation,
)
from .registry_setup import component_registries


class CBAMPostProcessor(nn.Module):
    """Applies CBAM as a post-processor at the end of the model."""

    def __init__(self, original_model: nn.Module, cbam: nn.Module) -> None:
        super().__init__()
        self.model: nn.Module = original_model
        self.cbam: nn.Module = cbam

    def forward(self, x: Any) -> Any:  # noqa: D401 - simple forward
        output = self.model(x)
        return self.cbam(output)


class FinalCBAMDecoder(nn.Module):
    """Decorator that applies CBAM at the end of a decoder pipeline."""

    def __init__(self, decoder: nn.Module, cbam: nn.Module) -> None:
        super().__init__()
        self.add_module("decoder", decoder)
        self.add_module("cbam", cbam)
        self.decoder: nn.Module = decoder
        self.cbam: nn.Module = cbam
        self._out_channels: int = getattr(decoder, "out_channels", 1)

    def forward(self, x: Any, skips: Any = None) -> Any:  # noqa: D401
        decoded = self.decoder(x, skips)
        return self.cbam(decoded)

    @property
    def out_channels(self) -> int:
        return self._out_channels


def create_cbam_module(
    in_channels: int, cbam_params: dict[str, Any] | None = None
) -> nn.Module:
    """Create a CBAM module via attention registry."""
    cbam_params = cbam_params or {}
    attention_registry = component_registries.get("attention")
    if attention_registry is None:
        raise ValueError("Attention registry not found for CBAM.")

    attention_type = cbam_params.get("attention_type", "CBAM")
    attention_params = cbam_params.get(
        "attention_params", {"channels": in_channels}
    )
    return attention_registry.instantiate(attention_type, **attention_params)


def apply_cbam_to_model(
    model: nn.Module,
    cbam_enabled: bool,
    cbam_params: dict[str, Any] | None = None,
    output_channels: int | None = None,
) -> nn.Module:
    """Apply CBAM wrapper to a model if enabled."""
    if not cbam_enabled:
        return model

    channels = output_channels
    if channels is None:
        channels = getattr(model, "out_channels", 1)
        if not isinstance(channels, int):
            channels = 1

    cbam = create_cbam_module(channels, cbam_params)
    return CBAMPostProcessor(model, cbam)


def insert_cbam_if_enabled(
    component: nn.Module, config: dict[str, Any] | DictConfig
) -> nn.Module:
    """Optionally insert CBAM after a component based on config flags."""
    cbam_enabled: bool = config.get("cbam_enabled", False)  # type: ignore[reportAttributeAccessIssue]
    cbam_params: dict[str, Any] = config.get("cbam_params", {})  # type: ignore[reportAttributeAccessIssue]
    if not cbam_enabled:
        return component

    # Determine channels for CBAM
    if "in_channels" in cbam_params:
        channels = cbam_params["in_channels"]
    elif hasattr(component, "out_channels"):
        channels = getattr(component, "out_channels", 64)
    else:
        channels = 64

    cbam_args: dict[str, Any] = dict(cbam_params)
    cbam_args["in_channels"] = channels
    cbam = create_cbam_module(channels, cbam_args)

    # For decoders, decorate with FinalCBAMDecoder
    if hasattr(component, "skip_channels"):
        return FinalCBAMDecoder(component, cbam)
    return nn.Sequential(component, cbam)


def instantiate_unet_components(
    config: DictConfig,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    """Instantiate encoder, bottleneck, and decoder for a UNet from config."""
    encoder_cfg = hydra_to_dict(config.encoder)
    bottleneck_cfg = hydra_to_dict(config.bottleneck)
    decoder_cfg = hydra_to_dict(config.decoder)

    encoder = instantiate_encoder(encoder_cfg)
    log_component_creation("Encoder", type(encoder).__name__)

    bottleneck_runtime_params = extract_runtime_params(
        encoder, {"out_channels": "in_channels"}
    )
    bottleneck = instantiate_bottleneck(
        bottleneck_cfg, runtime_params=bottleneck_runtime_params
    )
    log_component_creation("Bottleneck", type(bottleneck).__name__)

    # Optional: derive decoder runtime params (out_channels → in_channels)
    decoder_runtime_params = extract_runtime_params(
        bottleneck, {"out_channels": "in_channels"}
    )
    decoder = instantiate_decoder(
        decoder_cfg, runtime_params=decoder_runtime_params
    )
    log_component_creation("Decoder", type(decoder).__name__)

    return encoder, bottleneck, decoder


__all__ = [
    "CBAMPostProcessor",
    "FinalCBAMDecoder",
    "create_cbam_module",
    "apply_cbam_to_model",
    "insert_cbam_if_enabled",
    "instantiate_unet_components",
]
