"""Architecture-level factory helpers (UNet creation from components)."""

from __future__ import annotations

from typing import Any, cast

from omegaconf import DictConfig
from torch import nn

from crackseg.model.base.abstract import UNetBase

from .component_factory import apply_cbam_to_model, instantiate_unet_components
from .factory_utils import hydra_to_dict, log_component_creation
from .registry_setup import component_registries


def get_unet_class(config: DictConfig) -> type[UNetBase]:
    """Resolve UNet class from config (via Hydra target)."""
    unet_target = config.get("_target_", "src.model.core.unet.BaseUNet")
    if isinstance(unet_target, str):
        from hydra.utils import get_class

        return cast(type[UNetBase], get_class(unet_target))
    from crackseg.model.base.abstract import UNetBase as _UNetBase

    return _UNetBase


def add_final_activation(
    model: nn.Module, activation_config: Any
) -> nn.Module:
    """Append final activation layer to model (via Hydra instantiate)."""
    try:
        import hydra.utils

        activation = hydra.utils.instantiate(activation_config)
        model_with_activation = nn.Sequential(model, activation)
        log_component_creation("Final Activation", type(activation).__name__)
        return model_with_activation
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Failed to instantiate final_activation: {exc}"
        ) from exc


def create_unet_from_config(config: DictConfig) -> UNetBase:
    """Create a UNet model from explicit component config sections."""
    if not isinstance(config, dict):  # type: ignore[unreachable]
        # Accept DictConfig; convert pieces as needed
        pass

    # Normalize sections for compatibility (values not used directly)
    _ = hydra_to_dict(config.encoder)
    _ = hydra_to_dict(config.bottleneck)
    _ = hydra_to_dict(config.decoder)

    encoder, bottleneck, decoder = instantiate_unet_components(config)

    arch_type = config.get("architecture_type")
    architecture_registry = component_registries.get("architecture")
    if architecture_registry is None:
        raise ValueError(
            f"Architecture registry not found for type '{arch_type}'."
        )
    unet_class = architecture_registry.get(arch_type)

    from crackseg.model.base.abstract import (
        BottleneckBase,
        DecoderBase,
        EncoderBase,
    )

    unet = unet_class(
        cast(EncoderBase, encoder),
        cast(BottleneckBase, bottleneck),
        cast(DecoderBase, decoder),
    )
    log_component_creation(
        "UNet", str(getattr(unet_class, "__name__", "Unknown"))
    )

    # Optional wrappers
    if "final_activation" in config:
        unet = add_final_activation(unet, config.final_activation)

    if config.get("cbam_enabled", False):
        cbam_params = config.get("cbam_params", {})
        unet = apply_cbam_to_model(
            unet, True, cbam_params, getattr(decoder, "out_channels", None)
        )

    return cast(UNetBase, unet)


def create_unet(config: DictConfig) -> nn.Module:
    """Create UNet model from DictConfig (legacy entrypoint)."""
    # Validate minimal structure
    _ = hydra_to_dict(config)

    try:
        encoder, bottleneck, decoder = instantiate_unet_components(config)
        UnetClass: type[UNetBase] = get_unet_class(config)
        from crackseg.model.base.abstract import (
            BottleneckBase,
            DecoderBase,
            EncoderBase,
        )

        unet_model: UNetBase = UnetClass(
            encoder=cast(EncoderBase, encoder),
            bottleneck=cast(BottleneckBase, bottleneck),
            decoder=cast(DecoderBase, decoder),
        )
        log_component_creation(
            "UNet", str(getattr(UnetClass, "__name__", "Unknown"))
        )

        model: nn.Module = unet_model
        if "final_activation" in config:
            model = add_final_activation(model, config.final_activation)
        if config.get("cbam_enabled", False):
            model = apply_cbam_to_model(
                model,
                True,
                config.get("cbam_params", {}),
                getattr(decoder, "out_channels", None),
            )
        return model
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Error instantiating UNet model: {exc}") from exc


__all__ = [
    "get_unet_class",
    "add_final_activation",
    "create_unet_from_config",
    "create_unet",
]
