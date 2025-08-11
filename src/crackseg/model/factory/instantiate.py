"""Instantiation helpers split from the monolithic config factory."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from typing import Any, TypeVar, cast

import hydra.utils
from torch import nn

from crackseg.model.base.abstract import (
    BottleneckBase,
    DecoderBase,
    EncoderBase,
    UNetBase,
)

from .factory_utils import log_component_creation, merge_configs
from .normalize import normalize_config
from .registry_setup import (
    bottleneck_registry,
    decoder_registry,
    encoder_registry,
)
from .validation import validate_architecture_config, validate_component_config

log = logging.getLogger(__name__)


class InstantiationError(Exception):
    """Exception raised for errors during component instantiation."""

    pass


ComponentModelType = TypeVar("ComponentModelType", bound=nn.Module)


def parse_architecture_config(
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Parse a complete architecture configuration into component configs."""
    validate_architecture_config(config)
    return {
        "encoder": dict(config["encoder"]),
        "bottleneck": dict(config["bottleneck"]),
        "decoder": dict(config["decoder"]),
    }


def _try_instantiation_methods(
    config: dict[str, Any],
    component_type: str,
    registry: Any,
    base_class: type[nn.Module],
) -> nn.Module:
    """Try multiple methods to instantiate a component."""
    # 1) Hydra _target_
    if "_target_" in config:
        try:
            component = hydra.utils.instantiate(config)
            if not isinstance(component, base_class):
                raise TypeError(
                    f"Instantiated component is not a {base_class.__name__}"
                )
            log_component_creation(
                component_type.capitalize(), type(component).__name__
            )
            return component
        except Exception as e:  # noqa: BLE001 - keep broad here; logged below
            log.warning(
                f"Hydra instantiation failed for {component_type}: {e}"
            )

    # 2) Registry `type`
    if "type" in config:
        try:
            component_name = config["type"]
            params = {
                k: v
                for k, v in config.items()
                if k not in ["type", "_target_"]
            }
            if component_name in registry:
                component = registry.instantiate(component_name, **params)
                log_component_creation(
                    component_type.capitalize(), component_name
                )
                return cast(nn.Module, component)
        except Exception as e:  # noqa: BLE001
            log.warning(
                f"Registry instantiation failed for {component_type}: {e}"
            )

    # 3) Direct import fallback
    if "_target_" in config:
        try:
            target = config["_target_"]
            module_name, class_name = target.rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            params = {
                k: v
                for k, v in config.items()
                if k not in ["_target_", "type"]
            }
            component = cls(**params)
            if not isinstance(component, base_class):
                raise TypeError(
                    f"Instantiated component is not a {base_class.__name__}"
                )
            log_component_creation(component_type.capitalize(), class_name)
            return component
        except Exception as e:  # noqa: BLE001
            log.warning(
                f"Direct import instantiation failed for {component_type}: {e}"
            )

    raise InstantiationError(
        f"Failed to instantiate {component_type} component with config: {config}"
    )


def instantiate_encoder(
    config: Mapping[str, Any], runtime_params: Mapping[str, Any] | None = None
) -> EncoderBase:
    """Instantiate an encoder component from configuration."""
    try:
        validate_component_config(config, "encoder")
        full_config = normalize_config(config)
        if runtime_params:
            full_config = merge_configs(full_config, dict(runtime_params))
        return cast(
            EncoderBase,
            _try_instantiation_methods(
                full_config, "encoder", encoder_registry, nn.Module
            ),
        )
    except Exception as e:  # noqa: BLE001
        log.error(f"Error instantiating encoder: {e}", exc_info=True)
        raise InstantiationError(f"Failed to instantiate encoder: {e}") from e


def instantiate_bottleneck(
    config: Mapping[str, Any], runtime_params: Mapping[str, Any] | None = None
) -> BottleneckBase:
    """Instantiate a bottleneck component from configuration."""
    try:
        validate_component_config(config, "bottleneck")
        full_config = normalize_config(config)
        if runtime_params:
            full_config = merge_configs(full_config, dict(runtime_params))
        return cast(
            BottleneckBase,
            _try_instantiation_methods(
                full_config, "bottleneck", bottleneck_registry, nn.Module
            ),
        )
    except Exception as e:  # noqa: BLE001
        log.error(f"Error instantiating bottleneck: {e}", exc_info=True)
        raise InstantiationError(
            f"Failed to instantiate bottleneck: {e}"
        ) from e


def instantiate_decoder(
    config: Mapping[str, Any], runtime_params: Mapping[str, Any] | None = None
) -> DecoderBase:
    """Instantiate a decoder component from configuration."""
    try:
        validate_component_config(config, "decoder")
        full_config = normalize_config(config)
        if runtime_params:
            full_config = merge_configs(full_config, dict(runtime_params))
        return cast(
            DecoderBase,
            _try_instantiation_methods(
                full_config, "decoder", decoder_registry, nn.Module
            ),
        )
    except Exception as e:  # noqa: BLE001
        log.error(f"Error instantiating decoder: {e}", exc_info=True)
        raise InstantiationError(f"Failed to instantiate decoder: {e}") from e


def instantiate_hybrid_model(
    encoder: EncoderBase,
    bottleneck: BottleneckBase,
    decoder: DecoderBase,
    model_type: str = "BaseUNet",
) -> UNetBase:
    """Instantiate a hybrid model from pre-created components."""
    try:
        # Ensure registries are ready (best-effort)
        try:
            from ..components.registry_support import register_all_components

            register_all_components()
        except ImportError:
            pass

        from .registry_setup import architecture_registry

        if model_type not in architecture_registry:
            if model_type == "SwinV2CnnAsppUNet":
                try:
                    from ..architectures.swinv2_cnn_aspp_unet import (
                        SwinV2CnnAsppUNet,
                    )

                    model = SwinV2CnnAsppUNet(encoder, bottleneck, decoder)
                    log_component_creation("Hybrid Model", model_type)
                    return cast(UNetBase, model)
                except (
                    ImportError
                ) as import_error:  # noqa: PERF203 - logging clarity
                    log.error(
                        f"Direct import failed for {model_type}: {import_error}"
                    )

            available_models = architecture_registry.list_components()
            base_msg = f"Model class '{model_type}' not found in architecture registry."
            error_msg = f"{base_msg} Available models: {available_models}"
            raise InstantiationError(error_msg)

        model_cls = architecture_registry.get(model_type)
        model = model_cls(encoder, bottleneck, decoder)
        log_component_creation("Hybrid Model", model_type)
        return cast(UNetBase, model)
    except Exception as e:  # noqa: BLE001
        log.error(f"Error instantiating hybrid model: {e}", exc_info=True)
        raise InstantiationError(
            f"Failed to instantiate hybrid model: {e}"
        ) from e


def create_model_from_config(config: Mapping[str, Any]) -> UNetBase:
    """Create a complete model from a comprehensive configuration."""
    try:
        component_configs = parse_architecture_config(config)
        encoder = instantiate_encoder(component_configs["encoder"])
        bottleneck = instantiate_bottleneck(
            component_configs["bottleneck"],
            runtime_params={"in_channels": encoder.out_channels},
        )
        decoder = instantiate_decoder(
            component_configs["decoder"],
            runtime_params={
                "in_channels": bottleneck.out_channels,
                "skip_channels_list": list(reversed(encoder.skip_channels)),
            },
        )
        model_type = config.get("type", "BaseUNet")
        return instantiate_hybrid_model(
            encoder, bottleneck, decoder, model_type
        )
    except Exception as e:  # noqa: BLE001
        log.error(f"Error creating model from config: {e}", exc_info=True)
        raise InstantiationError(
            f"Failed to create model from config: {e}"
        ) from e


__all__ = [
    "InstantiationError",
    "parse_architecture_config",
    "instantiate_encoder",
    "instantiate_bottleneck",
    "instantiate_decoder",
    "instantiate_hybrid_model",
    "create_model_from_config",
]
