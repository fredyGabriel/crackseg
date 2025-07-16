"""
Dataclasses for model configuration schemas.

Defines structured configuration classes for different model components
(Encoder, Bottleneck, Decoder, UNet) using dataclasses and OmegaConf.
These schemas help ensure type safety and provide structure for Hydra
configuration files.

Example YAML configuration for a UNet:

unet:
  _target_: src.model.core.unet.BaseUNet
  type: BaseUNet
  encoder:
    _target_: src.model.encoder.MockEncoder
    type: MockEncoder
    in_channels: 3
  bottleneck:
    _target_: src.model.bottleneck.MockBottleneck
    type: MockBottleneck
    in_channels: 64
  decoder:
    _target_: src.model.decoder.MockDecoder
    type: MockDecoder
    in_channels: 64
    skip_channels: [32, 16]
    out_channels: 1
  final_activation:
    _target_: torch.nn.Sigmoid
"""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING, OmegaConf


@dataclass
class EncoderConfig:
    """Configuration schema for Encoder components."""

    _target_: str = MISSING  # Class path for instantiation
    type: str = MISSING  # Registered encoder name (used by factory)
    in_channels: int = MISSING
    # Add other common or required encoder parameters here
    # Example: depth: int = 5
    # Example: feature_channels: List[int] = field(default_factory=lambda:
    # [64, 128, 256, 512])


@dataclass
class BottleneckConfig:
    """Configuration schema for Bottleneck components."""

    _target_: str = MISSING
    type: str = MISSING
    in_channels: int = MISSING
    # Add other common or required bottleneck parameters here
    # Example: out_channels: int = 1024


@dataclass
class DecoderConfig:
    """Configuration schema for Decoder components."""

    _target_: str = MISSING
    type: str = MISSING
    in_channels: int = MISSING
    skip_channels: list[int] = MISSING
    out_channels: int = 1  # Default to 1 for binary segmentation
    # Add other common or required decoder parameters here


@dataclass
class UNetConfig:
    """Configuration schema for the complete UNet model."""

    _target_: str = "src.model.core.unet.BaseUNet"  # Updated default path
    type: str = "BaseUNet"  # Name for potential registration
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    bottleneck: BottleneckConfig = field(default_factory=BottleneckConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    # Config for final activation module
    final_activation: dict[str, Any] | None = None

    def __post_init__(self):
        # Example validation: encoder out_channels == bottleneck in_channels
        if hasattr(self.encoder, "out_channels") and hasattr(
            self.bottleneck, "in_channels"
        ):
            enc_out = getattr(self.encoder, "out_channels", None)
            bott_in = getattr(self.bottleneck, "in_channels", None)
            if enc_out is not None and bott_in is not None:
                if enc_out != bott_in:
                    raise ValueError(
                        "Encoder out must match Bottleneck in channels"
                    )
        # Add more validation as needed


# Note:
# - Using MISSING encourages explicit configuration.
# - _target_ is used by Hydra for instantiation, but our current factory uses
# 'type'.
#   We include both for flexibility and potential future refactoring.
# - Default values should be added where appropriate.
# - More complex validation can be added in __post_init__.

# Helper functions


def load_unet_config_from_yaml(yaml_path: str) -> UNetConfig:
    """Load a UNetConfig from a YAML file."""
    cfg = OmegaConf.load(yaml_path)
    obj = OmegaConf.to_object(cfg)

    def normalize_keys(d: dict[Any, Any]) -> dict[str, Any]:
        if not all(isinstance(k, str) for k in d.keys()):
            raise ValueError(
                "All keys must be str for UNetConfig construction."
            )
        return dict(d)

    if (
        isinstance(obj, dict)
        and "unet" in obj
        and isinstance(obj["unet"], dict)
    ):
        return UNetConfig(**normalize_keys(obj["unet"]))
    if isinstance(obj, dict):
        return UNetConfig(**normalize_keys(obj))
    raise ValueError(
        "YAML config could not be converted to UNetConfig: unexpected "
        "structure."
    )


def validate_unet_config(config: UNetConfig) -> None:
    """Validate a UNetConfig instance. Raises ValueError if invalid."""
    # This will trigger __post_init__
    UNetConfig(**vars(config))
