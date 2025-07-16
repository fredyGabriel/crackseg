"""
Integration tests for the model factory configuration processing system.

These tests verify that the configuration validation, normalization, and
processing system works correctly with different types of model configurations.
"""

import os
from typing import Any
from unittest.mock import patch

import hydra
import pytest
import torch
from torch import nn

from crackseg.model import (
    BottleneckBase,
    DecoderBase,
    EncoderBase,
    UNetBase,
)
from crackseg.model.config.factory import (
    get_model_config_schema,
    parse_architecture_config,
    parse_component_config,
)
from crackseg.model.config.validation import normalize_config
from crackseg.model.factory.registry import Registry
from crackseg.model.factory.registry_setup import (
    architecture_registry,
    bottleneck_registry,
    decoder_registry,
    encoder_registry,
)


# --- Dummy Components for Testing ---
class DummyCNNEncoder(EncoderBase):
    def __init__(
        self,
        in_channels: int,
        hidden_dims: list[int] | None = None,
        depth: int = 4,
        **kwargs: Any,
    ):
        super().__init__(in_channels=in_channels)
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims or [64 * (2**i) for i in range(depth)]
        self._out_channels: int = int(self.hidden_dims[-1])
        self._skip_channels: list[int] = [
            int(x) for x in self.hidden_dims[:-1]
        ]

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Dummy shapes, assuming batch size 1 for simplicity
        dummy_output = torch.randn(
            1,
            self._out_channels,
            x.shape[2] // (2 ** len(self.hidden_dims)),
            x.shape[3] // (2 ** len(self.hidden_dims)),
        )
        dummy_skips = [
            torch.randn(
                1,
                ch,
                x.shape[2] // (2 ** (i + 1)),
                x.shape[3] // (2 ** (i + 1)),
            )
            for i, ch in enumerate(self._skip_channels)
        ]
        return dummy_output, dummy_skips

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def skip_channels(self) -> list[int]:
        return self._skip_channels


class DummyIdentity(BottleneckBase):
    def __init__(
        self, in_channels: int, out_channels: int | None = None, **kwargs: Any
    ):
        super().__init__(in_channels=in_channels)
        self.in_channels = in_channels
        self._out_channels: int = int(
            out_channels if out_channels is not None else in_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Need to return tensor potentially with different out_channels
        if self.in_channels != self._out_channels:
            # Simple linear layer to adjust channels if needed, or just randn
            return torch.randn(
                x.shape[0], self._out_channels, x.shape[2], x.shape[3]
            )
        return x

    @property
    def out_channels(self) -> int:
        return self._out_channels


class DummyCNNDecoder(DecoderBase):
    def __init__(
        self,
        in_channels: int,
        skip_channels_list: list[int] | None = None,
        out_channels: int = 1,
        **kwargs: Any,
    ):
        _skip_channels = skip_channels_list or []
        super().__init__(in_channels=in_channels, skip_channels=_skip_channels)
        self._out_channels: int = int(out_channels)

    def forward(
        self, x: torch.Tensor, skips: list[torch.Tensor]
    ) -> torch.Tensor:
        # Dummy forward, returns tensor with final out_channels
        batch_size = x.shape[0]
        h = x.shape[2] * 2
        w = x.shape[3] * 2
        return torch.randn(batch_size, self._out_channels, h, w)

    @property
    def out_channels(self) -> int:
        return self._out_channels


class DummyCBAM(nn.Module):
    def __init__(self, channels: int, **kwargs: Any):
        super().__init__()
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DummyUNet(UNetBase):
    def __init__(
        self,
        encoder: EncoderBase,
        bottleneck: BottleneckBase,
        decoder: DecoderBase,
        **kwargs: Any,
    ):
        super().__init__(
            encoder=encoder, bottleneck=bottleneck, decoder=decoder
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# --- End Dummy Components ---


# Load Hydra config for tests
@pytest.fixture(scope="session")
def cfg():
    # Calculate absolute path to configs directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    config_dir = os.path.join(project_root, "configs")

    if not os.path.isdir(config_dir):
        # Fallback if running from different directory
        config_dir = os.path.abspath(os.path.join(os.getcwd(), "configs"))
        if not os.path.isdir(config_dir):
            raise FileNotFoundError(
                f"Config directory not found at {config_dir}"
            )

    # Carga la configuración principal de Hydra
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        config = hydra.compose(config_name="config.yaml")
    return config


class TestComponentConfigParser:
    """Test component configuration parsing."""

    def setup_method(self):
        "Register necessary dummies for these tests."
        # Register CNNEncoder if not present
        if "CNNEncoder" not in encoder_registry:
            encoder_registry.register(name="CNNEncoder")(DummyCNNEncoder)
        # Register Identity if not present
        if "Identity" not in bottleneck_registry:
            bottleneck_registry.register(name="Identity")(DummyIdentity)
        # Register CNNDecoder if not present
        if "CNNDecoder" not in decoder_registry:
            decoder_registry.register(name="CNNDecoder")(DummyCNNDecoder)
        # Register DummyUNet if not present
        if "DummyUNet" not in architecture_registry:
            architecture_registry.register(name="DummyUNet")(DummyUNet)

    def teardown_method(self):
        "Unregister dummies added by setup_method."
        # Use try/except for robustness in case they weren't registered
        try:
            encoder_registry.unregister("CNNEncoder")
        except KeyError:
            pass
        try:
            bottleneck_registry.unregister("Identity")
        except KeyError:
            pass
        try:
            decoder_registry.unregister("CNNDecoder")
        except KeyError:
            pass
        try:
            architecture_registry.unregister("DummyUNet")
        except KeyError:
            pass

    def test_encoder_config_parsing(self, cfg: Any):
        """Test parsing encoder configurations."""
        # Use config values
        in_channels = cfg.data.num_channels_rgb
        hidden_dims = [64, 128, 256, 512]  # Si existe en config, usarlo
        encoder_config = {
            "type": "CNNEncoder",
            "in_channels": in_channels,
            "hidden_dims": hidden_dims,
        }

        parsed = parse_component_config(encoder_config, "encoder")
        assert parsed["type"] == "CNNEncoder"
        assert parsed["in_channels"] == in_channels

        # Missing type
        invalid_config = {
            "in_channels": in_channels,
            "hidden_dims": hidden_dims,
        }

        with pytest.raises(ValueError) as excinfo:
            parse_component_config(invalid_config, "encoder")
        assert "must specify a 'type'" in str(excinfo.value)

        # Unknown type
        unknown_config = {
            "type": "NonExistentEncoder",
            "in_channels": in_channels,
        }

        with pytest.raises(ValueError) as excinfo:
            parse_component_config(unknown_config, "encoder")
        assert "Unknown encoder type" in str(excinfo.value)

    def test_bottleneck_config_parsing(self, cfg: Any):
        """Test parsing bottleneck configurations."""
        in_channels = 512  # Si existe en config usarlo
        out_channels = 512
        bottleneck_config = {
            "type": "Identity",
            "in_channels": in_channels,
            "out_channels": out_channels,
        }

        parsed = parse_component_config(bottleneck_config, "bottleneck")
        assert parsed["type"] == "Identity"
        assert parsed["in_channels"] == in_channels
        assert parsed["out_channels"] == out_channels


class TestArchitectureConfigParser:
    """Test architecture configuration parsing."""

    def setup_method(self):
        "Register necessary dummies."
        # Register UNet if not present
        if "DummyUNet" not in architecture_registry:
            architecture_registry.register(name="DummyUNet")(DummyUNet)
        # Register components needed for standard UNet test
        if "CNNEncoder" not in encoder_registry:
            encoder_registry.register(name="CNNEncoder")(DummyCNNEncoder)
        if "Identity" not in bottleneck_registry:
            bottleneck_registry.register(name="Identity")(DummyIdentity)
        if "CNNDecoder" not in decoder_registry:
            decoder_registry.register(name="CNNDecoder")(DummyCNNDecoder)

    def teardown_method(self):
        "Unregister dummies added by setup_method."
        # Use try/except for robustness
        try:
            architecture_registry.unregister("DummyUNet")
        except KeyError:
            pass
        try:
            encoder_registry.unregister("CNNEncoder")
        except KeyError:
            pass
        try:
            bottleneck_registry.unregister("Identity")
        except KeyError:
            pass
        try:
            decoder_registry.unregister("CNNDecoder")
        except KeyError:
            pass

    def test_standard_architecture_parsing(self, cfg: Any):
        """Test parsing standard architecture configurations."""
        in_channels = cfg.data.num_channels_rgb
        out_channels = 1  # Si existe en config usarlo
        arch_config = {
            "type": "DummyUNet",
            "in_channels": in_channels,
            "out_channels": out_channels,
            "encoder": {"type": "CNNEncoder", "in_channels": in_channels},
            "bottleneck": {
                "type": "Identity",
                "in_channels": 512,
                "out_channels": 512,
            },
            "decoder": {
                "type": "CNNDecoder",
                "in_channels": 512,
                "out_channels": out_channels,
            },
        }

        parsed = parse_architecture_config(arch_config)
        assert parsed["type"] == "DummyUNet"
        assert parsed["in_channels"] == in_channels
        assert parsed["out_channels"] == out_channels
        assert "encoder" in parsed
        assert "bottleneck" in parsed
        assert "decoder" in parsed

        # Missing type
        invalid_config = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "encoder": {"type": "CNNEncoder", "in_channels": in_channels},
            "bottleneck": {
                "type": "Identity",
                "in_channels": 512,
                "out_channels": 512,
            },
            "decoder": {
                "type": "CNNDecoder",
                "in_channels": 512,
                "out_channels": out_channels,
            },
        }

        with pytest.raises(ValueError) as excinfo:
            parse_architecture_config(invalid_config)
        assert "must specify a 'type'" in str(excinfo.value)

        # Unknown architecture type
        unknown_config = {
            "type": "NonExistentArch",
            "in_channels": in_channels,
            "out_channels": out_channels,
            "encoder": {"type": "CNNEncoder", "in_channels": in_channels},
            "bottleneck": {
                "type": "Identity",
                "in_channels": 512,
                "out_channels": 512,
            },
            "decoder": {
                "type": "CNNDecoder",
                "in_channels": 512,
                "out_channels": out_channels,
            },
        }

        with pytest.raises(ValueError) as excinfo:
            parse_architecture_config(unknown_config)
        assert "Unknown architecture type" in str(excinfo.value)

    def test_hybrid_architecture_parsing(self):
        """Test parsing hybrid architecture configurations."""
        # Valid hybrid architecture config
        hybrid_config = {
            "type": "DummyUNet",  # Using DummyUNet as hybrid for testing
            "in_channels": 3,
            "out_channels": 1,
            "encoder": {"type": "CNNEncoder", "in_channels": 3},
            "bottleneck": {
                "type": "Identity",
                "in_channels": 512,
                "out_channels": 512,
            },
            "decoder": {
                "type": "CNNDecoder",
                "in_channels": 512,
                "out_channels": 1,
            },
            "components": {"attention": {"type": "CBAM", "channels": 256}},
        }

        # --- Mock component_registries for this test --- #
        # Create a specific attention registry for the mock
        mock_attention_registry = Registry(
            name="attention", base_class=nn.Module
        )
        mock_attention_registry.register(name="CBAM")(DummyCBAM)
        mock_registries = {
            "attention": mock_attention_registry
            # Add other global registries if needed for the test scope
        }

        # Patch component_registries within the factory module
        with patch(
            "crackseg.model.config.factory.component_registries",
            mock_registries,
        ):
            # Parse the hybrid config (this will now use the mocked registries)
            parsed = parse_architecture_config(hybrid_config)

        # --- End Mock --- #

        # Assertions remain the same
        assert parsed["type"] == "DummyUNet"
        assert "components" in parsed
        assert "attention" in parsed["components"]

        # Missing component type
        invalid_component_config = {
            "type": "DummyUNet",
            "in_channels": 3,
            "out_channels": 1,
            "encoder": {"type": "CNNEncoder", "in_channels": 3},
            "bottleneck": {
                "type": "Identity",
                "in_channels": 512,
                "out_channels": 512,
            },
            "decoder": {
                "type": "CNNDecoder",
                "in_channels": 512,
                "out_channels": 1,
            },
            "components": {
                "attention": {
                    # Missing type
                    "channels": 256
                }
            },
        }

        with pytest.raises(ValueError) as excinfo:
            parse_architecture_config(invalid_component_config)
        assert "must specify a 'type'" in str(excinfo.value)


class TestConfigNormalization:
    """Test configuration normalization with defaults."""

    def test_encoder_normalization(self, cfg: Any):
        """Test encoder configuration normalization."""
        # Minimal encoder config
        encoder_config = {
            "type": "CNNEncoder",
            "in_channels": cfg.data.num_channels_rgb,
        }

        # Create a dummy schema for testing
        from crackseg.model.config.core import (
            ConfigParam,
            ConfigSchema,
            ParamType,
        )

        encoder_schema = ConfigSchema(
            name="encoder",
            params=[
                ConfigParam(
                    name="type", param_type=ParamType.STRING, required=True
                ),
                ConfigParam(
                    name="in_channels",
                    param_type=ParamType.INTEGER,
                    required=True,
                ),
                ConfigParam(
                    name="hidden_dims",
                    param_type=ParamType.LIST,
                    required=False,
                    default=[64, 128, 256, 512],  # Si existe en config, usarlo
                ),
                ConfigParam(
                    name="dropout",
                    param_type=ParamType.FLOAT,
                    required=False,
                    default=0.0,
                ),
            ],
        )

        normalized = encoder_schema.normalize(encoder_config)
        assert normalized["type"] == "CNNEncoder"
        assert normalized["in_channels"] == cfg.data.num_channels_rgb
        assert normalized["hidden_dims"] == [
            64,
            128,
            256,
            512,
        ]
        assert normalized["dropout"] == 0.0

    def test_architecture_normalization(self, cfg: Any):
        """Test architecture configuration normalization."""
        # This test relies on the normalize_config function which uses
        # the schemas defined in config_validation.py

        # Minimal architecture config
        arch_config = {
            "type": "DummyUNet",
            "in_channels": cfg.data.num_channels_rgb,
            "out_channels": 1,
            "encoder": {
                "type": "CNNEncoder",
                "in_channels": cfg.data.num_channels_rgb,
            },
            "bottleneck": {
                "type": "Identity",
                "in_channels": 512,
                "out_channels": 512,
            },
            "decoder": {
                "type": "CNNDecoder",
                "in_channels": 512,
                "out_channels": 1,
            },
        }

        # We're primarily checking that normalization doesn't raise exceptions
        # Full testing of defaults is done in the config_validation tests
        normalized = normalize_config(arch_config)
        assert normalized["type"] == "DummyUNet"
        assert normalized["in_channels"] == cfg.data.num_channels_rgb
        assert normalized["out_channels"] == 1
        assert "encoder" in normalized
        assert "bottleneck" in normalized
        assert "decoder" in normalized


class TestModelConfigSchema:
    """Test model configuration schema generation."""

    def test_get_model_schema(self, cfg: Any):
        """Test getting a schema for a model type."""
        # Ensure DummyUNet is registered
        if "DummyUNet" not in architecture_registry:
            architecture_registry.register(name="DummyUNet")(DummyUNet)

        # Get schema for DummyUNet
        schema = get_model_config_schema("DummyUNet")

        # Check basic schema structure
        assert "type" in schema
        assert schema["type"]["default"] == "DummyUNet"
        assert "in_channels" in schema
        assert "out_channels" in schema
        assert "encoder" in schema
        assert "bottleneck" in schema
        assert "decoder" in schema

        # Unknown model type
        with pytest.raises(ValueError) as excinfo:
            get_model_config_schema("NonExistentModel")
        assert "Unknown model type" in str(excinfo.value)
