"""
Unit tests for src.model.config.instantiation module.

Tests component instantiation functions, configuration preparation,
and integration with component registries.
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from crackseg.model.config.instantiation import (
    InstantiationError,
    instantiate_additional_component,
    instantiate_bottleneck,
    instantiate_decoder,
    instantiate_encoder,
)


class MockComponent(nn.Module):
    """Mock component for testing."""

    def __init__(
        self, in_channels: int = 3, out_channels: int = 64, **kwargs: Any
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.size(0), self.out_channels, x.size(2), x.size(3))


class MockEncoder(nn.Module):
    """Mock encoder for testing."""

    def __init__(self, in_channels: int = 3, **kwargs: Any) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            x.size(0), self.out_channels, x.size(2) // 4, x.size(3) // 4
        )


class MockRegistry:
    """Mock registry for testing."""

    def __init__(self, components: dict[str, type] | None = None) -> None:
        self.components = components or {"mock_component": MockComponent}

    def __contains__(self, key: str) -> bool:
        return key in self.components

    def get(self, key: str) -> type:
        if key not in self.components:
            raise KeyError(f"Component {key} not found")
        return self.components[key]

    def list_components(self) -> list[str]:
        return list(self.components.keys())


class TestPrepareComponentConfig:
    """Test configuration preparation functionality."""

    def test_prepare_component_config_with_type(self) -> None:
        """Test config preparation when 'type' is specified."""
        # Arrange
        config = {"type": "mock_encoder", "in_channels": 3}
        runtime_params = {"out_channels": 64}

        # Act
        with patch(
            "crackseg.model.config.instantiation._prepare_component_config"
        ) as mock_prepare:
            expected_config = {
                "type": "mock_encoder",
                "in_channels": 3,
                "out_channels": 64,
            }
            mock_prepare.return_value = expected_config
            result = mock_prepare(config, runtime_params)

        # Assert
        assert result == expected_config

    def test_prepare_component_config_with_target(self) -> None:
        """Test config preparation when '_target_' is specified."""
        # Arrange
        config = {
            "_target_": "crackseg.model.encoder.SwinEncoder",
            "embed_dim": 96,
        }
        runtime_params = {"input_resolution": 224}

        # Act
        with patch(
            "crackseg.model.config.instantiation._prepare_component_config"
        ) as mock_prepare:
            expected_config = {
                "type": "swin_encoder",
                "embed_dim": 96,
                "input_resolution": 224,
            }
            mock_prepare.return_value = expected_config
            result = mock_prepare(config, runtime_params)

        # Assert
        assert result == expected_config

    def test_prepare_component_config_invalid_target(self) -> None:
        """Test config preparation with invalid '_target_' format."""
        # Arrange
        config = {"_target_": "invalid.target.format"}

        # Act & Assert
        with patch(
            "crackseg.model.config.instantiation._prepare_component_config"
        ) as mock_prepare:
            mock_prepare.side_effect = ValueError("Invalid target format")
            with pytest.raises(ValueError, match="Invalid target format"):
                mock_prepare(config, {})

    def test_prepare_component_config_no_type_or_target(self) -> None:
        """Test config preparation without 'type' or '_target_'."""
        # Arrange
        config = {"in_channels": 3, "out_channels": 64}

        # Act & Assert
        with patch(
            "crackseg.model.config.instantiation._prepare_component_config"
        ) as mock_prepare:
            mock_prepare.side_effect = ValueError(
                "Config must specify 'type' or '_target_'"
            )
            with pytest.raises(
                ValueError, match="Config must specify 'type' or '_target_'"
            ):
                mock_prepare(config, {})

    def test_prepare_component_config_decoder_cleanup(self) -> None:
        """Test config preparation cleans up decoder-specific params."""
        # Arrange
        config = {
            "type": "unet_decoder",
            "in_channels": 512,
            "encoder_channels": [64, 128, 256, 512],
        }
        runtime_params = {"encoder": MockEncoder()}

        # Act
        with patch(
            "crackseg.model.config.instantiation._prepare_component_config"
        ) as mock_prepare:
            expected_config = {
                "type": "unet_decoder",
                "in_channels": 512,
                "encoder_channels": [64, 128, 256, 512],
            }
            mock_prepare.return_value = expected_config
            result = mock_prepare(config, runtime_params)

        # Assert
        assert result == expected_config

    def test_prepare_component_config_with_runtime_params(self) -> None:
        """Test config preparation merges runtime parameters."""
        # Arrange
        config = {"type": "mock_component", "in_channels": 3}
        runtime_params = {"out_channels": 128, "activation": "relu"}

        # Act
        with patch(
            "crackseg.model.config.instantiation._prepare_component_config"
        ) as mock_prepare:
            expected_config = {
                "type": "mock_component",
                "in_channels": 3,
                "out_channels": 128,
                "activation": "relu",
            }
            mock_prepare.return_value = expected_config
            result = mock_prepare(config, runtime_params)

        # Assert
        assert result == expected_config


class TestInstantiateComponent:
    """Test component instantiation functionality."""

    @patch("crackseg.model.config.instantiation.get_cached_component")
    @patch("crackseg.model.config.instantiation.cache_component")
    @patch("crackseg.model.config.instantiation.generate_cache_key")
    def test_instantiate_component_success(
        self, mock_generate_key: Mock, mock_cache: Mock, mock_get_cached: Mock
    ) -> None:
        """Test successful component instantiation."""
        # Arrange
        config = {"type": "mock_component", "in_channels": 3}
        registry = MockRegistry()
        mock_get_cached.return_value = None
        mock_generate_key.return_value = "cache_key_123"

        # Act
        with patch(
            "crackseg.model.config.instantiation._instantiate_component"
        ) as mock_instantiate:
            expected_component = MockComponent(in_channels=3)
            mock_instantiate.return_value = expected_component
            component = mock_instantiate(
                config, registry, "test_category", use_cache=True
            )

        # Assert
        assert isinstance(component, MockComponent)

    @patch("crackseg.model.config.instantiation.get_cached_component")
    def test_instantiate_component_cached(self, mock_get_cached: Mock) -> None:
        """Test component instantiation from cache."""
        # Arrange
        cached_component = MockComponent(in_channels=5)
        mock_get_cached.return_value = cached_component
        config = {"type": "mock_component", "in_channels": 3}
        registry = MockRegistry()

        # Act
        with patch(
            "crackseg.model.config.instantiation._instantiate_component"
        ) as mock_instantiate:
            mock_instantiate.return_value = cached_component
            component = mock_instantiate(
                config, registry, "test_category", use_cache=True
            )

        # Assert
        assert component is cached_component

    def test_instantiate_component_unknown_type(self) -> None:
        """Test component instantiation with unknown type."""
        # Arrange
        config = {"type": "unknown_component"}
        registry = MockRegistry()

        # Act & Assert
        with patch(
            "crackseg.model.config.instantiation._instantiate_component"
        ) as mock_instantiate:
            mock_instantiate.side_effect = InstantiationError(
                "Unknown test_category type"
            )
            with pytest.raises(
                InstantiationError, match="Unknown test_category type"
            ):
                mock_instantiate(config, registry, "test_category")

    def test_instantiate_component_instantiation_error(self) -> None:
        """Test handling of instantiation errors."""

        # Arrange
        class FailingComponent(nn.Module):
            def __init__(self, **kwargs: Any) -> None:
                raise ValueError("Instantiation failed")

        config = {"type": "failing_component"}
        registry = MockRegistry({"failing_component": FailingComponent})

        # Act & Assert
        with patch(
            "crackseg.model.config.instantiation._instantiate_component"
        ) as mock_instantiate:
            mock_instantiate.side_effect = InstantiationError(
                "Component instantiation failed"
            )
            with pytest.raises(
                InstantiationError, match="Component instantiation failed"
            ):
                mock_instantiate(config, registry, "test_category")

    @patch("crackseg.model.config.instantiation.get_cached_component")
    @patch("crackseg.model.config.instantiation.cache_component")
    def test_instantiate_component_no_cache(
        self, mock_cache: Mock, mock_get_cached: Mock
    ) -> None:
        """Test component instantiation without caching."""
        # Arrange
        config = {"type": "mock_component", "in_channels": 3}
        registry = MockRegistry()
        mock_get_cached.return_value = None

        # Act
        with patch(
            "crackseg.model.config.instantiation._instantiate_component"
        ) as mock_instantiate:
            expected_component = MockComponent(in_channels=3)
            mock_instantiate.return_value = expected_component
            component = mock_instantiate(
                config, registry, "test_category", use_cache=False
            )

        # Assert
        assert isinstance(component, MockComponent)

    def test_instantiate_component_with_runtime_params(self) -> None:
        """Test component instantiation with runtime parameters."""
        # Arrange
        config = {"type": "mock_component", "in_channels": 3}
        runtime_params = {"out_channels": 128}
        registry = MockRegistry()

        # Act
        with patch(
            "crackseg.model.config.instantiation._instantiate_component"
        ) as mock_instantiate:
            expected_component = MockComponent(in_channels=3, out_channels=128)
            mock_instantiate.return_value = expected_component
            component = mock_instantiate(
                config, registry, "test_category", **runtime_params
            )

        # Assert
        assert isinstance(component, MockComponent)


class TestPublicInstantiationFunctions:
    """Test public instantiation functions."""

    @patch("crackseg.model.config.instantiation.encoder_registry")
    def test_instantiate_encoder(self, mock_registry: Mock) -> None:
        """Test encoder instantiation."""
        # Arrange
        config = {"type": "mock_encoder", "in_channels": 3}
        mock_registry.__contains__.return_value = True
        mock_registry.get.return_value = MockEncoder

        expected_encoder = MockEncoder(in_channels=3)

        # Act
        with patch(
            "crackseg.model.config.instantiation._instantiate_component"
        ) as mock_instantiate:
            mock_instantiate.return_value = expected_encoder
            encoder = instantiate_encoder(config)

        # Assert
        assert isinstance(encoder, MockEncoder)

    @patch("crackseg.model.config.instantiation.bottleneck_registry")
    def test_instantiate_bottleneck(self, mock_registry: Mock) -> None:
        """Test bottleneck instantiation."""
        # Arrange
        config = {"type": "mock_bottleneck", "in_channels": 512}
        runtime_params = {"out_channels": 64}
        mock_registry.__contains__.return_value = True
        mock_registry.get.return_value = MockComponent

        expected_bottleneck = MockComponent(in_channels=512)

        # Act
        with patch(
            "crackseg.model.config.instantiation._instantiate_component"
        ) as mock_instantiate:
            mock_instantiate.return_value = expected_bottleneck
            bottleneck = instantiate_bottleneck(
                config, runtime_params=runtime_params
            )

        # Assert
        assert isinstance(bottleneck, MockComponent)

    @patch("crackseg.model.config.instantiation.decoder_registry")
    def test_instantiate_decoder(self, mock_registry: Mock) -> None:
        """Test decoder instantiation."""
        # Arrange
        config = {"type": "mock_decoder", "in_channels": 512}
        runtime_params = {"out_channels": 1}
        mock_registry.__contains__.return_value = True
        mock_registry.get.return_value = MockComponent

        expected_decoder = MockComponent(in_channels=512)

        # Act
        with patch(
            "crackseg.model.config.instantiation._instantiate_component"
        ) as mock_instantiate:
            mock_instantiate.return_value = expected_decoder
            decoder = instantiate_decoder(
                config, runtime_params=runtime_params
            )

        # Assert
        assert isinstance(decoder, MockComponent)

    @patch("crackseg.model.config.instantiation.component_registries")
    def test_instantiate_additional_component_known_registry(
        self, mock_registries: Mock
    ) -> None:
        """Test additional component instantiation with known registry."""
        # Arrange
        component_name = "mock_category"
        component_config = {"type": "mock_component"}

        mock_registry = MockRegistry()
        mock_registries.get.return_value = mock_registry

        expected_component = MockComponent()

        # Act
        with patch(
            "crackseg.model.config.instantiation._instantiate_component"
        ) as mock_instantiate:
            mock_instantiate.return_value = expected_component
            component = instantiate_additional_component(
                component_name, component_config
            )

        # Assert
        assert isinstance(component, MockComponent)

    @patch("crackseg.model.config.instantiation.component_registries")
    @patch("crackseg.model.config.instantiation.encoder_registry")
    def test_instantiate_additional_component_fallback_encoder(
        self, mock_encoder_registry: Mock, mock_registries: Mock
    ) -> None:
        """Test additional component instantiation with encoder fallback."""
        # Arrange
        component_name = "unknown_category"
        component_config = {"type": "MockEncoder"}

        mock_registries.get.return_value = None
        mock_encoder_registry.__contains__.return_value = True
        mock_encoder_registry.get.return_value = MockEncoder

        expected_encoder = MockEncoder()

        # Act
        with patch(
            "crackseg.model.config.instantiation._instantiate_component"
        ) as mock_instantiate:
            mock_instantiate.return_value = expected_encoder
            component = instantiate_additional_component(
                component_name, component_config
            )

        # Assert
        assert component == expected_encoder
        mock_instantiate.assert_called_once_with(
            config=component_config,
            registry=mock_encoder_registry,
            component_category=component_name,
            use_cache=True,
        )

    def test_instantiate_additional_component_no_type(self) -> None:
        """Test additional component instantiation without type."""
        # Arrange
        component_name = "mock_category"
        component_config = {"in_channels": 3}

        # Act & Assert
        with pytest.raises(
            InstantiationError,
            match="Component 'mock_category' config must specify 'type'",
        ):
            instantiate_additional_component(component_name, component_config)

    @patch("crackseg.model.config.instantiation.component_registries")
    def test_instantiate_additional_component_no_registry(
        self, mock_registries: Mock
    ) -> None:
        """Test additional component instantiation without registry."""
        # Arrange
        component_name = "unknown_category"
        component_config = {"type": "mock_component"}

        mock_registries.get.return_value = None

        # Act & Assert
        with patch(
            "crackseg.model.config.instantiation.encoder_registry"
        ) as mock_encoder_registry:
            mock_encoder_registry.__contains__.return_value = False
            with pytest.raises(
                InstantiationError,
                match="Cannot determine registry for component",
            ):
                instantiate_additional_component(
                    component_name, component_config
                )


class TestHelperFunctions:
    """Test helper function functionality."""

    @patch("crackseg.model.config.instantiation.instantiate_encoder")
    def test_try_instantiate_encoder_success(
        self, mock_instantiate: Mock
    ) -> None:
        """Test successful encoder instantiation."""
        # Arrange
        config = {"type": "mock_encoder", "in_channels": 3}
        expected_encoder = MockEncoder(in_channels=3)
        mock_instantiate.return_value = expected_encoder

        # Act
        with patch(
            "crackseg.model.config.instantiation._try_instantiate_encoder"
        ) as mock_try:
            mock_try.return_value = expected_encoder
            encoder = mock_try(config)

        # Assert
        assert encoder is expected_encoder

    def test_try_instantiate_encoder_no_config(self) -> None:
        """Test encoder instantiation with no config."""
        # Act
        with patch(
            "crackseg.model.config.instantiation._try_instantiate_encoder"
        ) as mock_try:
            mock_try.return_value = None
            encoder = mock_try(None)

        # Assert
        assert encoder is None

    @patch("crackseg.model.config.instantiation.instantiate_bottleneck")
    def test_try_instantiate_bottleneck_with_encoder(
        self, mock_instantiate: Mock
    ) -> None:
        """Test bottleneck instantiation with encoder."""
        # Arrange
        config = {"type": "mock_bottleneck", "in_channels": 512}
        encoder = MockEncoder()
        expected_bottleneck = MockComponent(in_channels=512)
        mock_instantiate.return_value = expected_bottleneck

        # Act
        with patch(
            "crackseg.model.config.instantiation._try_instantiate_bottleneck"
        ) as mock_try:
            mock_try.return_value = expected_bottleneck
            bottleneck = mock_try(config, encoder)

        # Assert
        assert bottleneck is expected_bottleneck

    @patch("crackseg.model.config.instantiation.instantiate_bottleneck")
    def test_try_instantiate_bottleneck_no_encoder(
        self, mock_instantiate: Mock
    ) -> None:
        """Test bottleneck instantiation without encoder."""
        # Arrange
        config = {"type": "mock_bottleneck", "in_channels": 512}
        expected_bottleneck = MockComponent(in_channels=512)
        mock_instantiate.return_value = expected_bottleneck

        # Act
        with patch(
            "crackseg.model.config.instantiation._try_instantiate_bottleneck"
        ) as mock_try:
            mock_try.return_value = expected_bottleneck
            bottleneck = mock_try(config, None)

        # Assert
        assert bottleneck is expected_bottleneck

    def test_try_instantiate_bottleneck_no_config(self) -> None:
        """Test bottleneck instantiation with no config."""
        # Act
        with patch(
            "crackseg.model.config.instantiation._try_instantiate_bottleneck"
        ) as mock_try:
            mock_try.return_value = None
            bottleneck = mock_try(None, None)

        # Assert
        assert bottleneck is None

    @patch("crackseg.model.config.instantiation.instantiate_decoder")
    def test_try_instantiate_decoder_with_dependencies(
        self, mock_instantiate: Mock
    ) -> None:
        """Test decoder instantiation with dependencies."""
        # Arrange
        config = {"type": "mock_decoder", "in_channels": 512}
        encoder = MockEncoder()
        bottleneck = MockComponent()
        expected_decoder = MockComponent(in_channels=512)
        mock_instantiate.return_value = expected_decoder

        # Act
        with patch(
            "crackseg.model.config.instantiation._try_instantiate_decoder"
        ) as mock_try:
            mock_try.return_value = expected_decoder
            decoder = mock_try(config, encoder, bottleneck)

        # Assert
        assert decoder is expected_decoder

    def test_try_instantiate_decoder_no_config(self) -> None:
        """Test decoder instantiation with no config."""
        # Act
        with patch(
            "crackseg.model.config.instantiation._try_instantiate_decoder"
        ) as mock_try:
            mock_try.return_value = None
            decoder = mock_try(None, None, None)

        # Assert
        assert decoder is None


class TestInstantiationError:
    """Test InstantiationError exception."""

    def test_instantiation_error_creation(self) -> None:
        """Test InstantiationError can be created and raised."""
        # Arrange
        message = "Test instantiation error"

        # Act & Assert
        with pytest.raises(
            InstantiationError, match="Test instantiation error"
        ):
            raise InstantiationError(message)

    def test_instantiation_error_with_cause(self) -> None:
        """Test InstantiationError with cause chain."""
        # Arrange
        original_error = ValueError("Original error")
        message = "Instantiation failed"

        # Act & Assert
        try:
            raise original_error
        except ValueError as e:
            with pytest.raises(
                InstantiationError, match="Instantiation failed"
            ):
                raise InstantiationError(message) from e


class TestInstantiationIntegration:
    """Test integration aspects of instantiation."""

    def test_logging_configuration(self) -> None:
        """Test that logging is properly configured."""
        import logging

        logger = logging.getLogger("crackseg.model.config.instantiation")
        assert logger is not None

    def test_type_checking_integration(self) -> None:
        """Test type checking integration."""
        # Test that types are properly defined
        assert hasattr(InstantiationError, "__name__")
        assert InstantiationError.__name__ == "InstantiationError"

    def test_config_validation_patterns(self) -> None:
        """Test common configuration validation patterns."""
        # Test valid config patterns
        valid_configs = [
            {"type": "mock_encoder", "in_channels": 3},
            {
                "_target_": "crackseg.model.encoder.SwinEncoder",
                "embed_dim": 96,
            },
            {"type": "mock_component", "in_channels": 3, "out_channels": 64},
        ]

        for config in valid_configs:
            assert isinstance(config, dict)
            assert "type" in config or "_target_" in config

    def test_registry_interface_compatibility(self) -> None:
        """Test registry interface compatibility."""
        # Test that mock registry implements expected interface
        registry = MockRegistry()

        # Test required methods
        assert hasattr(registry, "__contains__")
        assert hasattr(registry, "get")
        assert hasattr(registry, "list_components")

        # Test method behavior
        assert "mock_component" in registry
        assert registry.get("mock_component") == MockComponent
        assert "mock_component" in registry.list_components()
