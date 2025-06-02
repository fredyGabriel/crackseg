"""
Integration tests for Model Factory → Training flow.

Tests the complete integration between model factory instantiation
and training pipeline, ensuring models created by the factory
can be properly used in training workflows.
"""

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.model.factory.factory import create_unet
from src.training.trainer import TrainingComponents


class MockDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Mock dataset for testing."""

    def __init__(self, size: int = 4) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.randn(3, 32, 32), torch.zeros(1, 32, 32)


class MockUNetBase(nn.Module):
    """Mock UNet for integration testing."""

    def __init__(
        self, encoder: nn.Module, bottleneck: nn.Module, decoder: nn.Module
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.out_channels = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple forward pass simulation
        batch_size = x.shape[0]
        height, width = x.shape[2], x.shape[3]
        return torch.zeros(batch_size, 1, height, width)


class MockEncoder(nn.Module):
    """Mock encoder for testing."""

    def __init__(self, in_channels: int = 3, **kwargs: Any) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, 64, 3, padding=1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = self.conv(x)
        skip_connections = [features]
        return features, skip_connections


class MockBottleneck(nn.Module):
    """Mock bottleneck for testing."""

    def __init__(self, in_channels: int = 64, **kwargs: Any) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, 128, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MockDecoder(nn.Module):
    """Mock decoder for testing."""

    def __init__(self, in_channels: int = 128, **kwargs: Any) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 1
        self.conv = nn.Conv2d(in_channels, 1, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        skip_connections: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return self.conv(x)


class TestModelFactoryTrainingIntegration:
    """Test integration between model factory and training components."""

    @pytest.fixture
    def base_model_config(self) -> DictConfig:
        """Create a base model configuration for testing."""
        config_dict = {
            "encoder": {
                "type": "MockEncoder",
                "in_channels": 3,
            },
            "bottleneck": {
                "type": "MockBottleneck",
                "in_channels": 64,
            },
            "decoder": {
                "type": "MockDecoder",
                "in_channels": 128,
            },
            "unet_class": "MockUNetBase",
        }
        return OmegaConf.create(config_dict)

    @pytest.fixture
    def training_config(self) -> DictConfig:
        """Create a training configuration for testing."""
        config_dict = {
            "training": {
                "epochs": 2,
                "device": "cpu",
                "use_amp": False,
                "gradient_accumulation_steps": 1,
                "verbose": False,
                "save_freq": 0,
                "optimizer": {
                    "_target_": "torch.optim.Adam",
                    "lr": 0.001,
                },
                "scheduler": {
                    "_target_": "torch.optim.lr_scheduler.StepLR",
                    "step_size": 1,
                    "gamma": 0.9,
                },
            },
            "checkpoints": {
                "enabled": True,
                "dir": "outputs/test_checkpoints",
            },
        }
        return OmegaConf.create(config_dict)

    @pytest.fixture
    def mock_data_loaders(self) -> tuple[DataLoader[Any], DataLoader[Any]]:
        """Create mock data loaders for testing."""
        # Create proper mock dataset
        mock_dataset = MockDataset(size=4)

        train_loader = DataLoader(mock_dataset, batch_size=2, shuffle=False)
        val_loader = DataLoader(mock_dataset, batch_size=2, shuffle=False)

        return train_loader, val_loader

    @pytest.fixture
    def mock_loss_function(self) -> nn.Module:
        """Create a mock loss function."""
        return nn.BCEWithLogitsLoss()

    @pytest.fixture
    def mock_metrics(self) -> dict[str, Any]:
        """Create mock metrics for testing."""
        return {"accuracy": MagicMock(), "iou": MagicMock()}

    @patch("src.model.factory.factory.get_unet_class")
    @patch("src.model.factory.factory.instantiate_encoder")
    @patch("src.model.factory.factory.instantiate_bottleneck")
    @patch("src.model.factory.factory.instantiate_decoder")
    def test_model_factory_creates_trainable_model(
        self,
        mock_instantiate_decoder: Mock,
        mock_instantiate_bottleneck: Mock,
        mock_instantiate_encoder: Mock,
        mock_get_unet_class: Mock,
        base_model_config: DictConfig,
    ) -> None:
        """Test that model factory creates a model compatible with training."""
        # Arrange
        mock_encoder = MockEncoder()
        mock_bottleneck = MockBottleneck()
        mock_decoder = MockDecoder()

        mock_instantiate_encoder.return_value = mock_encoder
        mock_instantiate_bottleneck.return_value = mock_bottleneck
        mock_instantiate_decoder.return_value = mock_decoder
        mock_get_unet_class.return_value = MockUNetBase

        # Act
        model = create_unet(base_model_config)

        # Assert
        assert isinstance(model, nn.Module)
        assert hasattr(model, "forward")

        # Test that model can process input
        test_input = torch.randn(1, 3, 32, 32)
        output = model(test_input)
        assert output.shape == (1, 1, 32, 32)

    @patch("src.model.factory.factory.get_unet_class")
    @patch("src.model.factory.factory.instantiate_encoder")
    @patch("src.model.factory.factory.instantiate_bottleneck")
    @patch("src.model.factory.factory.instantiate_decoder")
    def test_complete_factory_to_training_flow(
        self,
        mock_instantiate_decoder: Mock,
        mock_instantiate_bottleneck: Mock,
        mock_instantiate_encoder: Mock,
        mock_get_unet_class: Mock,
        base_model_config: DictConfig,
        training_config: DictConfig,
        mock_data_loaders: tuple[DataLoader[Any], DataLoader[Any]],
        mock_loss_function: nn.Module,
        mock_metrics: dict[str, Any],
    ) -> None:
        """Test complete flow from model factory to training initialization."""
        # Arrange
        mock_encoder = MockEncoder()
        mock_bottleneck = MockBottleneck()
        mock_decoder = MockDecoder()

        mock_instantiate_encoder.return_value = mock_encoder
        mock_instantiate_bottleneck.return_value = mock_bottleneck
        mock_instantiate_decoder.return_value = mock_decoder
        mock_get_unet_class.return_value = MockUNetBase

        train_loader, val_loader = mock_data_loaders

        # Act - Create model using factory
        model = create_unet(base_model_config)

        # Act - Initialize training components
        components = TrainingComponents(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=mock_loss_function,
            metrics_dict=mock_metrics,
        )

        # Act - Test that components integrate properly
        # Forward pass test
        sample_batch = next(iter(train_loader))
        inputs, targets = sample_batch

        # Test model forward pass with training data
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)

        # Test loss calculation
        loss = mock_loss_function(outputs, targets)

        # Assert
        assert components.model is model
        assert isinstance(components.model, nn.Module)
        assert components.train_loader is train_loader
        assert components.val_loader is val_loader
        assert components.loss_fn is mock_loss_function
        assert components.metrics_dict is mock_metrics

        # Assert integration works
        assert outputs.shape == targets.shape
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0.0  # Loss should be non-negative

    @patch("src.model.factory.factory.get_unet_class")
    @patch("src.model.factory.factory.instantiate_encoder")
    @patch("src.model.factory.factory.instantiate_bottleneck")
    @patch("src.model.factory.factory.instantiate_decoder")
    def test_model_forward_pass_integration(
        self,
        mock_instantiate_decoder: Mock,
        mock_instantiate_bottleneck: Mock,
        mock_instantiate_encoder: Mock,
        mock_get_unet_class: Mock,
        base_model_config: DictConfig,
    ) -> None:
        """Test that factory-created model integrates properly."""
        # Arrange
        mock_encoder = MockEncoder()
        mock_bottleneck = MockBottleneck()
        mock_decoder = MockDecoder()

        mock_instantiate_encoder.return_value = mock_encoder
        mock_instantiate_bottleneck.return_value = mock_bottleneck
        mock_instantiate_decoder.return_value = mock_decoder
        mock_get_unet_class.return_value = MockUNetBase

        # Act
        model = create_unet(base_model_config)

        # Test various input sizes
        test_cases = [
            (1, 3, 32, 32),
            (2, 3, 64, 64),
            (4, 3, 128, 128),
        ]

        for batch_size, channels, height, width in test_cases:
            test_input = torch.randn(batch_size, channels, height, width)

            # Act
            output = model(test_input)

            # Assert
            assert output.shape == (batch_size, 1, height, width)
            assert not torch.isnan(output).any()
            assert output.dtype == torch.float32

    @patch("src.model.factory.factory.get_unet_class")
    @patch("src.model.factory.factory.instantiate_encoder")
    @patch("src.model.factory.factory.instantiate_bottleneck")
    @patch("src.model.factory.factory.instantiate_decoder")
    def test_model_factory_error_handling_integration(
        self,
        mock_instantiate_decoder: Mock,
        mock_instantiate_bottleneck: Mock,
        mock_instantiate_encoder: Mock,
        mock_get_unet_class: Mock,
        base_model_config: DictConfig,
    ) -> None:
        """Test error handling in model factory integration."""
        # Arrange - Simulate instantiation failure
        mock_instantiate_encoder.side_effect = RuntimeError(
            "Encoder creation failed"
        )

        # Act & Assert
        from src.model.factory.factory_utils import ConfigurationError

        with pytest.raises(
            ConfigurationError, match="Error instantiating UNet model"
        ):
            create_unet(base_model_config)

    def test_config_integration_validation(
        self, base_model_config: DictConfig
    ) -> None:
        """Test that configuration validation works in integration flow."""
        # Test missing required components
        incomplete_config = OmegaConf.create(
            {
                "encoder": {"type": "MockEncoder"},
                # Missing bottleneck and decoder
            }
        )

        from src.model.factory.factory_utils import ConfigurationError

        with pytest.raises(ConfigurationError):
            create_unet(incomplete_config)
