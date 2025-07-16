"""Tests for Hydra configuration loading and validation."""

import os

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


class TestHydraConfiguration:
    """Test suite for Hydra configuration loading and validation."""

    @pytest.fixture
    def config_dir(self) -> str:
        """Get absolute path to configs directory."""
        return os.path.abspath("configs")

    def test_config_directory_exists(self, config_dir: str) -> None:
        """Test that the configs directory exists."""
        assert os.path.exists(
            config_dir
        ), f"Config directory not found: {config_dir}"

    def test_basic_config_loading(self, config_dir: str) -> None:
        """Test basic Hydra configuration loading."""
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config.yaml")
            assert isinstance(
                cfg, DictConfig
            ), "Configuration should be a DictConfig"

    def test_model_configuration(self, config_dir: str) -> None:
        """Test model configuration sections."""
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config.yaml")

            # Check model section exists
            assert hasattr(
                cfg, "model"
            ), "Model configuration section not found"

            # Check model target
            assert hasattr(cfg.model, "_target_"), "Model _target_ not found"
            assert cfg.model._target_ == "crackseg.model.core.unet.BaseUNet"

            # Check encoder configuration
            assert hasattr(
                cfg.model, "encoder"
            ), "Model encoder configuration not found"
            assert hasattr(
                cfg.model.encoder, "_target_"
            ), "Encoder _target_ not found"
            assert (
                cfg.model.encoder._target_
                == "crackseg.model.encoder.CNNEncoder"
            )

    def test_training_configuration(self, config_dir: str) -> None:
        """Test training configuration sections."""
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config.yaml")

            # Check training section exists
            assert hasattr(
                cfg, "training"
            ), "Training configuration section not found"

            # Check basic training parameters
            assert hasattr(cfg.training, "epochs"), "Training epochs not found"
            assert isinstance(
                cfg.training.epochs, int
            ), "Epochs should be an integer"
            assert cfg.training.epochs > 0, "Epochs should be positive"

            assert hasattr(
                cfg.training, "learning_rate"
            ), "Learning rate not found"
            assert isinstance(
                cfg.training.learning_rate, float
            ), "Learning rate should be a float"
            assert (
                cfg.training.learning_rate > 0
            ), "Learning rate should be positive"

    def test_early_stopping_configuration(self, config_dir: str) -> None:
        """Test early stopping configuration."""
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config.yaml")

            # Early stopping should exist from trainer config
            if hasattr(cfg.training, "early_stopping"):
                assert hasattr(
                    cfg.training.early_stopping, "monitor"
                ), "Early stopping monitor not found"
                assert isinstance(
                    cfg.training.early_stopping.monitor, str
                ), "Monitor should be a string"

    def test_experiment_configuration(self, config_dir: str) -> None:
        """Test experiment configuration sections."""
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config.yaml")

            # Check experiment section exists
            assert hasattr(
                cfg, "experiment"
            ), "Experiment configuration section not found"
            assert hasattr(cfg.experiment, "name"), "Experiment name not found"
            assert isinstance(
                cfg.experiment.name, str
            ), "Experiment name should be a string"

    def test_data_configuration(self, config_dir: str) -> None:
        """Test data configuration sections."""
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config.yaml")

            # Check data section exists
            assert hasattr(cfg, "data"), "Data configuration section not found"
            assert hasattr(cfg.data, "root_dir"), "Data root_dir not found"
            assert isinstance(
                cfg.data.root_dir, str
            ), "Data root_dir should be a string"

    def test_configuration_completeness(self, config_dir: str) -> None:
        """Test that all major configuration sections are present."""
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config.yaml")

            required_sections = ["model", "training", "data", "experiment"]
            for section in required_sections:
                assert hasattr(
                    cfg, section
                ), f"Required configuration section '{section}' not found"


# If run as a script, execute the tests manually for verification
if __name__ == "__main__":
    config_dir = os.path.abspath("configs")
    print(f"Config dir: {config_dir}")
    print(f"Exists: {os.path.exists(config_dir)}")

    test_instance = TestHydraConfiguration()

    try:
        test_instance.test_config_directory_exists(config_dir)
        print("✅ Config directory exists")

        test_instance.test_basic_config_loading(config_dir)
        print("✅ Basic config loading")

        test_instance.test_model_configuration(config_dir)
        print("✅ Model configuration")

        test_instance.test_training_configuration(config_dir)
        print("✅ Training configuration")

        test_instance.test_early_stopping_configuration(config_dir)
        print("✅ Early stopping configuration")

        test_instance.test_experiment_configuration(config_dir)
        print("✅ Experiment configuration")

        test_instance.test_data_configuration(config_dir)
        print("✅ Data configuration")

        test_instance.test_configuration_completeness(config_dir)
        print("✅ Configuration completeness")

        print("✅ All Hydra configuration tests passed")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        exit(1)
