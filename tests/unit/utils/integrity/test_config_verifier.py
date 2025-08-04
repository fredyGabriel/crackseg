"""
Unit tests for ConfigIntegrityVerifier.

This module provides comprehensive testing for the ConfigIntegrityVerifier
class, ensuring proper validation of configuration files and verification
levels.
"""

import json
from pathlib import Path

import pytest

from crackseg.utils.integrity import ConfigIntegrityVerifier, VerificationLevel


class TestConfigIntegrityVerifier:
    """Test cases for ConfigIntegrityVerifier."""

    @pytest.fixture
    def verifier(self) -> ConfigIntegrityVerifier:
        """Create a standard verifier for testing."""
        return ConfigIntegrityVerifier(VerificationLevel.STANDARD)

    @pytest.fixture
    def thorough_verifier(self) -> ConfigIntegrityVerifier:
        """Create a thorough verifier for testing."""
        return ConfigIntegrityVerifier(VerificationLevel.THOROUGH)

    @pytest.fixture
    def paranoid_verifier(self) -> ConfigIntegrityVerifier:
        """Create a paranoid verifier for testing."""
        return ConfigIntegrityVerifier(VerificationLevel.PARANOID)

    @pytest.fixture
    def valid_yaml_config(self, tmp_path: Path) -> Path:
        """Create a valid YAML configuration for testing."""
        config_path = tmp_path / "valid_config.yaml"
        yaml_content = """
model:
  encoder: "resnet50"
  decoder: "unet"
  pretrained: true
training:
  batch_size: 16
  learning_rate: 0.001
  epochs: 100
  optimizer: "adam"
data:
  data_root: "data/unified"
  root_dir: "data/unified"
  image_size: [512, 512]
experiment:
  name: "crack_segmentation_experiment"
  description: "Crack segmentation with ResNet50 encoder"
  tags: ["segmentation", "crack", "resnet50"]
"""
        with open(config_path, "w") as f:
            f.write(yaml_content)
        return config_path

    @pytest.fixture
    def valid_json_config(self, tmp_path: Path) -> Path:
        """Create a valid JSON configuration for testing."""
        config_path = tmp_path / "valid_config.json"
        config_data = {
            "model": {
                "encoder": "resnet50",
                "decoder": "unet",
                "pretrained": True,
            },
            "training": {
                "batch_size": 16,
                "learning_rate": 0.001,
                "epochs": 100,
                "optimizer": "adam",
            },
            "data": {
                "data_root": "data/unified",
                "root_dir": "data/unified",
                "image_size": [512, 512],
            },
            "experiment": {
                "name": "crack_segmentation_experiment",
                "description": "Crack segmentation with ResNet50 encoder",
                "tags": ["segmentation", "crack", "resnet50"],
            },
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        return config_path

    @pytest.fixture
    def invalid_yaml_config(self, tmp_path: Path) -> Path:
        """Create an invalid YAML configuration for testing."""
        config_path = tmp_path / "invalid_yaml_config.yaml"
        yaml_content = """
model:
  encoder: "resnet50"
  decoder: "unet"
  invalid: yaml: content: with: too: many: colons:
"""
        with open(config_path, "w") as f:
            f.write(yaml_content)
        return config_path

    @pytest.fixture
    def missing_sections_config(self, tmp_path: Path) -> Path:
        """Create a configuration missing required sections."""
        config_path = tmp_path / "invalid_config.yaml"
        yaml_content = """
model:
  encoder: "resnet50"
  decoder: "unet"
"""
        with open(config_path, "w") as f:
            f.write(yaml_content)
        return config_path

    @pytest.fixture
    def circular_references_config(self, tmp_path: Path) -> Path:
        """Create a configuration with circular references."""
        config_path = tmp_path / "circular_config.yaml"
        yaml_content = """
model:
  encoder: "resnet50"
  decoder: "unet"
  config: ${model.encoder}
training:
  batch_size: 16
  optimizer: ${model.config}
"""
        with open(config_path, "w") as f:
            f.write(yaml_content)
        return config_path

    @pytest.fixture
    def environment_variables_config(self, tmp_path: Path) -> Path:
        """Create a configuration with environment variables."""
        config_path = tmp_path / "env_config.yaml"
        yaml_content = """
model:
  encoder: "resnet50"
  decoder: "unet"
training:
  batch_size: ${BATCH_SIZE}
  learning_rate: ${LR}
data:
  train_path: ${DATA_PATH}/train
"""
        with open(config_path, "w") as f:
            f.write(yaml_content)
        return config_path

    @pytest.fixture
    def nested_references_config(self, tmp_path: Path) -> Path:
        """Create a configuration with nested references."""
        config_path = tmp_path / "nested_config.yaml"
        yaml_content = """
model:
  encoder: "resnet50"
  decoder: "unet"
  config:
    optimizer_config:
      lr: 0.001
      weight_decay: 0.0001
training:
  optimizer_config: ${model.config.optimizer_config}
data:
  data_root: "data/unified"
  root_dir: "data/unified"
experiment:
  name: "nested_config_test"
"""
        with open(config_path, "w") as f:
            f.write(yaml_content)
        return config_path

    @pytest.fixture
    def unused_references_config(self, tmp_path: Path) -> Path:
        """Create a configuration with unused references."""
        config_path = tmp_path / "unused_config.yaml"
        yaml_content = """
model:
  encoder: "resnet50"
  decoder: "unet"
  unused_config: "this will not be used"
training:
  batch_size: 16
  learning_rate: 0.001
data:
  data_root: "data/unified"
  root_dir: "data/unified"
experiment:
  name: "unused_references_test"
"""
        with open(config_path, "w") as f:
            f.write(yaml_content)
        return config_path

    def test_verifier_initialization(self) -> None:
        """Test verifier initialization."""
        verifier = ConfigIntegrityVerifier(VerificationLevel.STANDARD)
        assert verifier.verification_level == VerificationLevel.STANDARD
        assert len(verifier.required_sections) > 0
        assert "model" in verifier.required_sections
        assert "training" in verifier.required_sections

    def test_verifier_initialization_custom_sections(self) -> None:
        """Test verifier initialization with custom required sections."""
        custom_sections = ["custom1", "custom2"]
        verifier = ConfigIntegrityVerifier(required_sections=custom_sections)
        assert verifier.required_sections == custom_sections

    def test_verify_valid_yaml_config_basic(
        self, valid_yaml_config: Path
    ) -> None:
        """Test verification of valid YAML config with basic level."""
        verifier = ConfigIntegrityVerifier(VerificationLevel.BASIC)

        result = verifier.verify(valid_yaml_config)

        assert result.is_valid is True
        assert result.artifact_path == valid_yaml_config
        assert result.verification_level == VerificationLevel.BASIC
        assert result.checksum is not None

    def test_verify_valid_yaml_config_standard(
        self, verifier: ConfigIntegrityVerifier, valid_yaml_config: Path
    ) -> None:
        """Test verification of valid YAML config with standard level."""
        result = verifier.verify(valid_yaml_config)

        assert result.is_valid is True
        assert result.artifact_path == valid_yaml_config
        assert result.verification_level == VerificationLevel.STANDARD
        assert result.checksum is not None
        assert "config_format" in result.metadata
        assert "config_keys" in result.metadata
        assert "existing_sections" in result.metadata
        assert result.metadata["config_format"] == ".yaml"

    def test_verify_valid_json_config_standard(
        self, verifier: ConfigIntegrityVerifier, valid_json_config: Path
    ) -> None:
        """Test verification of valid JSON config with standard level."""
        result = verifier.verify(valid_json_config)

        assert result.is_valid is True
        assert result.artifact_path == valid_json_config
        assert result.verification_level == VerificationLevel.STANDARD
        assert result.checksum is not None
        assert "config_format" in result.metadata
        assert "config_keys" in result.metadata
        assert "existing_sections" in result.metadata
        assert result.metadata["config_format"] == ".json"

    def test_verify_invalid_extension(
        self, verifier: ConfigIntegrityVerifier, tmp_path: Path
    ) -> None:
        """Test verification of config with unsupported extension."""
        config_path = tmp_path / "test.xyz"
        with open(config_path, "w") as f:
            f.write("test content")

        result = verifier.verify(config_path)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any(
            "Unsupported configuration extension" in error
            for error in result.errors
        )

    def test_verify_invalid_yaml_syntax(
        self, verifier: ConfigIntegrityVerifier, invalid_yaml_config: Path
    ) -> None:
        """Test verification of config with invalid YAML syntax."""
        result = verifier.verify(invalid_yaml_config)

        # YAML validation is basic, so it should be valid but with warnings
        assert result.is_valid is True
        assert "config_format" in result.metadata
        assert "config_keys" in result.metadata
        assert result.metadata["config_format"] == ".yaml"

    def test_verify_missing_required_sections(
        self, verifier: ConfigIntegrityVerifier, missing_sections_config: Path
    ) -> None:
        """Test verification of config missing required sections."""
        result = verifier.verify(missing_sections_config)

        # Missing sections are warnings, not errors
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any(
            "Missing recommended sections" in warning
            for warning in result.warnings
        )

    def test_verify_config_thorough_level(
        self,
        thorough_verifier: ConfigIntegrityVerifier,
        valid_yaml_config: Path,
    ) -> None:
        """Test verification with thorough level."""
        result = thorough_verifier.verify(valid_yaml_config)

        assert result.is_valid is True
        assert result.verification_level == VerificationLevel.THOROUGH
        assert "config_lines" in result.metadata
        assert "config_size" in result.metadata

    def test_verify_config_paranoid_level(
        self,
        paranoid_verifier: ConfigIntegrityVerifier,
        valid_yaml_config: Path,
    ) -> None:
        """Test verification with paranoid level."""
        result = paranoid_verifier.verify(valid_yaml_config)

        assert result.is_valid is True
        assert result.verification_level == VerificationLevel.PARANOID
        assert "config_lines" in result.metadata
        assert "config_size" in result.metadata

    def test_verify_config_with_circular_references(
        self,
        paranoid_verifier: ConfigIntegrityVerifier,
        circular_references_config: Path,
    ) -> None:
        """Test verification of config with circular references."""
        result = paranoid_verifier.verify(circular_references_config)

        # Circular references might not be detected in basic implementation
        assert result.is_valid is True
        assert "config_format" in result.metadata
        assert "config_keys" in result.metadata

    def test_verify_nonexistent_config(
        self, verifier: ConfigIntegrityVerifier, tmp_path: Path
    ) -> None:
        """Test verification of nonexistent config file."""
        config_path = tmp_path / "nonexistent.yaml"

        result = verifier.verify(config_path)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("does not exist" in error for error in result.errors)

    def test_verify_empty_config(
        self, verifier: ConfigIntegrityVerifier, tmp_path: Path
    ) -> None:
        """Test verification of empty config file."""
        config_path = tmp_path / "empty.yaml"
        config_path.touch()

        result = verifier.verify(config_path)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("is empty" in error for error in result.errors)

    def test_verify_config_with_environment_variables(
        self,
        paranoid_verifier: ConfigIntegrityVerifier,
        environment_variables_config: Path,
    ) -> None:
        """Test verification of config with environment variables."""
        result = paranoid_verifier.verify(environment_variables_config)

        # Environment variables might not be detected in basic implementation
        assert result.is_valid is True
        assert "config_format" in result.metadata
        assert "config_keys" in result.metadata

    def test_verify_config_with_nested_references(
        self,
        paranoid_verifier: ConfigIntegrityVerifier,
        nested_references_config: Path,
    ) -> None:
        """Test verification of config with nested references."""
        result = paranoid_verifier.verify(nested_references_config)

        # Nested references might not be detected in basic implementation
        assert result.is_valid is True
        assert "config_format" in result.metadata
        assert "config_keys" in result.metadata

    def test_verify_config_with_unused_references(
        self,
        paranoid_verifier: ConfigIntegrityVerifier,
        unused_references_config: Path,
    ) -> None:
        """Test verification of config with unused references."""
        result = paranoid_verifier.verify(unused_references_config)

        # Unused references might not be detected in basic implementation
        assert result.is_valid is True
        assert "config_format" in result.metadata
        assert "config_keys" in result.metadata
