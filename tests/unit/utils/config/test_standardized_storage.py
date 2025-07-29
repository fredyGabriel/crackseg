"""Tests for standardized configuration storage utilities."""

import json
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from crackseg.utils.config.standardized_storage import (
    ConfigurationSchema,
    StandardizedConfigStorage,
    compare_configurations,
    create_configuration_backup,
    enrich_configuration_with_environment,
    generate_environment_metadata,
    migrate_legacy_configuration,
    validate_configuration_completeness,
)


@pytest.fixture
def sample_config() -> DictConfig:
    """Create a sample configuration for testing."""
    return OmegaConf.create(
        {
            "experiment": {"name": "test_experiment"},
            "model": {"_target_": "crackseg.model.UNet"},
            "training": {
                "epochs": 10,
                "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
            },
            "data": {"root_dir": "data/", "batch_size": 16},
            "random_seed": 42,
        }
    )


@pytest.fixture
def incomplete_config() -> DictConfig:
    """Create an incomplete configuration for testing validation."""
    return OmegaConf.create(
        {
            "model": {"_target_": "crackseg.model.UNet"},
            "training": {"epochs": 10},
        }
    )


@pytest.fixture
def temp_storage_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for storage testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestConfigurationSchema:
    """Test configuration schema definition."""

    def test_schema_initialization(self) -> None:
        """Test schema initialization with default fields."""
        schema = ConfigurationSchema()

        assert "experiment.name" in schema.required_fields
        assert "model._target_" in schema.required_fields
        assert "training.epochs" in schema.required_fields
        assert "random_seed" in schema.required_fields

        assert "training.learning_rate" in schema.recommended_fields
        assert "data.batch_size" in schema.recommended_fields

        assert "environment.pytorch_version" in schema.environment_fields
        assert "environment.timestamp" in schema.environment_fields


class TestEnvironmentMetadata:
    """Test environment metadata generation."""

    def test_generate_environment_metadata(self) -> None:
        """Test generation of environment metadata."""
        metadata = generate_environment_metadata()

        assert "pytorch_version" in metadata
        assert "python_version" in metadata
        assert "platform" in metadata
        assert "cuda_available" in metadata
        assert "timestamp" in metadata

        assert metadata["pytorch_version"] == torch.__version__
        assert isinstance(metadata["cuda_available"], bool)
        assert "." in metadata["python_version"]  # Version format

    def test_cuda_metadata_when_available(self) -> None:
        """Test CUDA metadata when CUDA is available."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
            patch("torch.cuda.get_device_name", return_value="RTX 3080"),
        ):
            metadata = generate_environment_metadata()

            assert (
                metadata["cuda_version"] == "available"
            )  # Simplified approach
            assert metadata["cuda_device_count"] == "2"  # String format
            assert metadata["cuda_device_name"] == "RTX 3080"

    def test_cuda_metadata_when_unavailable(self) -> None:
        """Test CUDA metadata when CUDA is not available."""
        with patch("torch.cuda.is_available", return_value=False):
            metadata = generate_environment_metadata()

            assert (
                metadata["cuda_version"] == "not_available"
            )  # Simplified approach
            assert metadata["cuda_device_count"] == "0"  # String format
            assert (
                metadata["cuda_device_name"] == "not_available"
            )  # Simplified approach


class TestConfigurationValidation:
    """Test configuration validation functionality."""

    def test_validate_complete_configuration(
        self, sample_config: DictConfig
    ) -> None:
        """Test validation of complete configuration."""
        result = validate_configuration_completeness(sample_config)

        assert result["is_valid"] is True
        assert len(result["missing_required"]) == 0
        # Check for expected fields in validation result
        assert "completeness_score" in result
        assert "missing_recommended" in result
        assert "missing_environment" in result

    def test_validate_incomplete_configuration(
        self, incomplete_config: DictConfig
    ) -> None:
        """Test validation of incomplete configuration."""
        result = validate_configuration_completeness(incomplete_config)

        assert result["is_valid"] is False
        assert "experiment.name" in result["missing_required"]
        assert "random_seed" in result["missing_required"]
        assert len(result["missing_recommended"]) > 0

    def test_strict_validation_raises_error(
        self, incomplete_config: DictConfig
    ) -> None:
        """Test that strict validation raises error for incomplete config."""
        with pytest.raises(
            ValueError, match="Configuration missing required fields"
        ):
            validate_configuration_completeness(incomplete_config, strict=True)

    def test_custom_schema_validation(self, sample_config: DictConfig) -> None:
        """Test validation with custom schema."""
        custom_schema = ConfigurationSchema()
        custom_schema.required_fields = ["experiment.name", "custom.field"]

        result = validate_configuration_completeness(
            sample_config, custom_schema
        )

        assert result["is_valid"] is False
        assert "custom.field" in result["missing_required"]


class TestConfigurationEnrichment:
    """Test configuration enrichment with environment data."""

    def test_enrich_with_environment(self, sample_config: DictConfig) -> None:
        """Test enriching configuration with environment metadata."""
        enriched = enrich_configuration_with_environment(sample_config)

        # Original config should be preserved
        assert enriched.experiment.name == "test_experiment"
        assert enriched.training.epochs == 10

        # Environment metadata should be added
        assert hasattr(enriched, "environment")
        assert "pytorch_version" in enriched.environment
        assert "timestamp" in enriched.environment

        # Config metadata should be added
        assert hasattr(enriched, "config_metadata")
        assert (
            enriched.config_metadata.format_version == "1.0"
        )  # Correct field name
        assert "config_hash" in enriched.config_metadata

    def test_enrich_without_environment(
        self, sample_config: DictConfig
    ) -> None:
        """Test enriching configuration without environment metadata."""
        enriched = enrich_configuration_with_environment(
            sample_config, include_environment=False
        )

        assert not hasattr(enriched, "environment")
        assert hasattr(enriched, "config_metadata")

    def test_config_hash_consistency(self, sample_config: DictConfig) -> None:
        """Test that config hash is consistent for same configuration."""
        enriched1 = enrich_configuration_with_environment(
            sample_config, include_environment=False
        )
        enriched2 = enrich_configuration_with_environment(
            sample_config, include_environment=False
        )

        # Hashes should be the same for same config (excluding timestamps)
        # Note: We exclude environment to avoid timestamp differences
        assert (
            enriched1.config_metadata.config_hash
            == enriched2.config_metadata.config_hash
        )


class TestStandardizedConfigStorage:
    """Test standardized configuration storage manager."""

    def test_storage_initialization(self, temp_storage_dir: Path) -> None:
        """Test storage manager initialization."""
        storage = StandardizedConfigStorage(temp_storage_dir)

        assert storage.base_dir == temp_storage_dir
        assert storage.format_version == "1.0"
        assert storage.include_environment is True
        assert storage.validate_on_save is True
        assert temp_storage_dir.exists()

    def test_save_and_load_yaml_configuration(
        self, temp_storage_dir: Path, sample_config: DictConfig
    ) -> None:
        """Test saving and loading configuration in YAML format."""
        storage = StandardizedConfigStorage(temp_storage_dir)
        experiment_id = "test_exp_001"

        # Save configuration
        saved_path = storage.save_configuration(
            sample_config, experiment_id, format_type="yaml"
        )

        assert saved_path.exists()
        assert saved_path.suffix == ".yaml"

        # Load configuration
        loaded_config = storage.load_configuration(experiment_id)

        assert loaded_config.experiment.name == "test_experiment"
        assert loaded_config.training.epochs == 10
        assert hasattr(loaded_config, "environment")  # Should be enriched

    def test_save_and_load_json_configuration(
        self, temp_storage_dir: Path, sample_config: DictConfig
    ) -> None:
        """Test saving and loading configuration in JSON format."""
        storage = StandardizedConfigStorage(temp_storage_dir)
        experiment_id = "test_exp_002"

        # Save configuration
        saved_path = storage.save_configuration(
            sample_config, experiment_id, format_type="json"
        )

        assert saved_path.exists()
        assert saved_path.suffix == ".json"

        # Load configuration
        loaded_config = storage.load_configuration(experiment_id)

        assert loaded_config.experiment.name == "test_experiment"
        assert hasattr(loaded_config, "environment")

    def test_validation_report_creation(
        self, temp_storage_dir: Path, incomplete_config: DictConfig
    ) -> None:
        """
        Test that validation report is created for invalid configurations.
        """
        storage = StandardizedConfigStorage(
            temp_storage_dir, validate_on_save=True
        )
        experiment_id = "test_exp_003"

        # Save incomplete configuration
        storage.save_configuration(incomplete_config, experiment_id)

        # Check validation report exists
        validation_file = (
            temp_storage_dir / experiment_id / "config_validation.json"
        )
        assert validation_file.exists()

        with open(validation_file) as f:
            validation_data = json.load(f)

        assert validation_data["is_valid"] is False
        assert len(validation_data["missing_required"]) > 0

    def test_list_experiments(
        self, temp_storage_dir: Path, sample_config: DictConfig
    ) -> None:
        """Test listing available experiments."""
        storage = StandardizedConfigStorage(temp_storage_dir)

        # Initially empty
        assert storage.list_experiments() == []

        # Save some experiments
        storage.save_configuration(sample_config, "exp_001")
        storage.save_configuration(sample_config, "exp_002")

        experiments = storage.list_experiments()
        assert "exp_001" in experiments
        assert "exp_002" in experiments
        assert len(experiments) == 2

    def test_get_experiment_metadata(
        self, temp_storage_dir: Path, sample_config: DictConfig
    ) -> None:
        """Test getting experiment metadata."""
        storage = StandardizedConfigStorage(temp_storage_dir)
        experiment_id = "test_exp_004"

        storage.save_configuration(sample_config, experiment_id)
        metadata = storage.get_experiment_metadata(experiment_id)

        assert metadata["experiment_id"] == experiment_id
        assert metadata["experiment_name"] == "test_experiment"
        assert "created_at" in metadata
        assert "config_hash" in metadata
        assert "pytorch_version" in metadata

    def test_load_nonexistent_configuration(
        self, temp_storage_dir: Path
    ) -> None:
        """Test loading configuration that doesn't exist."""
        storage = StandardizedConfigStorage(temp_storage_dir)

        with pytest.raises(FileNotFoundError):
            storage.load_configuration("nonexistent_exp")


class TestConfigurationComparison:
    """Test configuration comparison functionality."""

    def test_compare_identical_configurations(
        self, sample_config: DictConfig
    ) -> None:
        """Test comparison of identical configurations."""
        config1 = enrich_configuration_with_environment(sample_config)
        config2 = enrich_configuration_with_environment(sample_config)

        result = compare_configurations(config1, config2)

        assert result["are_identical"] is True
        assert result["total_differences"] == 0
        assert len(result["differences"]) == 0

    def test_compare_different_configurations(
        self, sample_config: DictConfig
    ) -> None:
        """Test comparison of different configurations."""
        config1 = sample_config.copy()
        config2 = sample_config.copy()

        # Use OmegaConf.update to safely modify the configuration
        OmegaConf.update(config2, "training.epochs", 20)

        result = compare_configurations(config1, config2)

        assert result["are_identical"] is False
        assert result["total_differences"] == 1

    def test_ignore_fields_in_comparison(
        self, sample_config: DictConfig
    ) -> None:
        """Test comparison with ignored fields."""
        config1 = sample_config.copy()
        config2 = sample_config.copy()

        # Modify two fields
        OmegaConf.update(config2, "training.epochs", 20)
        OmegaConf.update(config2, "data.batch_size", 64)

        # Compare ignoring one field
        result = compare_configurations(
            config1, config2, ignore_fields=["training.epochs"]
        )

        assert result["are_identical"] is False
        assert (
            result["total_differences"] == 1
        )  # Only data.batch_size should differ


class TestConfigurationBackup:
    """Test configuration backup functionality."""

    def test_create_configuration_backup(
        self, temp_storage_dir: Path, sample_config: DictConfig
    ) -> None:
        """Test creating configuration backup."""
        backup_dir = temp_storage_dir / "backups"
        experiment_id = "test_exp_backup"

        backup_path = create_configuration_backup(
            sample_config, backup_dir, experiment_id
        )

        assert backup_path.exists()
        assert backup_path.suffix == ".yaml"
        assert experiment_id in backup_path.name
        assert "config_backup" in backup_path.name

        # Load and verify backup
        backup_config = OmegaConf.load(backup_path)
        assert backup_config.experiment.name == "test_experiment"
        assert hasattr(backup_config, "environment")  # Should be enriched


class TestLegacyMigration:
    """Test legacy configuration migration."""

    def test_migrate_legacy_dict_configuration(self) -> None:
        """Test migrating legacy dictionary configuration."""
        legacy_config = {
            "model": {"_target_": "crackseg.model.UNet"},
            "training": {"epochs": 10, "lr": 0.001},
        }

        migrated = migrate_legacy_configuration(legacy_config)

        # Should preserve original fields
        assert migrated.model._target_ == "crackseg.model.UNet"
        assert migrated.training.epochs == 10
        assert migrated.training.lr == 0.001

        # Should have environment metadata
        assert hasattr(migrated, "environment")

        # Should have migration metadata
        assert hasattr(migrated, "migration_metadata")
        assert migrated.migration_metadata.source_format == "legacy"
        assert migrated.migration_metadata.target_format == "standardized_v1.0"

    def test_migrate_legacy_dictconfig(
        self, incomplete_config: DictConfig
    ) -> None:
        """Test migrating legacy DictConfig."""
        migrated = migrate_legacy_configuration(incomplete_config)

        # Should preserve original fields
        assert hasattr(migrated, "model")
        assert hasattr(migrated, "training")

        # Should have environment metadata
        assert hasattr(migrated, "environment")

        # Should have migration metadata
        assert hasattr(migrated, "migration_metadata")


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple features."""

    def test_full_workflow_scenario(
        self, temp_storage_dir: Path, sample_config: DictConfig
    ) -> None:
        """Test complete workflow with multiple configurations."""
        storage = StandardizedConfigStorage(temp_storage_dir)

        # Save first experiment
        exp1_id = "experiment_001"
        storage.save_configuration(sample_config, exp1_id)

        # Modify config and save second experiment
        modified_config = sample_config.copy()
        OmegaConf.update(modified_config, "training.epochs", 50)

        exp2_id = "experiment_002"
        storage.save_configuration(modified_config, exp2_id)

        # Load and compare
        loaded_config1 = storage.load_configuration(exp1_id)
        loaded_config2 = storage.load_configuration(exp2_id)

        assert loaded_config1.training.epochs == 10
        assert loaded_config2.training.epochs == 50

        # Test experiment listing
        experiments = storage.list_experiments()
        assert len(experiments) == 2
        assert exp1_id in experiments
        assert exp2_id in experiments

        # Test metadata retrieval
        metadata1 = storage.get_experiment_metadata(exp1_id)
        assert metadata1["experiment_id"] == exp1_id
