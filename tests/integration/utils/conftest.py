"""
Integration test fixtures and utilities for the CrackSeg project.

Provides shared fixtures and utilities for integration testing across
different system components and workflows.
"""

import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from src.crackseg.utils.traceability import (
    AccessControl,
    LineageManager,
    MetadataManager,
    TraceabilityIntegrationManager,
    TraceabilityQueryInterface,
    TraceabilityStorage,
)


@pytest.fixture
def temp_test_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for integration tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup is automatic with tempfile


@pytest.fixture
def traceability_storage(temp_test_dir: Path) -> TraceabilityStorage:
    """Create a traceability storage instance for testing."""
    return TraceabilityStorage(temp_test_dir)


@pytest.fixture
def traceability_components(
    temp_test_dir: Path,
) -> dict[str, Any]:
    """Create all traceability system components for integration testing."""
    storage = TraceabilityStorage(temp_test_dir)
    return {
        "storage": storage,
        "metadata_manager": MetadataManager(storage),
        "access_control": AccessControl(storage),
        "lineage_manager": LineageManager(storage),
        "query_interface": TraceabilityQueryInterface(storage),
        "integration_manager": TraceabilityIntegrationManager(storage),
    }


@pytest.fixture
def sample_experiment_data() -> dict[str, Any]:
    """Provide sample experiment data for testing."""
    return {
        "experiment_id": "test-exp-001",
        "username": "test_user",
        "status": "completed",
        "metadata": {
            "objective": "crack_detection",
            "model_type": "unet",
            "dataset": "crack_dataset_v1",
            "hyperparameters": {
                "learning_rate": 0.001,
                "batch_size": 16,
                "epochs": 100,
            },
        },
    }


@pytest.fixture
def sample_artifact_data() -> dict[str, Any]:
    """Provide sample artifact data for testing."""
    return {
        "artifact_id": "test-model-001",
        "artifact_type": "model",
        "file_path": "/models/test_model.pth",
        "file_size": 1024000,
        "checksum": "a" * 64,
        "name": "Test Crack Detection Model",
        "owner": "test_user",
        "compliance_level": "standard",
        "metadata": {
            "accuracy": 0.95,
            "model_type": "unet",
            "input_channels": 3,
            "output_channels": 2,
            "training_epochs": 100,
            "loss": 0.05,
        },
    }


@pytest.fixture
def sample_lineage_data() -> dict[str, Any]:
    """Provide sample lineage data for testing."""
    return {
        "lineage_id": "test-lineage-001",
        "source_entity_type": "experiment",
        "source_entity_id": "test-exp-001",
        "target_entity_type": "artifact",
        "target_entity_id": "test-model-001",
        "relationship_type": "produces",
        "metadata": {
            "timestamp": "2024-01-01T00:00:00",
            "description": "Model produced by experiment",
        },
    }
