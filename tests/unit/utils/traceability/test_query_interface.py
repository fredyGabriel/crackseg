"""Unit tests for TraceabilityQueryInterface."""

from pathlib import Path

import pytest

from src.crackseg.utils.traceability import TraceabilityQueryInterface
from src.crackseg.utils.traceability.storage import TraceabilityStorage


class TestTraceabilityQueryInterface:
    """Test suite for TraceabilityQueryInterface."""

    @pytest.fixture
    def temp_storage(self, tmp_path: Path) -> Path:
        """Create temporary storage directory."""
        storage_path = tmp_path / "traceability_storage"
        storage_path.mkdir(parents=True, exist_ok=True)
        return storage_path

    def test_initialization(self, temp_storage: Path) -> None:
        """Test query interface initialization."""
        storage = TraceabilityStorage(storage_path=temp_storage)
        query_interface = TraceabilityQueryInterface(storage=storage)

        assert query_interface.storage is not None
        assert isinstance(query_interface.storage, TraceabilityStorage)
