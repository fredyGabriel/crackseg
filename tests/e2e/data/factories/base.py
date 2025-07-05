"""Base classes and types for test data factories.

This module provides the foundational types, protocols, and abstract classes
for the test data factory system.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, TypedDict

from ...helpers.setup_teardown import TestEnvironmentManager

logger = logging.getLogger(__name__)


class TestData(TypedDict):
    """Type definition for test data artifacts."""

    data_type: str
    file_path: Path
    metadata: dict[str, Any]
    cleanup_required: bool


class DataFactory(Protocol):
    """Protocol for test data factories."""

    def generate(self, **kwargs: Any) -> TestData:
        """Generate test data."""
        ...

    def cleanup(self, test_data: TestData) -> bool:
        """Clean up generated test data."""
        ...


class BaseDataFactory(ABC):
    """Abstract base class for all test data factories."""

    def __init__(
        self, environment_manager: TestEnvironmentManager | None = None
    ) -> None:
        """Initialize the data factory.

        Args:
            environment_manager: Optional test environment manager for cleanup
        """
        self.environment_manager = environment_manager
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

    @abstractmethod
    def generate(self, **kwargs: Any) -> TestData:
        """Generate test data based on provided parameters.

        Args:
            **kwargs: Factory-specific parameters

        Returns:
            TestData containing generated data information
        """
        pass

    def cleanup(self, test_data: TestData) -> bool:
        """Clean up generated test data.

        Args:
            test_data: Test data to clean up

        Returns:
            True if cleanup was successful
        """
        try:
            if (
                test_data["cleanup_required"]
                and test_data["file_path"].exists()
            ):
                if test_data["file_path"].is_file():
                    test_data["file_path"].unlink()
                elif test_data["file_path"].is_dir():
                    import shutil

                    shutil.rmtree(test_data["file_path"])

                self.logger.debug(
                    f"Cleaned up test data: {test_data['file_path']}"
                )
                return True
            return True
        except Exception as e:
            self.logger.error(f"Failed to clean up test data: {e}")
            return False
