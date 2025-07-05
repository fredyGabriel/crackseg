"""Main test data factory coordinator.

This module provides the TestDataFactory class that coordinates all
other data factories and manages test suite generation.
"""

import logging
from typing import Any

from ...helpers.setup_teardown import TestEnvironmentManager
from .base import TestData
from .config_factory import ConfigDataFactory
from .image_factory import ImageDataFactory
from .model_factory import ModelDataFactory


class TestDataFactory:
    """Main factory class that coordinates all test data generation."""

    def __init__(
        self, environment_manager: TestEnvironmentManager | None = None
    ) -> None:
        """Initialize the test data factory.

        Args:
            environment_manager: Optional test environment manager
        """
        self.environment_manager = environment_manager
        self.config_factory = ConfigDataFactory(environment_manager)
        self.image_factory = ImageDataFactory(environment_manager)
        self.model_factory = ModelDataFactory(environment_manager)
        self.generated_data: list[TestData] = []
        self.logger = logging.getLogger(f"{__name__}.TestDataFactory")

    def generate_config(self, **kwargs: Any) -> TestData:
        """Generate configuration test data."""
        test_data = self.config_factory.generate(**kwargs)
        self.generated_data.append(test_data)
        return test_data

    def generate_image(self, **kwargs: Any) -> TestData:
        """Generate image test data."""
        test_data = self.image_factory.generate(**kwargs)
        self.generated_data.append(test_data)
        return test_data

    def generate_model(self, **kwargs: Any) -> TestData:
        """Generate model test data."""
        test_data = self.model_factory.generate(**kwargs)
        self.generated_data.append(test_data)
        return test_data

    def generate_test_suite(
        self, suite_type: str = "basic"
    ) -> dict[str, TestData]:
        """Generate a complete test suite with all required data.

        Args:
            suite_type: Type of test suite ('basic', 'comprehensive',
            'error_cases')

        Returns:
            Dictionary of generated test data
        """
        suite_data = {}

        if suite_type == "basic":
            suite_data["config"] = self.generate_config(config_type="basic")
            suite_data["image"] = self.generate_image(image_type="crack")
            suite_data["model"] = self.generate_model(model_type="simple")

        elif suite_type == "comprehensive":
            suite_data["basic_config"] = self.generate_config(
                config_type="basic"
            )
            suite_data["advanced_config"] = self.generate_config(
                config_type="advanced"
            )
            suite_data["crack_image"] = self.generate_image(image_type="crack")
            suite_data["clean_image"] = self.generate_image(
                image_type="no_crack"
            )
            suite_data["edge_image"] = self.generate_image(
                image_type="edge_case"
            )
            suite_data["simple_model"] = self.generate_model(
                model_type="simple"
            )
            suite_data["complex_model"] = self.generate_model(
                model_type="complex"
            )

        elif suite_type == "error_cases":
            suite_data["invalid_config"] = self.generate_config(invalid=True)
            suite_data["corrupted_model"] = self.generate_model(
                corrupt_data=True
            )

        self.logger.info(
            f"Generated {suite_type} test suite with {len(suite_data)} items"
        )
        return suite_data

    def cleanup_single(self, test_data: TestData) -> bool:
        """Clean up a single test data item.

        Args:
            test_data: Test data item to clean up

        Returns:
            True if cleanup was successful
        """
        try:
            if test_data["data_type"] == "config":
                return self.config_factory.cleanup(test_data)
            elif test_data["data_type"] == "image":
                return self.image_factory.cleanup(test_data)
            elif test_data["data_type"] == "model":
                return self.model_factory.cleanup(test_data)
            else:
                self.logger.warning(
                    f"Unknown data type: {test_data['data_type']}"
                )
                return False
        except Exception as e:
            self.logger.error(f"Error cleaning up test data: {e}")
            return False

    def cleanup_all(self) -> bool:
        """Clean up all generated test data.

        Returns:
            True if all cleanup was successful
        """
        success = True

        for test_data in self.generated_data:
            if test_data["data_type"] == "config":
                if not self.config_factory.cleanup(test_data):
                    success = False
            elif test_data["data_type"] == "image":
                if not self.image_factory.cleanup(test_data):
                    success = False
            elif test_data["data_type"] == "model":
                if not self.model_factory.cleanup(test_data):
                    success = False

        self.generated_data.clear()
        self.logger.info(
            f"Cleaned up all generated test data, success: {success}"
        )
        return success
