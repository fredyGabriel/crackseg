"""Core test data provisioning functionality.

This module provides the main TestDataProvisioner class with
basic provisioning operations and data management.
"""

import logging
from pathlib import Path
from typing import Any, TypedDict

from ...helpers.setup_teardown import TestEnvironmentManager
from ..factories import TestData, TestDataFactory

logger = logging.getLogger(__name__)


class ProvisioningResult(TypedDict):
    """Result of data provisioning operation."""

    success: bool
    duration: float
    provisioned_items: list[str]
    errors: list[str]
    metadata: dict[str, Any]


class TestDataProvisioner:
    """Handles test data provisioning and seeding operations."""

    def __init__(
        self,
        environment_manager: TestEnvironmentManager | None = None,
        data_factory: TestDataFactory | None = None,
    ) -> None:
        """Initialize test data provisioner.

        Args:
            environment_manager: Optional test environment manager
            data_factory: Optional test data factory
        """
        self.environment_manager = environment_manager
        self.data_factory = data_factory or TestDataFactory(
            environment_manager
        )
        self.provisioned_data: dict[str, TestData] = {}
        self.seeding_history: list[dict[str, Any]] = []
        self.logger = logging.getLogger(f"{__name__}.TestDataProvisioner")

    def provision_basic_suite(self) -> ProvisioningResult:
        """Provision a basic test data suite."""
        # Implementation will be bound from suites.py
        ...

    def provision_comprehensive_suite(self) -> ProvisioningResult:
        """Provision a comprehensive test data suite."""
        # Implementation will be bound from suites.py
        ...

    def provision_error_test_data(self) -> ProvisioningResult:
        """Provision test data for error condition testing."""
        # Implementation will be bound from suites.py
        ...

    def seed_test_database(self, database_path: Path | str) -> bool:
        """Seed a test database with provisioned data."""
        # Implementation will be bound from database.py
        ...

    def get_provisioning_summary(self) -> dict[str, Any]:
        """Get summary of current provisioning state."""
        # Implementation will be bound from database.py
        ...

    def get_provisioned_data(
        self, name: str | None = None
    ) -> TestData | dict[str, TestData] | None:
        """Get provisioned test data.

        Args:
            name: Optional specific data name to retrieve

        Returns:
            Single TestData if name specified, otherwise all provisioned data
        """
        if name:
            return self.provisioned_data.get(name)
        return self.provisioned_data.copy()

    def cleanup_provisioned_data(self) -> dict[str, bool]:
        """Clean up all provisioned test data.

        Returns:
            Dictionary mapping data names to cleanup success status
        """
        cleanup_results = {}

        for name, test_data in self.provisioned_data.items():
            try:
                success = self.data_factory.cleanup_single(test_data)
                cleanup_results[name] = success
                if not success:
                    self.logger.warning(
                        f"Failed to cleanup provisioned data: {name}"
                    )
            except Exception as e:
                self.logger.error(
                    f"Error cleaning up provisioned data {name}: {e}"
                )
                cleanup_results[name] = False

        # Clear the provisioned data registry
        self.provisioned_data.clear()

        success_count = sum(cleanup_results.values())
        total_count = len(cleanup_results)

        self.logger.info(
            f"Cleaned up {success_count}/{total_count} provisioned data items"
        )
        return cleanup_results

    def provision_custom_data(
        self, data_specs: list[dict[str, Any]]
    ) -> ProvisioningResult:
        """Provision custom test data based on specifications.

        Args:
            data_specs: List of data specifications with type and parameters

        Returns:
            ProvisioningResult with operation details
        """
        import time

        start_time = time.time()
        provisioned_items = []
        errors = []

        try:
            for spec in data_specs:
                data_type = spec.get("type")
                name = spec.get(
                    "name", f"{data_type}_{len(provisioned_items)}"
                )
                params = spec.get("params", {})

                if data_type == "config":
                    test_data = self.data_factory.generate_config(**params)
                elif data_type == "image":
                    test_data = self.data_factory.generate_image(**params)
                elif data_type == "model":
                    test_data = self.data_factory.generate_model(**params)
                else:
                    error_msg = f"Unknown data type: {data_type}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
                    continue

                self.provisioned_data[name] = test_data
                provisioned_items.append(name)

            duration = time.time() - start_time

            self.logger.info(
                f"Custom data provisioned in {duration:.2f}s: "
                f"{len(provisioned_items)} items"
            )

            return {
                "success": len(errors) == 0,
                "duration": duration,
                "provisioned_items": provisioned_items,
                "errors": errors,
                "metadata": {
                    "suite_type": "custom",
                    "item_count": len(provisioned_items),
                    "spec_count": len(data_specs),
                },
            }

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Custom data provisioning failed: {e}"
            self.logger.error(error_msg)
            errors.append(error_msg)

            return {
                "success": False,
                "duration": duration,
                "provisioned_items": provisioned_items,
                "errors": errors,
                "metadata": {"suite_type": "custom", "failure_reason": str(e)},
            }
