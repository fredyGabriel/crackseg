"""Test data isolation management for E2E testing.

This module provides isolation mechanisms for test data to prevent conflicts
between parallel test execution and ensure clean test environments.
Integrates with the existing resource management and helper framework.
"""

import logging
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..config.resource_manager import ResourceManager, WorkerIsolation
from ..helpers.setup_teardown import TestEnvironmentManager
from .factories import TestDataFactory

logger = logging.getLogger(__name__)


class TestDataIsolation:
    """Manages test data isolation for parallel test execution."""

    def __init__(
        self,
        resource_manager: ResourceManager | None = None,
        worker_isolation: WorkerIsolation | None = None,
    ) -> None:
        """Initialize test data isolation.

        Args:
            resource_manager: Optional resource manager for coordination
            worker_isolation: Optional worker isolation manager
        """
        self.resource_manager = resource_manager or ResourceManager()
        self.worker_isolation = worker_isolation or WorkerIsolation()
        self.active_isolations: dict[str, TestEnvironmentManager] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(f"{__name__}.TestDataIsolation")

    @contextmanager
    def isolate_test_data(
        self,
        test_id: str,
        memory_limit_mb: int = 256,
        cleanup_on_exit: bool = True,
    ) -> Generator[tuple[TestEnvironmentManager, TestDataFactory], None, None]:
        """Create isolated test data environment.

        Args:
            test_id: Unique identifier for the test
            memory_limit_mb: Memory limit for the isolated environment
            cleanup_on_exit: Whether to cleanup data on context exit

        Yields:
            Tuple of (TestEnvironmentManager, TestDataFactory)
        """
        self.logger.info(
            f"Creating isolated test data environment for {test_id}"
        )

        # Create isolated environment
        with self.worker_isolation.isolate_process(test_id):
            with self.resource_manager.acquire_resources(
                memory_limit_mb=memory_limit_mb,
                cpu_limit=1,
                port_count=1,
                worker_id=test_id,
            ) as _:
                # Create test environment manager
                environment_manager = TestEnvironmentManager(
                    test_name=test_id,
                    artifacts_base_dir=f"test-artifacts/isolated/{test_id}",
                )

                # Setup clean environment
                artifacts_dir = environment_manager.create_clean_environment()

                # Create test data factory
                data_factory = TestDataFactory(environment_manager)

                # Register isolation
                with self._lock:
                    self.active_isolations[test_id] = environment_manager

                try:
                    self.logger.info(
                        f"Isolated environment ready for {test_id}: "
                        f"{artifacts_dir}"
                    )
                    yield environment_manager, data_factory

                finally:
                    # Cleanup if requested
                    if cleanup_on_exit:
                        success = environment_manager.cleanup_environment(
                            preserve_artifacts=False
                        )
                        data_factory.cleanup_all()

                        if success:
                            self.logger.info(
                                f"Cleaned up isolation for {test_id}"
                            )
                        else:
                            self.logger.warning(
                                f"Cleanup issues for {test_id}"
                            )

                    # Remove from active isolations
                    with self._lock:
                        self.active_isolations.pop(test_id, None)

    @contextmanager
    def shared_test_data(
        self, test_group: str, data_types: list[str] | None = None
    ) -> Generator[TestDataFactory, None, None]:
        """Create shared test data environment for a group of tests.

        Args:
            test_group: Name of the test group
            data_types: Types of data to prepare ('config', 'image', 'model')

        Yields:
            TestDataFactory with pre-generated shared data
        """
        self.logger.info(f"Creating shared test data for group {test_group}")

        # Create shared environment
        environment_manager = TestEnvironmentManager(
            test_name=f"shared_{test_group}",
            artifacts_base_dir=f"test-artifacts/shared/{test_group}",
        )

        artifacts_dir = environment_manager.create_clean_environment()
        data_factory = TestDataFactory(environment_manager)

        # Pre-generate common data if specified
        if data_types:
            if "config" in data_types:
                data_factory.generate_config(config_type="basic")
                data_factory.generate_config(config_type="advanced")

            if "image" in data_types:
                data_factory.generate_image(image_type="crack")
                data_factory.generate_image(image_type="no_crack")

            if "model" in data_types:
                data_factory.generate_model(model_type="simple")

        try:
            self.logger.info(
                f"Shared test data ready for {test_group}: {artifacts_dir}"
            )
            yield data_factory

        finally:
            # Cleanup shared data
            success = environment_manager.cleanup_environment(
                preserve_artifacts=False
            )
            data_factory.cleanup_all()

            if success:
                self.logger.info(f"Cleaned up shared data for {test_group}")
            else:
                self.logger.warning(
                    f"Cleanup issues for shared data {test_group}"
                )

    def get_active_isolations(self) -> dict[str, TestEnvironmentManager]:
        """Get currently active test data isolations.

        Returns:
            Dictionary of active test isolations
        """
        with self._lock:
            return self.active_isolations.copy()

    def force_cleanup_isolation(self, test_id: str) -> bool:
        """Force cleanup of a specific test isolation.

        Args:
            test_id: Test ID to cleanup

        Returns:
            True if cleanup was successful
        """
        with self._lock:
            environment_manager = self.active_isolations.get(test_id)

            if environment_manager:
                success = environment_manager.cleanup_environment(
                    preserve_artifacts=False
                )
                self.active_isolations.pop(test_id, None)

                self.logger.info(f"Force cleanup of {test_id}: {success}")
                return success

            return True  # Already cleaned up

    def cleanup_all_isolations(self) -> dict[str, bool]:
        """Force cleanup of all active test isolations.

        Returns:
            Dictionary mapping test_id to cleanup success status
        """
        results = {}

        with self._lock:
            active_tests = list(self.active_isolations.keys())

        for test_id in active_tests:
            results[test_id] = self.force_cleanup_isolation(test_id)

        self.logger.info(f"Cleaned up {len(results)} active isolations")
        return results

    def create_test_workspace(
        self, test_id: str, workspace_type: str = "standard"
    ) -> Path:
        """Create an isolated workspace directory for a test.

        Args:
            test_id: Unique test identifier
            workspace_type: Type of workspace
                ('standard', 'large', 'temporary')

        Returns:
            Path to the created workspace
        """
        base_dir = Path("test-artifacts") / "workspaces" / workspace_type
        workspace_dir = base_dir / test_id

        # Create workspace directory
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # Create standard subdirectories
        (workspace_dir / "configs").mkdir(exist_ok=True)
        (workspace_dir / "images").mkdir(exist_ok=True)
        (workspace_dir / "models").mkdir(exist_ok=True)
        (workspace_dir / "outputs").mkdir(exist_ok=True)
        (workspace_dir / "logs").mkdir(exist_ok=True)

        self.logger.debug(
            f"Created {workspace_type} workspace for {test_id}: "
            f"{workspace_dir}"
        )
        return workspace_dir

    def verify_isolation_integrity(self) -> dict[str, Any]:
        """Verify the integrity of all active test isolations.

        Returns:
            Dictionary with isolation integrity status
        """
        healthy_isolations: list[str] = []
        issue_reports: list[dict[str, str]] = []

        integrity_report: dict[str, Any] = {
            "active_count": 0,
            "healthy": healthy_isolations,
            "issues": issue_reports,
            "resource_usage": {},
        }

        with self._lock:
            active_isolations = self.active_isolations.copy()

        integrity_report["active_count"] = len(active_isolations)

        for test_id, environment_manager in active_isolations.items():
            try:
                # Check if artifacts directory exists and is accessible
                artifacts_dir = environment_manager.state["artifacts_dir"]
                if artifacts_dir.exists() and artifacts_dir.is_dir():
                    healthy_isolations.append(test_id)
                else:
                    issue_reports.append(
                        {
                            "test_id": test_id,
                            "issue": "Missing artifacts directory",
                        }
                    )

            except Exception as e:
                issue_reports.append(
                    {
                        "test_id": test_id,
                        "issue": f"Integrity check failed: {e}",
                    }
                )

        # Get resource usage summary if available
        try:
            integrity_report["resource_usage"] = (
                self.resource_manager.get_resource_usage_summary()
            )
        except Exception as e:
            integrity_report["resource_usage"] = {"error": str(e)}

        return integrity_report
