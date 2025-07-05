"""Test coordination helpers for managing complex E2E test scenarios.

This module provides utilities for coordinating parallel tests, managing test
dependencies, synchronizing test states, and orchestrating complex test
sequences. These helpers are essential for advanced test scenarios and
parallel execution frameworks.
"""

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from selenium.webdriver.remote.webdriver import WebDriver

logger = logging.getLogger(__name__)


class TestResult(TypedDict):
    """Type definition for test execution results."""

    test_name: str
    success: bool
    duration: float
    error: str | None
    artifacts: list[Path]


class TestStatus(TypedDict):
    """Type definition for test status tracking."""

    test_name: str
    status: str  # pending, running, completed, failed, skipped
    start_time: float | None
    end_time: float | None
    dependencies_met: bool


@dataclass
class TestTask:
    """
    Data class representing a test task with dependencies and configuration.
    """

    name: str
    test_function: Callable[..., Any]
    dependencies: list[str] = field(default_factory=list)
    priority: int = 1
    timeout: float = 300.0  # 5 minutes default
    retry_count: int = 0
    parallel_safe: bool = True
    artifacts_dir: Path | None = None


class TestCoordinator:
    """Comprehensive test coordinator for complex E2E test orchestration."""

    def __init__(self, max_workers: int = 4) -> None:
        """Initialize test coordinator.

        Args:
            max_workers: Maximum number of parallel workers
        """
        self.max_workers = max_workers
        self.tasks: dict[str, TestTask] = {}
        self.results: dict[str, TestResult] = {}
        self.status: dict[str, TestStatus] = {}
        self.locks: dict[str, threading.Lock] = {}
        self._execution_lock = threading.Lock()
        self.logger = logging.getLogger(f"{__name__}.TestCoordinator")

    def register_test(
        self,
        name: str,
        test_function: Callable[..., Any],
        dependencies: list[str] | None = None,
        priority: int = 1,
        timeout: float = 300.0,
        parallel_safe: bool = True,
    ) -> None:
        """Register a test task with the coordinator.

        Args:
            name: Unique test name
            test_function: Test function to execute
            dependencies: List of test names this test depends on
            priority: Test priority (higher numbers = higher priority)
            timeout: Maximum execution time
            parallel_safe: Whether test can run in parallel with others
        """
        task = TestTask(
            name=name,
            test_function=test_function,
            dependencies=dependencies or [],
            priority=priority,
            timeout=timeout,
            parallel_safe=parallel_safe,
        )

        self.tasks[name] = task
        self.status[name] = {
            "test_name": name,
            "status": "pending",
            "start_time": None,
            "end_time": None,
            "dependencies_met": len(task.dependencies) == 0,
        }
        self.locks[name] = threading.Lock()

        self.logger.info(
            f"Registered test '{name}' with "
            f"{len(task.dependencies)} dependencies"
        )

    def execute_parallel_tests(
        self, test_names: list[str] | None = None
    ) -> dict[str, TestResult]:
        """Execute tests in parallel while respecting dependencies.

        Args:
            test_names: Specific tests to execute (None = all tests)

        Returns:
            Dictionary of test results
        """
        if test_names is None:
            test_names = list(self.tasks.keys())

        self.logger.info(
            f"Starting parallel execution of {len(test_names)} tests"
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tests that can start immediately
            futures = {}

            for test_name in test_names:
                if self._can_start_test(test_name):
                    future = executor.submit(
                        self._execute_single_test, test_name
                    )
                    futures[future] = test_name
                    self._update_test_status(test_name, "running")

            # Process completed tests and submit new ones
            while futures:
                try:
                    completed = as_completed(futures, timeout=1.0)

                    for future in completed:
                        test_name = futures.pop(future)

                        try:
                            result = future.result()
                            self.results[test_name] = result
                            self._update_test_status(
                                test_name,
                                "completed" if result["success"] else "failed",
                            )

                        except Exception as e:
                            self.logger.error(
                                f"Test {test_name} failed with exception: {e}"
                            )
                            self.results[test_name] = {
                                "test_name": test_name,
                                "success": False,
                                "duration": 0.0,
                                "error": str(e),
                                "artifacts": [],
                            }
                            self._update_test_status(test_name, "failed")
                except TimeoutError:
                    # Check for new tests to start
                    pass

        self.logger.info(
            f"Parallel execution completed. {len(self.results)} tests executed"
        )
        return self.results

    def _can_start_test(self, test_name: str) -> bool:
        """Check if a test can start based on dependencies and status."""
        if test_name not in self.tasks:
            return False

        status = self.status[test_name]
        if status["status"] != "pending":
            return False

        # Check if all dependencies are completed successfully
        task = self.tasks[test_name]
        for dep_name in task.dependencies:
            dep_status = self.status.get(dep_name)
            if not dep_status or dep_status["status"] != "completed":
                return False

            # Check if dependency was successful
            dep_result = self.results.get(dep_name)
            if not dep_result or not dep_result["success"]:
                return False

        return True

    def _execute_single_test(self, test_name: str) -> TestResult:
        """Execute a single test with timing and error handling."""
        task = self.tasks[test_name]
        start_time = time.time()

        self.logger.info(f"Executing test '{test_name}'")

        try:
            # Execute the test function
            task.test_function()

            duration = time.time() - start_time

            return {
                "test_name": test_name,
                "success": True,
                "duration": duration,
                "error": None,
                "artifacts": [],
            }

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test '{test_name}' failed: {e}")

            return {
                "test_name": test_name,
                "success": False,
                "duration": duration,
                "error": str(e),
                "artifacts": [],
            }

    def _update_test_status(self, test_name: str, status: str) -> None:
        """Update the status of a test."""
        with self.locks[test_name]:
            self.status[test_name]["status"] = status

            if (
                status == "running"
                and self.status[test_name]["start_time"] is None
            ):
                self.status[test_name]["start_time"] = time.time()
            elif (
                status in ["completed", "failed"]
                and self.status[test_name]["end_time"] is None
            ):
                self.status[test_name]["end_time"] = time.time()


def coordinate_parallel_tests(
    test_functions: dict[str, Callable[..., Any]], max_workers: int = 4
) -> dict[str, TestResult]:
    """Coordinate parallel execution of multiple test functions."""
    coordinator = TestCoordinator(max_workers)

    # Register all tests
    for test_name, test_func in test_functions.items():
        coordinator.register_test(test_name, test_func)

    # Execute all tests
    return coordinator.execute_parallel_tests()


def manage_test_dependencies(
    test_sequence: list[tuple[str, Callable[..., Any], list[str]]],
) -> TestCoordinator:
    """Set up a test coordinator with dependency management."""
    coordinator = TestCoordinator()

    for test_name, test_func, dependencies in test_sequence:
        coordinator.register_test(
            name=test_name, test_function=test_func, dependencies=dependencies
        )

    return coordinator


def synchronize_test_states(
    webdrivers: list[WebDriver], sync_point: str = "app_loaded"
) -> bool:
    """Synchronize multiple WebDriver instances to same app state."""
    try:
        from ..utils.streamlit import wait_for_streamlit_ready

        success_count = 0

        for i, driver in enumerate(webdrivers):
            try:
                # Navigate all drivers to the same starting point
                driver.refresh()
                wait_for_streamlit_ready(driver, timeout=30)
                success_count += 1
                logger.debug(f"Driver {i} synchronized to {sync_point}")

            except Exception as e:
                logger.error(f"Failed to synchronize driver {i}: {e}")

        success = success_count == len(webdrivers)
        logger.info(
            f"Test state synchronization: {success_count}/"
            f"{len(webdrivers)} drivers synchronized"
        )

        return success

    except Exception as e:
        logger.error(f"Test synchronization failed: {e}")
        return False


def orchestrate_test_sequence(
    test_phases: list[dict[str, Any]], cleanup_between_phases: bool = True
) -> dict[str, Any]:
    """Orchestrate a complex multi-phase test sequence."""
    orchestration_results: dict[str, Any] = {
        "phases_completed": 0,
        "total_phases": len(test_phases),
        "phase_results": [],
        "overall_success": True,
        "total_duration": 0.0,
    }

    start_time = time.time()

    try:
        for phase_idx, phase_config in enumerate(test_phases):
            phase_name = phase_config.get("name", f"Phase_{phase_idx + 1}")
            phase_tests = phase_config.get("tests", {})
            phase_timeout = phase_config.get("timeout", 600.0)

            logger.info(f"Starting orchestration phase: {phase_name}")

            # Execute phase tests
            coordinator = TestCoordinator()
            for test_name, test_func in phase_tests.items():
                coordinator.register_test(
                    test_name, test_func, timeout=phase_timeout
                )

            phase_results = coordinator.execute_parallel_tests()

            # Check phase success
            phase_success = all(
                result["success"] for result in phase_results.values()
            )

            orchestration_results["phase_results"].append(
                {
                    "phase_name": phase_name,
                    "success": phase_success,
                    "test_results": phase_results,
                }
            )

            if phase_success:
                orchestration_results["phases_completed"] += 1
                logger.info(f"Phase {phase_name} completed successfully")
            else:
                logger.error(f"Phase {phase_name} failed")
                orchestration_results["overall_success"] = False
                break

            # Cleanup between phases if requested
            if cleanup_between_phases and phase_idx < len(test_phases) - 1:
                logger.info(f"Performing cleanup after phase {phase_name}")
                time.sleep(2)  # Allow for cleanup

    except Exception as e:
        logger.error(f"Test orchestration failed: {e}")
        orchestration_results["overall_success"] = False

    orchestration_results["total_duration"] = time.time() - start_time

    logger.info(
        f"Test orchestration completed: "
        f"{orchestration_results['phases_completed']}/"
        f"{orchestration_results['total_phases']} phases successful"
    )

    return orchestration_results
