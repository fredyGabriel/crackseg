"""Resource contention testing for concurrent operations.

Handles resource conflict scenarios and access pattern validation.
Extracted from oversized concurrent_operation_mixin.py for modular organization.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


class ResourceContentionMixin:
    """Mixin providing resource contention testing capabilities."""

    def __init__(self) -> None:
        """Initialize resource contention testing utilities."""
        pass  # Base initialization handled by main mixin

    def test_resource_contention(
        self,
        resource_name: str,
        contention_scenarios: list[dict[str, Any]],
        max_concurrent_access: int = 2,
    ) -> dict[str, Any]:
        """Test resource contention scenarios.

        Args:
            resource_name: Name of resource to test
            contention_scenarios: List of scenarios to test
            max_concurrent_access: Maximum allowed concurrent access

        Returns:
            Resource contention test results
        """
        contention_result: dict[str, Any] = {
            "resource_name": resource_name,
            "max_concurrent_access": max_concurrent_access,
            "scenarios_tested": 0,
            "scenarios_passed": 0,
            "contention_violations": [],
            "access_patterns": [],
        }

        # Create resource lock for the test
        if not hasattr(self, "_resource_locks"):
            self._resource_locks = {}
        if resource_name not in self._resource_locks:
            self._resource_locks[resource_name] = threading.Lock()

        access_counter = threading.Semaphore(max_concurrent_access)

        def resource_access_function(
            scenario_id: int, access_duration: float
        ) -> dict[str, Any]:
            """Function to simulate resource access."""
            access_info = {
                "scenario_id": scenario_id,
                "start_time": time.time(),
                "acquired": False,
                "released": False,
                "duration": access_duration,
            }

            try:
                # Acquire resource with timeout
                if access_counter.acquire(timeout=5.0):
                    access_info["acquired"] = True
                    access_info["actual_start"] = time.time()

                    # Simulate resource usage
                    time.sleep(access_duration)

                    # Release resource
                    access_counter.release()
                    access_info["released"] = True
                    access_info["end_time"] = time.time()
                else:
                    contention_result["contention_violations"].append(
                        f"Scenario {scenario_id}: Resource access timeout"
                    )
            except Exception as e:
                contention_result["contention_violations"].append(
                    f"Scenario {scenario_id}: {str(e)}"
                )

            return access_info

        # Execute contention scenarios
        with ThreadPoolExecutor(
            max_workers=len(contention_scenarios)
        ) as executor:
            futures = []

            for i, scenario in enumerate(contention_scenarios):
                duration = scenario.get("duration", 0.1)
                future = executor.submit(resource_access_function, i, duration)
                futures.append(future)

            contention_result["scenarios_tested"] = len(futures)

            # Collect results
            for future in as_completed(futures):
                try:
                    access_info = future.result()
                    contention_result["access_patterns"].append(access_info)

                    if access_info["acquired"] and access_info["released"]:
                        contention_result["scenarios_passed"] += 1
                except Exception as e:
                    contention_result["contention_violations"].append(
                        f"Future error: {str(e)}"
                    )

        contention_result["success"] = (
            len(contention_result["contention_violations"]) == 0
        )

        return contention_result
