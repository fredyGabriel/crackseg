"""System stability testing under concurrent load.

Handles system stability monitoring and load testing scenarios.
Extracted from oversized concurrent_operation_mixin.py for modular organization.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


class SystemStabilityMixin:
    """Mixin providing system stability testing under load capabilities."""

    def __init__(self) -> None:
        """Initialize system stability testing utilities."""
        pass  # Base initialization handled by main mixin

    def test_system_stability_under_load(
        self,
        load_scenarios: list[dict[str, Any]],
        monitoring_duration: float = 5.0,
    ) -> dict[str, Any]:
        """Test system stability under concurrent load scenarios.

        Args:
            load_scenarios: List of load scenarios to execute
            monitoring_duration: Duration to monitor system metrics

        Returns:
            System stability test results
        """
        stability_result: dict[str, Any] = {
            "monitoring_duration": monitoring_duration,
            "load_scenarios": len(load_scenarios),
            "stability_violations": [],
            "system_metrics": {
                "max_threads": 0,
                "response_times": [],
                "error_count": 0,
                "avg_response_time": 0.0,
                "max_response_time": 0.0,
            },
        }

        # System monitoring setup
        monitor_active = threading.Event()
        monitor_active.set()
        metrics_lock = threading.Lock()

        def system_monitor() -> None:
            """Monitor system metrics during load testing."""
            while monitor_active.is_set():
                # Simulate system metric collection
                current_threads = threading.active_count()

                with metrics_lock:
                    if (
                        current_threads
                        > stability_result["system_metrics"]["max_threads"]
                    ):
                        stability_result["system_metrics"][
                            "max_threads"
                        ] = current_threads

                time.sleep(0.1)  # Monitor every 100ms

        # Start monitoring
        monitor_thread = threading.Thread(target=system_monitor, daemon=True)
        monitor_thread.start()

        # Execute load scenarios concurrently
        with ThreadPoolExecutor(max_workers=len(load_scenarios)) as executor:
            futures = []

            for scenario in load_scenarios:
                scenario_type = scenario.get("type", "cpu_intensive")
                duration = scenario.get("duration", 1.0)

                future = executor.submit(
                    self._execute_load_scenario, scenario_type, duration
                )
                futures.append(future)

            # Collect results and monitor stability
            for future in as_completed(futures):
                start_time = time.time()
                try:
                    scenario_result = future.result()
                    response_time = time.time() - start_time

                    with metrics_lock:
                        stability_result["system_metrics"][
                            "response_times"
                        ].append(response_time)

                    if not scenario_result.get("success", False):
                        stability_result["stability_violations"].append(
                            f"Load scenario failed: {scenario_result.get('error', 'Unknown')}"
                        )
                except Exception as e:
                    with metrics_lock:
                        stability_result["system_metrics"]["error_count"] += 1
                        stability_result["stability_violations"].append(str(e))

        # Stop monitoring
        monitor_active.clear()
        monitor_thread.join(timeout=1.0)

        # Analyze stability
        response_times = stability_result["system_metrics"]["response_times"]
        if response_times:
            avg_response = sum(response_times) / len(response_times)
            max_response = max(response_times)

            # Check for stability violations
            if max_response > 1.0:  # 1 second threshold
                stability_result["stability_violations"].append(
                    f"High response time: {max_response:.3f}s"
                )

            stability_result["system_metrics"][
                "avg_response_time"
            ] = avg_response
            stability_result["system_metrics"][
                "max_response_time"
            ] = max_response

        stability_result["success"] = (
            len(stability_result["stability_violations"]) == 0
            and stability_result["system_metrics"]["error_count"] == 0
        )

        return stability_result

    def _execute_load_scenario(
        self, scenario_type: str, duration: float
    ) -> dict[str, Any]:
        """Execute a specific load scenario."""
        scenario_result = {
            "type": scenario_type,
            "duration": duration,
            "success": False,
        }

        start_time = time.time()
        try:
            if scenario_type == "cpu_intensive":
                # Simulate CPU intensive work
                end_time = start_time + duration
                while time.time() < end_time:
                    sum(range(1000))  # Simple CPU work
            elif scenario_type == "memory_intensive":
                # Simulate memory intensive work
                data = [list(range(1000)) for _ in range(100)]
                time.sleep(duration)
                del data
            elif scenario_type == "io_intensive":
                # Simulate I/O intensive work
                for _ in range(int(duration * 10)):
                    time.sleep(0.1)

            scenario_result["success"] = True
        except Exception as e:
            scenario_result["error"] = str(e)

        scenario_result["actual_duration"] = time.time() - start_time
        return scenario_result
