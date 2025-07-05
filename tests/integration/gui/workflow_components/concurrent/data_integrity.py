"""Data integrity testing under concurrent operations.

Handles data consistency and integrity validation during concurrent access.
Extracted from oversized concurrent_operation_mixin.py for modular
organization.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


class DataIntegrityMixin:
    """Mixin providing data integrity testing under concurrent operations."""

    def __init__(self) -> None:
        """Initialize data integrity testing utilities."""
        pass  # Base initialization handled by main mixin

    def test_data_integrity_under_concurrency(
        self,
        data_operations: list[dict[str, Any]],
        shared_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Test data integrity under concurrent operations.

        Args:
            data_operations: List of data operations to execute concurrently
            shared_data: Shared data structure for concurrent access

        Returns:
            Data integrity test results
        """
        if shared_data is None:
            shared_data = {"counter": 0, "data_store": {}}

        integrity_result: dict[str, Any] = {
            "operations_count": len(data_operations),
            "operations_completed": 0,
            "operations_failed": 0,
            "integrity_violations": [],
            "data_consistency_checks": [],
            "final_data_state": {},
        }

        # Data access lock
        data_lock = threading.Lock()
        results_lock = threading.Lock()

        def execute_data_operation(
            operation: dict[str, Any],
        ) -> dict[str, Any]:
            """Execute a single data operation with integrity checks."""
            op_type = operation.get("type", "read")
            op_id = operation.get("id", "unknown")

            operation_result = {
                "operation_type": op_type,
                "operation_id": op_id,
                "success": False,
                "start_time": time.time(),
            }

            try:
                if op_type == "read":
                    # Read operation
                    with data_lock:
                        value = shared_data.get("counter", 0)
                        operation_result["read_value"] = value

                elif op_type == "write":
                    # Write operation
                    with data_lock:
                        shared_data["counter"] = (
                            shared_data.get("counter", 0) + 1
                        )
                        shared_data["data_store"][op_id] = f"data_{op_id}"
                        operation_result["written_value"] = shared_data[
                            "counter"
                        ]

                elif op_type == "modify":
                    # Modify existing data
                    with data_lock:
                        current_value = shared_data.get("counter", 0)
                        shared_data["counter"] = current_value * 2
                        operation_result["modified_value"] = shared_data[
                            "counter"
                        ]

                elif op_type == "delete":
                    # Delete operation
                    with data_lock:
                        key_to_delete = f"data_{op_id}"
                        if key_to_delete in shared_data["data_store"]:
                            del shared_data["data_store"][key_to_delete]
                            operation_result["deleted_key"] = key_to_delete
                        else:
                            operation_result["delete_failed"] = "key_not_found"

                operation_result["success"] = True

                with results_lock:
                    integrity_result["operations_completed"] += 1

            except Exception as e:
                operation_result["error"] = str(e)
                with results_lock:
                    integrity_result["operations_failed"] += 1
                    integrity_result["integrity_violations"].append(
                        f"Operation {op_id}: {str(e)}"
                    )

            operation_result["end_time"] = time.time()
            return operation_result

        # Execute all operations concurrently
        with ThreadPoolExecutor(max_workers=len(data_operations)) as executor:
            futures = []

            for operation in data_operations:
                future = executor.submit(execute_data_operation, operation)
                futures.append(future)

            # Collect results
            operation_results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    operation_results.append(result)
                except Exception as e:
                    with results_lock:
                        integrity_result["integrity_violations"].append(
                            f"Future error: {str(e)}"
                        )

        # Perform data consistency checks
        with data_lock:
            final_counter = shared_data.get("counter", 0)
            final_data_store = shared_data.get("data_store", {}).copy()

        # Validate data consistency
        write_operations = [
            op for op in data_operations if op.get("type") == "write"
        ]
        expected_writes = len(write_operations)

        if expected_writes > 0:
            # Check if counter reflects write operations
            integrity_result["data_consistency_checks"].append(
                {
                    "check_type": "write_counter_consistency",
                    "expected_writes": expected_writes,
                    "actual_counter": final_counter,
                    "consistent": final_counter >= expected_writes,
                }
            )

        # Store final state
        integrity_result["final_data_state"] = {
            "counter": final_counter,
            "data_store_keys": list(final_data_store.keys()),
            "data_store_count": len(final_data_store),
        }

        # Overall integrity assessment
        integrity_result["success"] = (
            len(integrity_result["integrity_violations"]) == 0
            and integrity_result["operations_failed"] == 0
            and all(
                check["consistent"]
                for check in integrity_result["data_consistency_checks"]
            )
        )

        return integrity_result
