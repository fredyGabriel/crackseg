"""
Multi-user workflow simulation for concurrent operation testing.
Handles concurrent user simulation and workflow execution patterns.
Extracted from oversized concurrent_operation_mixin.py for modular
organization.
"""

import threading
import time
from collections.abc import Callable
from typing import Any


class MultiUserMixin:
    """Mixin providing multi-user workflow simulation capabilities."""

    def __init__(self) -> None:
        """Initialize multi-user testing utilities."""
        pass  # Base initialization handled by main mixin

    def simulate_multi_user_workflow(
        self,
        user_count: int = 3,
        workflow_function: Callable[[int], Any] | None = None,
        shared_resources: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Simulate multiple users executing workflows simultaneously.

        Args:
            user_count: Number of concurrent users to simulate
            workflow_function: Function to execute for each user
            shared_resources: Resources shared between users

        Returns:
            Multi-user workflow test results
        """
        if workflow_function is None:
            workflow_function = self._default_user_workflow

        if shared_resources is None:
            shared_resources = {}

        multi_user_result: dict[str, Any] = {
            "user_count": user_count,
            "workflows_started": 0,
            "workflows_completed": 0,
            "workflows_failed": 0,
            "execution_times": [],
            "resource_conflicts": [],
            "user_results": {},
            "concurrent_issues": [],
        }

        # Track active users
        user_threads: list[threading.Thread] = []
        results_lock = threading.Lock()

        def user_workflow_wrapper(user_id: int) -> None:
            """Wrapper for individual user workflow execution."""
            start_time = time.time()
            try:
                with results_lock:
                    multi_user_result["workflows_started"] += 1

                result = workflow_function(user_id)
                execution_time = time.time() - start_time

                with results_lock:
                    multi_user_result["workflows_completed"] += 1
                    multi_user_result["execution_times"].append(execution_time)
                    multi_user_result["user_results"][
                        f"user_{user_id}"
                    ] = result

            except Exception as e:
                with results_lock:
                    multi_user_result["workflows_failed"] += 1
                    multi_user_result["concurrent_issues"].append(
                        f"User {user_id}: {str(e)}"
                    )

        # Start all user workflows concurrently
        for user_id in range(user_count):
            thread = threading.Thread(
                target=user_workflow_wrapper,
                args=(user_id,),
                name=f"User-{user_id}",
            )
            user_threads.append(thread)
            thread.start()

        # Wait for all workflows to complete
        for thread in user_threads:
            thread.join(timeout=30.0)  # 30 second timeout per user

        # Calculate statistics
        if multi_user_result["execution_times"]:
            times = multi_user_result["execution_times"]
            multi_user_result["avg_execution_time"] = sum(times) / len(times)
            multi_user_result["max_execution_time"] = max(times)
            multi_user_result["min_execution_time"] = min(times)

        multi_user_result["success"] = (
            multi_user_result["workflows_failed"] == 0
        )

        return multi_user_result

    def _default_user_workflow(self, user_id: int) -> dict[str, Any]:
        """Default user workflow for multi-user testing."""
        return {
            "user_id": user_id,
            "workflow_steps": ["config_load", "training_setup", "execution"],
            "success": True,
            "execution_time": 0.1,
        }
