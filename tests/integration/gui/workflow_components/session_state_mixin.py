"""Session state mixin for modular integration testing.

This module provides reusable session state testing patterns that can be mixed
into workflow components to add comprehensive session state verification
capabilities. Builds on the workflow and error testing foundation from 9.1/9.2.
"""

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol


class SessionStateTestUtilities(Protocol):
    """Protocol for test utilities needed by session state components."""

    temp_path: Path


class SessionStateMixin:
    """Mixin providing reusable session state testing patterns for workflow
    components."""

    def create_session_state_proxy(
        self, initial_state: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create a proxy for session state testing.

        Args:
            initial_state: Initial state values

        Returns:
            Session state proxy dictionary
        """
        proxy = initial_state.copy() if initial_state else {}
        default_keys = {
            "_crackseg_state": None,
            "app_state": None,
            "config_path": None,
            "run_directory": None,
            "current_page": "Config",
            "theme": "dark",
            "training_active": False,
        }
        for key, value in default_keys.items():
            if key not in proxy:
                proxy[key] = value
        return proxy

    def simulate_session_state_persistence(
        self,
        session_proxy: dict[str, Any],
        state_transitions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Simulate session state persistence across multiple operations.

        Args:
            session_proxy: Session state proxy
            state_transitions: List of state changes to apply

        Returns:
            Persistence test results
        """
        persistence_result: dict[str, Any] = {
            "initial_state": session_proxy.copy(),
            "transitions_applied": 0,
            "transitions_successful": 0,
            "final_state": {},
            "state_history": [],
            "persistence_violations": [],
        }

        try:
            for i, transition in enumerate(state_transitions):
                pre_state = session_proxy.copy()
                persistence_result["state_history"].append(
                    {"step": i, "pre_state": pre_state.copy()}
                )

                for key, value in transition.items():
                    if key in session_proxy:
                        old_value = session_proxy[key]
                        session_proxy[key] = value

                        if value is None and old_value is not None:
                            persistence_result[
                                "persistence_violations"
                            ].append(f"Step {i}: Lost state for key '{key}'")

                persistence_result["transitions_applied"] += 1

                if self._validate_session_state_consistency(session_proxy):
                    persistence_result["transitions_successful"] += 1
                else:
                    persistence_result["persistence_violations"].append(
                        f"Step {i}: State consistency violation"
                    )

            persistence_result["final_state"] = session_proxy.copy()

        except Exception as e:
            persistence_result["error"] = str(e)

        return persistence_result

    def test_session_state_transitions(
        self,
        session_proxy: dict[str, Any],
        transition_scenario: str,
    ) -> dict[str, Any]:
        """Test session state transitions for specific scenarios.

        Args:
            session_proxy: Session state proxy
            transition_scenario: Type of transition to test

        Returns:
            Transition test results
        """
        # Initialize with explicit list types
        expected_transitions: list[str] = []
        actual_transitions: list[str] = []
        transition_errors: list[str] = []

        transition_result: dict[str, Any] = {
            "scenario": transition_scenario,
            "success": False,
            "initial_state": session_proxy.copy(),
            "expected_transitions": expected_transitions,
            "actual_transitions": actual_transitions,
            "transition_errors": transition_errors,
        }

        scenarios = {
            "config_to_training": self._simulate_config_to_training_transition,
            "training_to_results": (
                self._simulate_training_to_results_transition
            ),
            "page_navigation": self._simulate_page_navigation_transition,
            "error_recovery": self._simulate_error_recovery_transition,
            "session_timeout": self._simulate_session_timeout_transition,
        }

        if transition_scenario in scenarios:
            try:
                transition_func = scenarios[transition_scenario]
                result = transition_func(session_proxy)
                # Update only the relevant fields, preserve our lists
                if "expected_transitions" in result:
                    transition_result["expected_transitions"] = result[
                        "expected_transitions"
                    ]
                if "actual_transitions" in result:
                    transition_result["actual_transitions"] = result[
                        "actual_transitions"
                    ]
                transition_result["success"] = len(transition_errors) == 0
            except Exception as e:
                transition_errors.append(str(e))

        return transition_result

    def validate_session_state_lifecycle(
        self,
        create_function: Callable[[dict[str, Any]], Any],
        use_function: Callable[[dict[str, Any]], Any],
        cleanup_function: Callable[[dict[str, Any]], Any],
    ) -> dict[str, Any]:
        """Test complete session state lifecycle.

        Args:
            create_function: Function to create session state
            use_function: Function to use session state
            cleanup_function: Function to cleanup session state

        Returns:
            Lifecycle test results
        """
        lifecycle_result: dict[str, Any] = {
            "creation_successful": False,
            "usage_successful": False,
            "cleanup_successful": False,
            "lifecycle_errors": [],
            "state_snapshots": {},
        }

        session_proxy = self.create_session_state_proxy()

        try:
            create_function(session_proxy)
            lifecycle_result["creation_successful"] = True
            lifecycle_result["state_snapshots"][
                "after_creation"
            ] = session_proxy.copy()

            use_function(session_proxy)
            lifecycle_result["usage_successful"] = True
            lifecycle_result["state_snapshots"][
                "after_usage"
            ] = session_proxy.copy()

            cleanup_function(session_proxy)
            lifecycle_result["cleanup_successful"] = True
            lifecycle_result["state_snapshots"][
                "after_cleanup"
            ] = session_proxy.copy()

        except Exception as e:
            lifecycle_result["lifecycle_errors"].append(str(e))

        return lifecycle_result

    def create_session_state_snapshot(
        self, session_proxy: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a deep snapshot of session state for comparison.

        Args:
            session_proxy: Session state proxy

        Returns:
            Deep copy snapshot with metadata
        """
        import copy

        snapshot = {
            "timestamp": time.time(),
            "state_data": copy.deepcopy(session_proxy),
            "state_keys": list(session_proxy.keys()),
            "state_size": len(json.dumps(session_proxy, default=str)),
        }
        return snapshot

    def compare_session_state_snapshots(
        self, snapshot1: dict[str, Any], snapshot2: dict[str, Any]
    ) -> dict[str, Any]:
        """Compare two session state snapshots.

        Args:
            snapshot1: First snapshot
            snapshot2: Second snapshot

        Returns:
            Comparison results
        """
        # Create explicit typed lists to prevent type inference issues
        keys_added: list[str] = []
        keys_removed: list[str] = []
        keys_modified: list[str] = []

        comparison: dict[str, Any] = {
            "snapshots_equal": False,
            "keys_added": keys_added,
            "keys_removed": keys_removed,
            "keys_modified": keys_modified,
            "size_difference": 0,
            "time_difference": 0,
        }

        state1 = snapshot1["state_data"]
        state2 = snapshot2["state_data"]

        keys1 = set(state1.keys())
        keys2 = set(state2.keys())

        keys_added.extend(keys2 - keys1)
        keys_removed.extend(keys1 - keys2)

        common_keys = keys1 & keys2
        for key in common_keys:
            if state1[key] != state2[key]:
                keys_modified.append(key)

        comparison["snapshots_equal"] = (
            len(keys_added) == 0
            and len(keys_removed) == 0
            and len(keys_modified) == 0
        )

        comparison["size_difference"] = (
            snapshot2["state_size"] - snapshot1["state_size"]
        )
        comparison["time_difference"] = (
            snapshot2["timestamp"] - snapshot1["timestamp"]
        )

        return comparison

    def _validate_session_state_consistency(
        self, session_proxy: dict[str, Any]
    ) -> bool:
        """Validate session state consistency.

        Args:
            session_proxy: Session state to validate

        Returns:
            True if state is consistent
        """
        try:
            if "_crackseg_state" in session_proxy:
                crackseg_state = session_proxy["_crackseg_state"]
                if crackseg_state is not None:
                    if hasattr(crackseg_state, "training_active"):
                        training_active = crackseg_state.training_active
                        process_state = getattr(
                            crackseg_state, "process_state", "idle"
                        )

                        if training_active and process_state not in [
                            "running",
                            "starting",
                        ]:
                            return False

            return True

        except Exception:
            return False

    def _simulate_config_to_training_transition(
        self, session_proxy: dict[str, Any]
    ) -> dict[str, Any]:
        """Simulate transition from config to training page."""
        result = {
            "expected_transitions": [
                "config_loaded",
                "page_change",
                "training_ready",
            ],
            "actual_transitions": [],
        }

        session_proxy["config_path"] = str(Path("test_config.yaml"))
        session_proxy["config_loaded"] = True
        result["actual_transitions"].append("config_loaded")

        session_proxy["current_page"] = "Training"
        result["actual_transitions"].append("page_change")

        session_proxy["training_ready"] = True
        result["actual_transitions"].append("training_ready")

        return result

    def _simulate_training_to_results_transition(
        self, session_proxy: dict[str, Any]
    ) -> dict[str, Any]:
        """Simulate transition from crackseg.training to results page."""
        result = {
            "expected_transitions": [
                "training_complete",
                "results_available",
                "page_change",
            ],
            "actual_transitions": [],
        }

        session_proxy["training_active"] = False
        session_proxy["training_progress"] = 1.0
        result["actual_transitions"].append("training_complete")

        session_proxy["results_available"] = True
        result["actual_transitions"].append("results_available")

        session_proxy["current_page"] = "Results"
        result["actual_transitions"].append("page_change")

        return result

    def _simulate_page_navigation_transition(
        self, session_proxy: dict[str, Any]
    ) -> dict[str, Any]:
        """Simulate page navigation transitions."""
        result = {
            "expected_transitions": ["state_preserved", "page_changed"],
            "actual_transitions": [],
        }

        original_config = session_proxy.get("config_path")
        original_theme = session_proxy.get("theme", "dark")

        session_proxy["current_page"] = "Architecture"
        result["actual_transitions"].append("page_changed")

        if (
            session_proxy.get("config_path") == original_config
            and session_proxy.get("theme") == original_theme
        ):
            result["actual_transitions"].append("state_preserved")

        return result

    def _simulate_error_recovery_transition(
        self, session_proxy: dict[str, Any]
    ) -> dict[str, Any]:
        """Simulate error recovery state transitions."""
        result = {
            "expected_transitions": [
                "error_occurred",
                "state_preserved",
                "recovery_initiated",
            ],
            "actual_transitions": [],
        }

        pre_error_state = session_proxy.copy()

        session_proxy["error_state"] = True
        session_proxy["error_message"] = "Simulated error"
        result["actual_transitions"].append("error_occurred")

        if session_proxy.get("config_path") == pre_error_state.get(
            "config_path"
        ):
            result["actual_transitions"].append("state_preserved")

        session_proxy["error_state"] = False
        session_proxy["error_message"] = None
        result["actual_transitions"].append("recovery_initiated")

        return result

    def _simulate_session_timeout_transition(
        self, session_proxy: dict[str, Any]
    ) -> dict[str, Any]:
        """Simulate session timeout and cleanup."""
        result = {
            "expected_transitions": [
                "timeout_detected",
                "cleanup_executed",
                "state_reset",
            ],
            "actual_transitions": [],
        }

        session_proxy["session_timeout"] = True
        result["actual_transitions"].append("timeout_detected")

        volatile_keys = ["training_active", "training_progress", "error_state"]
        for key in volatile_keys:
            if key in session_proxy:
                del session_proxy[key]
        result["actual_transitions"].append("cleanup_executed")

        if not any(key in session_proxy for key in volatile_keys):
            result["actual_transitions"].append("state_reset")

        return result
