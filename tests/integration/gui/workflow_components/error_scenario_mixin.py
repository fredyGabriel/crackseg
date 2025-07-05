"""Error scenario mixin for modular integration testing.

This module provides reusable error testing patterns that can be mixed into
workflow components to add comprehensive error scenario capabilities.
"""

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch


class ErrorScenarioMixin:
    """Mixin providing reusable error testing patterns for workflow
    components."""

    def create_corrupted_config_file(
        self, corruption_type: str = "yaml_syntax"
    ) -> Path:
        """Create various types of corrupted configuration files.

        Args:
            corruption_type: Type of corruption to introduce

        Returns:
            Path to corrupted config file
        """
        temp_dir = getattr(self, "temp_path", Path(tempfile.mkdtemp()))
        corrupted_file = temp_dir / f"corrupted_{corruption_type}.yaml"

        corruption_patterns = {
            "yaml_syntax": "invalid_yaml: [unclosed list\nmodel: test",
            "missing_required": "# Missing required sections\noptional: value",
            "invalid_values": (
                "model:\n  epochs: 'not_a_number'\n  learning_rate: -1"
            ),
            "circular_reference": (
                "defaults:\n  - _self_\n  - config: ${defaults}"
            ),
            "empty_file": "",
            "binary_data": b"\x00\x01\x02\xff\xfe\xfd",
        }

        content = corruption_patterns.get(
            corruption_type, corruption_patterns["yaml_syntax"]
        )

        if isinstance(content, bytes):
            with open(corrupted_file, "wb") as f:
                f.write(content)
        else:
            with open(corrupted_file, "w", encoding="utf-8") as f:
                f.write(content)

        return corrupted_file

    def simulate_file_system_error(
        self,
        error_type: str = "permission_denied",
        target_path: Path | None = None,
    ) -> dict[str, Any]:
        """Simulate various file system error conditions.

        Args:
            error_type: Type of file system error to simulate
            target_path: Target path for error simulation

        Returns:
            Error simulation context and cleanup function
        """
        error_context: dict[str, Any] = {
            "error_type": error_type,
            "target_path": str(target_path) if target_path else None,
            "mock_objects": [],
            "cleanup_functions": [],
        }

        if error_type == "permission_denied":
            # Simulate permission denied error
            def permission_error(*args: Any, **kwargs: Any) -> None:
                raise PermissionError("Permission denied")

            mock_open = patch("builtins.open", side_effect=permission_error)
            error_context["mock_objects"].append(mock_open)

        elif error_type == "disk_full":
            # Simulate disk full error
            def disk_full_error(*args: Any, **kwargs: Any) -> None:
                raise OSError("No space left on device")

            mock_write = patch(
                "pathlib.Path.write_text", side_effect=disk_full_error
            )
            error_context["mock_objects"].append(mock_write)

        elif error_type == "file_not_found":
            # Simulate file not found
            def file_not_found(*args: Any, **kwargs: Any) -> None:
                raise FileNotFoundError("File not found")

            mock_open = patch("builtins.open", side_effect=file_not_found)
            error_context["mock_objects"].append(mock_open)

        elif error_type == "network_timeout":
            # Simulate network timeout for external resources
            def timeout_error(*args: Any, **kwargs: Any) -> None:
                raise TimeoutError("Network timeout")

            mock_request = patch("requests.get", side_effect=timeout_error)
            error_context["mock_objects"].append(mock_request)

        return error_context

    def execute_error_recovery_test(
        self,
        error_function: Callable[[], Any],
        recovery_function: Callable[[], Any],
        expected_recovery_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Test error recovery mechanisms.

        Args:
            error_function: Function that should trigger an error
            recovery_function: Function that should handle recovery
            expected_recovery_state: Expected state after recovery

        Returns:
            Recovery test results
        """
        recovery_result: dict[str, Any] = {
            "error_triggered": False,
            "recovery_attempted": False,
            "recovery_successful": False,
            "final_state": {},
            "error_details": None,
        }

        try:
            # Step 1: Trigger error condition
            error_function()
            recovery_result["error_triggered"] = False  # Should not reach here
        except Exception as e:
            recovery_result["error_triggered"] = True
            recovery_result["error_details"] = str(e)

            try:
                # Step 2: Attempt recovery
                recovery_result["recovery_attempted"] = True
                final_state = recovery_function()
                recovery_result["final_state"] = final_state

                # Step 3: Validate recovery
                recovery_successful = all(
                    recovery_result["final_state"].get(key) == value
                    for key, value in expected_recovery_state.items()
                )
                recovery_result["recovery_successful"] = recovery_successful

            except Exception as recovery_error:
                recovery_result["recovery_error"] = str(recovery_error)

        return recovery_result

    def simulate_vram_exhaustion(
        self, model_size_mb: int = 9000
    ) -> dict[str, Any]:
        """Simulate VRAM exhaustion for RTX 3070 Ti (8GB limit).

        Args:
            model_size_mb: Simulated model size in MB

        Returns:
            VRAM exhaustion simulation result
        """
        vram_simulation: dict[str, Any] = {
            "model_size_mb": model_size_mb,
            "vram_limit_mb": 8192,  # RTX 3070 Ti limit
            "vram_exhausted": model_size_mb > 8192,
            "error_triggered": False,
            "fallback_activated": False,
        }

        if vram_simulation["vram_exhausted"]:
            # Simulate CUDA out of memory error
            vram_simulation["error_triggered"] = True
            vram_simulation["error_message"] = (
                f"CUDA out of memory. Tried to allocate {model_size_mb}MB, "
                f"but only {8192}MB available."
            )

            # Simulate fallback to CPU
            vram_simulation["fallback_activated"] = True
            vram_simulation["fallback_device"] = "cpu"

        return vram_simulation

    def validate_error_isolation(
        self, test_function: Callable[[], Any], shared_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate that error scenarios don't affect other tests.

        Args:
            test_function: Test function that might cause side effects
            shared_state: Shared state to monitor for contamination

        Returns:
            Isolation validation results
        """
        isolation_result: dict[str, Any] = {
            "initial_state": shared_state.copy(),
            "test_executed": False,
            "state_contaminated": False,
            "cleanup_required": False,
            "final_state": {},
        }

        try:
            # Execute test function
            test_function()
            isolation_result["test_executed"] = True

            # Check for state contamination
            isolation_result["final_state"] = shared_state.copy()
            state_changed = (
                isolation_result["initial_state"]
                != isolation_result["final_state"]
            )
            isolation_result["state_contaminated"] = state_changed

            if state_changed:
                isolation_result["cleanup_required"] = True

        except Exception as e:
            isolation_result["test_error"] = str(e)

        return isolation_result
