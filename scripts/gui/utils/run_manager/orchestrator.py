"""Core training session orchestration and process management.

This module provides the main functions for starting, stopping, and
monitoring training sessions, along with global process manager singleton
pattern for GUI integration.
"""

from pathlib import Path
from typing import Any

from ..parsing import AdvancedOverrideParser, OverrideParsingError
from ..process import ProcessManager, TrainingProcessError

# Global instance for singleton pattern
_global_manager: ProcessManager | None = None


def get_process_manager() -> ProcessManager:
    """Get or create the global process manager instance.

    Provides singleton access to the process manager to ensure
    only one training process can run at a time across the GUI.

    Returns:
        Global ProcessManager instance
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = ProcessManager()
    return _global_manager


def cleanup_global_manager() -> None:
    """Clean up and reset the global process manager.

    Should be called when the GUI is shutting down or when
    a clean reset is needed. Stops any running processes.
    """
    global _global_manager
    if _global_manager is not None:
        if _global_manager.is_running:
            _global_manager.stop_training(timeout=10.0)
        _global_manager = None


def start_training_session(
    config_path: Path,
    config_name: str,
    overrides_text: str = "",
    working_dir: Path | None = None,
    validate_overrides: bool = True,
) -> tuple[bool, list[str]]:
    """Start a new training session with validation.

    High-level function that orchestrates override parsing,
    validation, and process execution for the GUI.

    Args:
        config_path: Path to configuration directory
        config_name: Configuration file name (without .yaml)
        overrides_text: Raw override text from GUI input
        working_dir: Working directory for execution
        validate_overrides: Whether to validate override types

    Returns:
        Tuple of (success, error_messages)

    Example:
        >>> success, errors = start_training_session(
        ...     Path("configs"),
        ...     "train_baseline",
        ...     "trainer.max_epochs=50 model.encoder=resnet50"
        ... )
    """
    manager = get_process_manager()

    # Check if already running
    if manager.is_running:
        return False, ["Training process is already running"]

    try:
        # Parse and validate overrides if provided
        override_list: list[str] = []
        if overrides_text.strip():
            override_list, parse_errors = manager.parse_overrides_text(
                overrides_text, validate_overrides
            )

            if parse_errors:
                return False, parse_errors

        # Start the training process
        success = manager.start_training(
            config_path=config_path,
            config_name=config_name,
            overrides=override_list,
            working_dir=working_dir,
        )

        if not success:
            error_msg = manager.process_info.error_message or "Unknown error"
            return False, [error_msg]

        return True, []

    except (TrainingProcessError, OverrideParsingError) as e:
        return False, [str(e)]
    except Exception as e:
        return False, [f"Unexpected error: {e}"]


def stop_training_session(timeout: float = 30.0) -> bool:
    """Stop the current training session gracefully.

    Args:
        timeout: Maximum time to wait for graceful shutdown

    Returns:
        True if stopped successfully, False otherwise
    """
    manager = get_process_manager()
    return manager.stop_training(timeout)


def get_training_status() -> dict[str, Any]:
    """Get comprehensive training process status.

    Returns:
        Dictionary with process info, memory usage, and state
    """
    manager = get_process_manager()
    process_info = manager.process_info
    memory_usage = manager.get_memory_usage()

    return {
        "state": process_info.state.value,
        "is_running": manager.is_running,
        "pid": process_info.pid,
        "start_time": process_info.start_time,
        "return_code": process_info.return_code,
        "error_message": process_info.error_message,
        "command": process_info.command,
        "working_directory": (
            str(process_info.working_directory)
            if process_info.working_directory
            else None
        ),
        "memory_usage": memory_usage,
    }


def validate_overrides_interactive(overrides_text: str) -> dict[str, Any]:
    """Validate overrides and return detailed results for GUI feedback.

    Args:
        overrides_text: Raw override text to validate

    Returns:
        Dictionary with validation results, errors, and suggestions
    """
    parser = AdvancedOverrideParser()

    try:
        parsed_overrides = parser.parse_overrides(
            overrides_text, validate_types=True
        )
        valid_overrides = parser.get_valid_overrides()
        parsing_errors = parser.get_parsing_errors()

        # Categorize results
        valid_count = len([o for o in parsed_overrides if o.is_valid])
        invalid_count = len([o for o in parsed_overrides if not o.is_valid])

        return {
            "is_valid": invalid_count == 0,
            "total_overrides": len(parsed_overrides),
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "valid_overrides": valid_overrides,
            "errors": parsing_errors,
            "parsed_details": [
                {
                    "key": override.key,
                    "value": override.value,
                    "raw_value": override.raw_value,
                    "type": override.override_type,
                    "is_valid": override.is_valid,
                    "error": override.error_message,
                }
                for override in parsed_overrides
            ],
        }

    except OverrideParsingError as e:
        return {
            "is_valid": False,
            "total_overrides": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "valid_overrides": [],
            "errors": [str(e)],
            "parsed_details": [],
        }
