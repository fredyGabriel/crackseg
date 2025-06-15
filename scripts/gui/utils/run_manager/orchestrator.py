"""Core training session orchestration and process management.

This module provides the main functions for starting, stopping, and
monitoring training sessions, along with global process manager singleton
pattern for GUI integration.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..parsing import AdvancedOverrideParser, OverrideParsingError
from ..process import ProcessManager, TrainingProcessError
from ..process.error_handling import (
    ErrorCategory,
    ErrorSeverity,
    create_error_context,
    create_ui_error_message,
    error_recovery_context,
    get_error_reporter,
    get_process_error_handler,
)

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
    validation, and process execution for the GUI with enhanced
    error handling and user-friendly feedback.

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
    error_handler = get_process_error_handler()

    # Check if already running
    if manager.is_running:
        return False, ["Training process is already running"]

    try:
        with error_recovery_context(
            operation="start_training_session",
            component="TrainingOrchestrator",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            config_path=config_path,
            config_name=config_name,
        ) as _:
            # Parse and validate overrides if provided
            override_list: list[str] = []
            if overrides_text.strip():
                override_list, parse_errors = manager.parse_overrides_text(
                    overrides_text, validate_overrides
                )

                if parse_errors:
                    # Create user-friendly error messages for parsing errors
                    ui_errors = []
                    for error in parse_errors:
                        ui_errors.append(f"Override error: {error}")
                    return False, ui_errors

            # Update context with override information
            # Note: context enrichment handled automatically by context manager

            # Start the training process
            success = manager.start_training(
                config_path=config_path,
                config_name=config_name,
                overrides=override_list,
                working_dir=working_dir,
            )

            if not success:
                error_msg = (
                    manager.process_info.error_message or "Unknown error"
                )
                return False, [error_msg]

            return True, []

    except (TrainingProcessError, OverrideParsingError) as e:
        # Enhanced error handling with UI-friendly messages
        # Create error context for UI display

        error_context = create_error_context(
            category=(
                ErrorCategory.CONFIGURATION
                if isinstance(e, OverrideParsingError)
                else ErrorCategory.PROCESS_STARTUP
            ),
            component="TrainingOrchestrator",
            operation="start_training_session",
        )

        ui_message = create_ui_error_message(e, error_context)
        error_text = (
            f"{ui_message.get('title', 'Error')}: "
            f"{ui_message.get('message', str(e))}"
        )

        # Add suggestions if available
        suggestions = ui_message.get("suggestions", [])
        if suggestions:
            error_text += "\n\nSuggestions:\n" + "\n".join(
                f"• {s}" for s in suggestions[:3]
            )

        return False, [error_text]

    except Exception as e:
        # Handle unexpected errors
        enhanced_error = error_handler.handle_process_startup_error(
            e,
            config_path,
            config_name,
            overrides_text.split() if overrides_text else [],
        )

        ui_message = create_ui_error_message(
            enhanced_error, enhanced_error.context
        )
        error_text = (
            f"{ui_message.get('title', 'Unexpected Error')}: "
            f"{ui_message.get('message', str(e))}"
        )

        return False, [error_text]


def stop_training_session(timeout: float = 30.0) -> tuple[bool, list[str]]:
    """Stop the current training session with enhanced error handling.

    Args:
        timeout: Maximum time to wait for graceful shutdown

    Returns:
        Tuple of (success, error_messages)
    """
    manager = get_process_manager()
    error_handler = get_process_error_handler()

    if not manager.is_running:
        return True, []

    try:
        with error_recovery_context(
            operation="stop_training_session",
            component="TrainingOrchestrator",
            category=ErrorCategory.USER_INTERRUPTION,
            severity=ErrorSeverity.MEDIUM,
            process_id=manager.process_info.pid,
        ) as _:
            success = manager.stop_training(timeout)

            if not success:
                error_msg = (
                    manager.process_info.error_message
                    or "Failed to stop training"
                )
                return False, [error_msg]

            return True, []

    except Exception as e:
        # Enhanced error handling for stop failures
        enhanced_error = error_handler.handle_user_interruption(
            e, "stop_session"
        )

        ui_message = create_ui_error_message(
            enhanced_error, enhanced_error.context
        )
        error_text = (
            f"{ui_message.get('title', 'Stop Error')}: "
            f"{ui_message.get('message', str(e))}"
        )

        return False, [error_text]


def get_training_status() -> dict[str, Any]:
    """Get current training status with error history.

    Returns:
        Dictionary with training status and recent errors
    """
    manager = get_process_manager()
    error_reporter = get_error_reporter()

    process_info = manager.process_info

    status: dict[str, Any] = {
        "is_running": manager.is_running,
        "state": process_info.state.value if process_info.state else "unknown",
        "pid": process_info.pid,
        "start_time": process_info.start_time,
        "return_code": process_info.return_code,
        "error_message": process_info.error_message,
        "working_directory": (
            str(process_info.working_directory)
            if process_info.working_directory
            else None
        ),
    }

    # Add recent error history for debugging
    error_history = error_reporter.get_error_history()
    recent_errors = []

    for timestamp, error, context in error_history[-5:]:  # Last 5 errors
        recent_errors.append(
            {
                "timestamp": timestamp.isoformat(),
                "error_type": type(error).__name__,
                "message": str(error),
                "category": context.category.value,
                "severity": context.severity.value,
                "component": context.component,
                "operation": context.operation,
            }
        )

    status["recent_errors"] = recent_errors

    return status


def clear_error_history() -> None:
    """Clear the error history for fresh start."""
    error_reporter = get_error_reporter()
    error_reporter.clear_history()


def register_error_callback(
    callback: Callable[[Exception, Any], None],
) -> None:
    """Register a callback for error notifications.

    Args:
        callback: Function that takes (Exception, ErrorContext) parameters
    """
    error_reporter = get_error_reporter()
    error_reporter.add_callback(callback)


def unregister_error_callback(
    callback: Callable[[Exception, Any], None],
) -> None:
    """Unregister an error callback.

    Args:
        callback: Previously registered callback function
    """
    error_reporter = get_error_reporter()
    error_reporter.remove_callback(callback)


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
