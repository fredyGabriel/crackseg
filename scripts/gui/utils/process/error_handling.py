"""Comprehensive error handling system for CrackSeg training execution.

This module provides enhanced error handling capabilities including:
- Rich exception hierarchy with contextual information
- Error recovery mechanisms with automatic retry logic
- Thread-safe error reporting and user feedback
- Integration with UI components for status updates

Designed to handle subprocess failures, parsing errors, user interruptions,
and provide graceful degradation when possible.
"""

import time
import traceback
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .states import ProcessState, TrainingProcessError


class ErrorSeverity(Enum):
    """Error severity levels for prioritizing handling and recovery."""

    LOW = "low"  # Non-critical, recoverable errors
    MEDIUM = "medium"  # Important errors requiring attention
    HIGH = "high"  # Critical errors requiring immediate action
    FATAL = "fatal"  # Unrecoverable errors requiring termination


class ErrorCategory(Enum):
    """Categories of errors for specialized handling."""

    PROCESS_STARTUP = "process_startup"  # Process initialization failures
    SUBPROCESS_EXECUTION = "subprocess_exec"  # Runtime subprocess errors
    CONFIGURATION = "configuration"  # Config/override validation
    LOG_STREAMING = "log_streaming"  # Real-time logging issues
    RESOURCE_MANAGEMENT = "resource_mgmt"  # Memory, file, cleanup issues
    USER_INTERRUPTION = "user_interrupt"  # User-initiated aborts
    SYSTEM_ENVIRONMENT = "system_env"  # OS/environment related
    NETWORK_TIMEOUT = "network_timeout"  # Timeout-related errors


@dataclass
class ErrorContext:
    """Rich context information for error analysis and recovery."""

    timestamp: datetime = field(default_factory=datetime.now)
    category: ErrorCategory = ErrorCategory.SUBPROCESS_EXECUTION
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    component: str = ""  # Module/class where error occurred
    operation: str = ""  # Specific operation that failed
    process_state: ProcessState | None = None
    config_path: Path | None = None
    config_name: str = ""
    overrides: list[str] = field(default_factory=list)
    working_directory: Path | None = None
    process_id: int | None = None
    memory_usage_mb: float | None = None
    disk_space_gb: float | None = None
    additional_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert error context to dictionary for logging/serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "severity": self.severity.value,
            "component": self.component,
            "operation": self.operation,
            "process_state": (
                self.process_state.value if self.process_state else None
            ),
            "config_path": str(self.config_path) if self.config_path else None,
            "config_name": self.config_name,
            "overrides": self.overrides,
            "working_directory": (
                str(self.working_directory) if self.working_directory else None
            ),
            "process_id": self.process_id,
            "memory_usage_mb": self.memory_usage_mb,
            "disk_space_gb": self.disk_space_gb,
            "additional_data": self.additional_data,
        }


class ProcessStartupError(TrainingProcessError):
    """Errors during process initialization and startup.

    Raised when training process fails to start due to:
    - Invalid command construction
    - Missing executable or files
    - Permission issues
    - Resource constraints (memory, disk space)
    """

    def __init__(
        self, message: str, context: ErrorContext | None = None
    ) -> None:
        super().__init__(message)
        self.context = context or ErrorContext(
            category=ErrorCategory.PROCESS_STARTUP, severity=ErrorSeverity.HIGH
        )


class ProcessExecutionError(TrainingProcessError):
    """Errors during subprocess execution.

    Raised when running process encounters runtime failures:
    - Subprocess crashes or exits unexpectedly
    - Resource exhaustion during training
    - GPU/CUDA errors
    - Training algorithm failures
    """

    def __init__(
        self, message: str, context: ErrorContext | None = None
    ) -> None:
        super().__init__(message)
        self.context = context or ErrorContext(
            category=ErrorCategory.SUBPROCESS_EXECUTION,
            severity=ErrorSeverity.HIGH,
        )


class OverrideValidationError(TrainingProcessError):
    """Errors in Hydra override parsing and validation.

    Raised when override strings are malformed or invalid:
    - Syntax errors in override format
    - Type validation failures
    - Security constraint violations
    - Unknown configuration keys
    """

    def __init__(
        self, message: str, context: ErrorContext | None = None
    ) -> None:
        super().__init__(message)
        self.context = context or ErrorContext(
            category=ErrorCategory.CONFIGURATION, severity=ErrorSeverity.MEDIUM
        )


class LogStreamingError(TrainingProcessError):
    """Errors in real-time log streaming system.

    Raised when log capture or streaming fails:
    - File watcher initialization failures
    - Stdout/stderr pipe errors
    - Log parsing or buffering issues
    - Thread synchronization problems
    """

    def __init__(
        self, message: str, context: ErrorContext | None = None
    ) -> None:
        super().__init__(message)
        self.context = context or ErrorContext(
            category=ErrorCategory.LOG_STREAMING, severity=ErrorSeverity.LOW
        )


class ProcessRecoveryError(TrainingProcessError):
    """Errors during automatic error recovery attempts.

    Raised when recovery mechanisms themselves fail:
    - State recovery failures
    - Resource cleanup issues
    - Recovery retry exhaustion
    - Inconsistent system state after recovery
    """

    def __init__(
        self, message: str, context: ErrorContext | None = None
    ) -> None:
        super().__init__(message)
        self.context = context or ErrorContext(
            category=ErrorCategory.RESOURCE_MANAGEMENT,
            severity=ErrorSeverity.HIGH,
        )


class UserInterruptionError(TrainingProcessError):
    """Errors related to user-initiated process interruptions.

    Raised when user actions cause process state issues:
    - Abort operation failures
    - Force termination problems
    - UI synchronization issues during interrupts
    """

    def __init__(
        self, message: str, context: ErrorContext | None = None
    ) -> None:
        super().__init__(message)
        self.context = context or ErrorContext(
            category=ErrorCategory.USER_INTERRUPTION,
            severity=ErrorSeverity.MEDIUM,
        )


class SystemEnvironmentError(TrainingProcessError):
    """Errors related to system environment and resources.

    Raised when system-level issues affect training:
    - Insufficient disk space or memory
    - Missing system dependencies
    - Permission or access control issues
    - OS-specific subprocess limitations
    """

    def __init__(
        self, message: str, context: ErrorContext | None = None
    ) -> None:
        super().__init__(message)
        self.context = context or ErrorContext(
            category=ErrorCategory.SYSTEM_ENVIRONMENT,
            severity=ErrorSeverity.HIGH,
        )


@dataclass
class RecoveryStrategy:
    """Strategy for automatic error recovery."""

    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    exponential_backoff: bool = True
    recoverable_categories: set[ErrorCategory] = field(
        default_factory=lambda: {
            ErrorCategory.LOG_STREAMING,
            ErrorCategory.NETWORK_TIMEOUT,
            ErrorCategory.RESOURCE_MANAGEMENT,
        }
    )
    recovery_callbacks: list[Callable[[Exception, ErrorContext], bool]] = (
        field(default_factory=list)
    )

    def is_recoverable(self, error: Exception, context: ErrorContext) -> bool:
        """Check if error is recoverable based on category and severity."""
        if context.severity == ErrorSeverity.FATAL:
            return False

        return context.category in self.recoverable_categories


# Type alias for error callback functions
ErrorCallback = Callable[[Exception, ErrorContext], None]


class ErrorReporter:
    """Thread-safe error reporting and user feedback system."""

    def __init__(self) -> None:
        """Initialize error reporter."""
        self._callbacks: list[ErrorCallback] = []
        self._error_history: list[tuple[datetime, Exception, ErrorContext]] = (
            []
        )
        self._max_history = 100

    def add_callback(self, callback: ErrorCallback) -> None:
        """Add error callback for UI notifications."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: ErrorCallback) -> None:
        """Remove error callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def report_error(self, error: Exception, context: ErrorContext) -> None:
        """Report error to all registered callbacks and store in history."""
        # Store in history
        self._error_history.append((datetime.now(), error, context))
        if len(self._error_history) > self._max_history:
            self._error_history.pop(0)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(error, context)
            except Exception as callback_error:
                # Prevent callback errors from cascading
                print(f"Error in error callback: {callback_error}")

    def get_error_history(
        self,
    ) -> list[tuple[datetime, Exception, ErrorContext]]:
        """Get recent error history for debugging."""
        return self._error_history.copy()

    def clear_history(self) -> None:
        """Clear error history."""
        self._error_history.clear()


# Global error reporter instance
_global_error_reporter = ErrorReporter()


def get_error_reporter() -> ErrorReporter:
    """Get global error reporter instance."""
    return _global_error_reporter


def create_error_context(
    category: ErrorCategory,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    component: str = "",
    operation: str = "",
    **kwargs: Any,
) -> ErrorContext:
    """Create error context with standard fields."""
    return ErrorContext(
        category=category,
        severity=severity,
        component=component,
        operation=operation,
        **kwargs,
    )


@contextmanager
def error_recovery_context(
    operation: str,
    component: str = "",
    category: ErrorCategory = ErrorCategory.SUBPROCESS_EXECUTION,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    recovery_strategy: RecoveryStrategy | None = None,
    **context_kwargs: Any,
) -> Generator[ErrorContext, None, None]:
    """Context manager for automatic error recovery and reporting.

    Provides structured error handling with automatic retry logic,
    error context enrichment, and recovery attempts.

    Args:
        operation: Description of the operation being performed
        component: Component/module name where operation occurs
        category: Category of error for specialized handling
        severity: Severity level for the operation
        recovery_strategy: Custom recovery strategy, uses default if None
        **context_kwargs: Additional context data

    Yields:
        ErrorContext object that can be enriched during operation

    Example:
        >>> strategy = RecoveryStrategy(max_retries=2)
        >>> with error_recovery_context(
        ...     "start_training",
        ...     component="ProcessManager",
        ...     recovery_strategy=strategy
        ... ) as ctx:
        ...     # Perform risky operation
        ...     ctx.process_id = process.pid
        ...     result = subprocess.run(...)
    """
    context = create_error_context(
        category=category,
        severity=severity,
        component=component,
        operation=operation,
        **context_kwargs,
    )

    strategy = recovery_strategy or RecoveryStrategy()
    error_reporter = get_error_reporter()

    retry_count = 0
    last_exception: Exception | None = None

    while retry_count <= strategy.max_retries:
        try:
            yield context
            return  # Success - exit context

        except Exception as e:
            last_exception = e

            # Enrich context with exception details
            context.additional_data.update(
                {
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "traceback": traceback.format_exc(),
                    "retry_count": retry_count,
                }
            )

            # Report error
            error_reporter.report_error(e, context)

            # Check if recoverable
            if (
                not strategy.is_recoverable(e, context)
                or retry_count >= strategy.max_retries
            ):
                # Re-raise original exception
                raise e

            # Attempt recovery
            for recovery_callback in strategy.recovery_callbacks:
                try:
                    if recovery_callback(e, context):
                        break
                except Exception as recovery_error:
                    # Recovery callback failed, continue to next one
                    context.additional_data[
                        f"recovery_error_{len(strategy.recovery_callbacks)}"
                    ] = str(recovery_error)

            # Wait before retry (with exponential backoff if enabled)
            if retry_count < strategy.max_retries:
                retry_count += 1
                delay = strategy.retry_delay_seconds
                if strategy.exponential_backoff:
                    delay *= 2 ** (retry_count - 1)

                context.additional_data["retry_delay_s"] = delay
                time.sleep(delay)

    # All retries exhausted
    if last_exception:
        raise ProcessRecoveryError(
            f"Operation '{operation}' failed after {strategy.max_retries} "
            f"retries: {last_exception}",
            context,
        )


class ProcessErrorHandler:
    """Centralized error handler for training process management.

    Provides specialized handling for different types of process-related errors
    with appropriate recovery strategies and user feedback.
    """

    def __init__(self) -> None:
        """Initialize process error handler."""
        self._recovery_strategies: dict[ErrorCategory, RecoveryStrategy] = {
            ErrorCategory.LOG_STREAMING: RecoveryStrategy(
                max_retries=2,
                retry_delay_seconds=1.0,
                exponential_backoff=False,
            ),
            ErrorCategory.NETWORK_TIMEOUT: RecoveryStrategy(
                max_retries=3,
                retry_delay_seconds=5.0,
                exponential_backoff=True,
            ),
            ErrorCategory.RESOURCE_MANAGEMENT: RecoveryStrategy(
                max_retries=1,
                retry_delay_seconds=2.0,
                exponential_backoff=False,
            ),
        }

        # Default strategy for categories not explicitly configured
        self._default_strategy = RecoveryStrategy(
            max_retries=1,
            retry_delay_seconds=1.0,
            recoverable_categories=set(),  # Conservative: don't retry by default  # noqa: E501
        )

    def get_recovery_strategy(
        self, category: ErrorCategory
    ) -> RecoveryStrategy:
        """Get recovery strategy for error category."""
        return self._recovery_strategies.get(category, self._default_strategy)

    def set_recovery_strategy(
        self, category: ErrorCategory, strategy: RecoveryStrategy
    ) -> None:
        """Set custom recovery strategy for error category."""
        self._recovery_strategies[category] = strategy

    def handle_process_startup_error(
        self,
        error: Exception,
        config_path: Path,
        config_name: str,
        overrides: list[str],
    ) -> ProcessStartupError:
        """Handle process startup errors with context enrichment."""
        context = create_error_context(
            category=ErrorCategory.PROCESS_STARTUP,
            severity=ErrorSeverity.HIGH,
            component="ProcessManager",
            operation="start_training",
            config_path=config_path,
            config_name=config_name,
            overrides=overrides,
            additional_data={
                "original_error": str(error),
                "original_error_type": type(error).__name__,
            },
        )

        # Report through error reporter
        enhanced_error = ProcessStartupError(
            f"Failed to start training process: {error}", context
        )
        get_error_reporter().report_error(enhanced_error, context)

        return enhanced_error

    def handle_subprocess_execution_error(
        self,
        error: Exception,
        process_id: int | None = None,
        return_code: int | None = None,
    ) -> ProcessExecutionError:
        """Handle subprocess execution errors."""
        context = create_error_context(
            category=ErrorCategory.SUBPROCESS_EXECUTION,
            severity=ErrorSeverity.HIGH,
            component="ProcessManager",
            operation="monitor_process",
            process_id=process_id,
            additional_data={
                "return_code": return_code,
                "original_error": str(error),
                "original_error_type": type(error).__name__,
            },
        )

        enhanced_error = ProcessExecutionError(
            f"Training process execution failed: {error}", context
        )
        get_error_reporter().report_error(enhanced_error, context)

        return enhanced_error

    def handle_log_streaming_error(
        self,
        error: Exception,
        log_source: str = "",
        buffer_size: int | None = None,
    ) -> LogStreamingError:
        """Handle log streaming errors with potential recovery."""
        context = create_error_context(
            category=ErrorCategory.LOG_STREAMING,
            severity=ErrorSeverity.LOW,
            component="LogStreamManager",
            operation="stream_logs",
            additional_data={
                "log_source": log_source,
                "buffer_size": buffer_size,
                "original_error": str(error),
                "original_error_type": type(error).__name__,
            },
        )

        enhanced_error = LogStreamingError(
            f"Log streaming failed: {error}", context
        )
        get_error_reporter().report_error(enhanced_error, context)

        return enhanced_error

    def handle_user_interruption(
        self, error: Exception, interruption_type: str = "abort"
    ) -> UserInterruptionError:
        """Handle user interruption errors."""
        context = create_error_context(
            category=ErrorCategory.USER_INTERRUPTION,
            severity=ErrorSeverity.MEDIUM,
            component="ProcessManager",
            operation=f"user_{interruption_type}",
            additional_data={
                "interruption_type": interruption_type,
                "original_error": str(error),
                "original_error_type": type(error).__name__,
            },
        )

        enhanced_error = UserInterruptionError(
            f"User interruption failed: {error}", context
        )
        get_error_reporter().report_error(enhanced_error, context)

        return enhanced_error


# Global process error handler instance
_global_process_error_handler = ProcessErrorHandler()


def get_process_error_handler() -> ProcessErrorHandler:
    """Get global process error handler instance."""
    return _global_process_error_handler


def create_ui_error_message(
    error: Exception, context: ErrorContext
) -> dict[str, Any]:
    """Create user-friendly error message for UI display.

    Converts technical error information into actionable user feedback
    with suggestions for resolution.

    Args:
        error: The exception that occurred
        context: Rich error context information

    Returns:
        Dictionary with user-friendly error information
    """
    # Base error information
    ui_message: dict[str, Any] = {
        "title": "Training Error",
        "severity": context.severity.value,
        "timestamp": context.timestamp.isoformat(),
        "category": context.category.value,
        "component": context.component or "System",
        "message": f"An error occurred: {str(error)}",
        "suggestions": ["Try the operation again"],
    }

    # Category-specific messages and suggestions
    if context.category == ErrorCategory.PROCESS_STARTUP:
        ui_message["title"] = "Failed to Start Training"
        ui_message["message"] = "The training process could not be started."
        ui_message["suggestions"] = [
            "Check that the configuration file exists and is valid",
            "Verify that Python and required dependencies are installed",
            "Ensure sufficient disk space and memory are available",
            "Check console output for detailed error information",
        ]

    elif context.category == ErrorCategory.SUBPROCESS_EXECUTION:
        ui_message["title"] = "Training Process Failed"
        ui_message["message"] = (
            "The training process encountered an error during execution."
        )
        ui_message["suggestions"] = [
            "Review the training logs for specific error details",
            "Check GPU memory availability if using CUDA",
            "Verify dataset integrity and accessibility",
            "Consider reducing batch size or model complexity",
        ]

    elif context.category == ErrorCategory.CONFIGURATION:
        ui_message.update(
            {
                "title": "Configuration Error",
                "message": "There was an error with the training "
                "configuration.",
                "suggestions": [
                    "Check the syntax of Hydra overrides",
                    "Verify that all configuration keys are valid",
                    "Ensure data types match expected values",
                    "Review the configuration file for errors",
                ],
            }
        )

    elif context.category == ErrorCategory.LOG_STREAMING:
        ui_message.update(
            {
                "title": "Log Streaming Issue",
                "message": "Real-time log streaming encountered an error.",
                "suggestions": [
                    "Training may continue normally in the background",
                    "Check the outputs directory for log files",
                    "Try refreshing the page or restarting the application",
                ],
            }
        )

    elif context.category == ErrorCategory.USER_INTERRUPTION:
        ui_message.update(
            {
                "title": "Process Interruption Error",
                "message": "An error occurred while stopping the training "
                "process.",
                "suggestions": [
                    "The process may still be running in the background",
                    "Try force-stopping through the system task manager",
                    "Restart the application if the interface becomes "
                    "unresponsive",
                ],
            }
        )

    else:
        ui_message.update(
            {
                "message": f"An unexpected error occurred: {str(error)}",
                "suggestions": [
                    "Try the operation again",
                    "Check system resources and permissions",
                    "Restart the application if problems persist",
                    "Contact support if the issue continues",
                ],
            }
        )

    # Add technical details for debugging (optional, can be collapsed in UI)
    ui_message["technical_details"] = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "operation": context.operation,
        "process_state": (
            context.process_state.value if context.process_state else None
        ),
        "additional_data": context.additional_data,
    }

    return ui_message
