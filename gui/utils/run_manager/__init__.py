"""
Training execution orchestration for CrackSeg GUI. This package
provides the main interface for training execution, integrating
override parsing and process management. Acts as the coordination
layer between GUI components and core functionality. The package is
organized by functionality for better maintainability: - orchestrator:
Main training session functions and core logic - streaming_api: Log
streaming functions and callbacks - abort_api: Enhanced abort
functionality and process tree management - ui_integration: UI
responsive functions and threading coordination - session_api: Session
state synchronization and management
"""

# Import all public APIs from submodules
# Re-export core components for backward compatibility
from ..parsing import (
    AdvancedOverrideParser,
    OverrideParsingError,
    ParsedOverride,
)
from ..process import (
    AbortCallback,
    AbortLevel,
    AbortProgress,
    AbortResult,
    ProcessInfo,
    ProcessManager,
    ProcessState,
    TrainingProcessError,
)
from ..session_sync import (
    SessionSyncCoordinator,
    cleanup_session_sync_coordinator,
    get_session_sync_coordinator,
    initialize_session_sync,
)
from ..streaming import (
    LogStreamManager,
    StreamedLog,
)
from ..streaming.core import LogCallback, LogLevel
from ..threading import (
    BackgroundTaskResult,
    CancellationToken,
    ProgressCallback,
    UIResponsiveWrapper,
)
from ..threading.coordinator import TaskPriority, ThreadCoordinator
from .abort_api import (
    abort_training_session,
    force_cleanup_orphans,
    get_process_tree_info,
)
from .orchestrator import (
    cleanup_global_manager,
    get_process_manager,
    get_training_status,
    start_training_session,
    stop_training_session,
    validate_overrides_interactive,
)
from .session_api import (
    force_session_state_sync,
    get_session_state_status,
    initialize_session_state_sync,
)
from .status_integration import (
    add_status_update_callback,
    broadcast_manual_status_update,
    cleanup_status_integration,
    get_comprehensive_status,
    get_status_integration_coordinator,
    initialize_status_integration,
    remove_status_update_callback,
)
from .status_updates import (
    StatusUpdate,
    StatusUpdateCallback,
    StatusUpdateManager,
    StatusUpdateType,
    cleanup_status_update_manager,
    get_status_update_manager,
)
from .streaming_api import (
    add_log_callback,
    clear_log_buffer,
    get_recent_logs,
    get_streaming_status,
    remove_log_callback,
)
from .ui_integration import (
    cleanup_ui_wrapper,
    execute_training_async,  # type: ignore[misc]
    execute_with_progress,  # type: ignore[misc]
    get_ui_wrapper,  # type: ignore[misc]
)
from .ui_status_helpers import (
    check_status_changes_since,
    cleanup_ui_status_callback,
    create_status_callback_for_ui,
    display_metrics_summary,
    display_status_updates_feed,
    display_training_status,
    get_training_status_indicator,
    initialize_ui_status_system,
)

# Complete public API for backward compatibility
__all__ = [
    # Process management
    "ProcessManager",
    "ProcessInfo",
    "ProcessState",
    "TrainingProcessError",
    # Enhanced abort functionality
    "AbortLevel",
    "AbortProgress",
    "AbortResult",
    "AbortCallback",
    # Override parsing
    "AdvancedOverrideParser",
    "ParsedOverride",
    "OverrideParsingError",
    # Log streaming
    "LogStreamManager",
    "StreamedLog",
    "LogLevel",
    "LogCallback",
    # Threading for UI responsiveness
    "ThreadCoordinator",
    "UIResponsiveWrapper",
    "ProgressCallback",
    "CancellationToken",
    "BackgroundTaskResult",
    "TaskPriority",
    # Session synchronization
    "SessionSyncCoordinator",
    "get_session_sync_coordinator",
    "cleanup_session_sync_coordinator",
    "initialize_session_sync",
    # Status updates system
    "StatusUpdate",
    "StatusUpdateCallback",
    "StatusUpdateManager",
    "StatusUpdateType",
    "get_status_update_manager",
    "cleanup_status_update_manager",
    "get_status_integration_coordinator",
    "initialize_status_integration",
    "cleanup_status_integration",
    "add_status_update_callback",
    "remove_status_update_callback",
    "broadcast_manual_status_update",
    "get_comprehensive_status",
    # UI status helpers
    "initialize_ui_status_system",
    "display_training_status",
    "display_status_updates_feed",
    "display_metrics_summary",
    "get_training_status_indicator",
    "create_status_callback_for_ui",
    "cleanup_ui_status_callback",
    "check_status_changes_since",
    # Main functions
    "get_process_manager",
    "cleanup_global_manager",
    "add_log_callback",
    "remove_log_callback",
    "get_recent_logs",
    "clear_log_buffer",
    "get_streaming_status",
    # Enhanced abort functions
    "abort_training_session",
    "get_process_tree_info",
    "force_cleanup_orphans",
    # UI responsive functions
    "get_ui_wrapper",
    "cleanup_ui_wrapper",
    "execute_training_async",
    "execute_with_progress",
    # Session state integration
    "initialize_session_state_sync",
    "get_session_state_status",
    "force_session_state_sync",
    # Core training functions
    "start_training_session",
    "stop_training_session",
    "get_training_status",
    "validate_overrides_interactive",
]
