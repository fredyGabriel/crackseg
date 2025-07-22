"""
E2E Resource Cleanup Automation System. This package provides
automated resource cleanup procedures to ensure test environments are
properly reset between runs. Integrates with ResourceMonitor for
real-time validation and comprehensive resource management. Features:
- Automated cleanup coordination with ResourceMonitor integration -
Specialized resource cleanup procedures for files, processes,
connections - Validation system with rollback capabilities - Test
environment reset automation Architecture: - CleanupManager: Main
coordinator with ResourceMonitor integration - ResourceCleanup:
Specialized cleanup procedures - ValidationSystem: Post-cleanup
validation and rollback
"""

from .cleanup_manager import (
    CleanupConfig,
    CleanupManager,
    CleanupResult,
    CleanupStatus,
)
from .resource_cleanup import (
    FileCleanup,
    NetworkCleanup,
    ProcessCleanup,
    ResourceCleanupRegistry,
    TempFileCleanup,
)
from .validation_system import (
    CleanupValidator,
    RollbackManager,
    ValidationResult,
    ValidationStatus,
    validate_and_rollback,
)

__all__ = [
    # Core management
    "CleanupManager",
    "CleanupConfig",
    "CleanupResult",
    "CleanupStatus",
    # Resource cleanup procedures
    "ResourceCleanupRegistry",
    "FileCleanup",
    "ProcessCleanup",
    "NetworkCleanup",
    "TempFileCleanup",
    # Validation system
    "CleanupValidator",
    "ValidationResult",
    "ValidationStatus",
    "RollbackManager",
    "validate_and_rollback",
]
