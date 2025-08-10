from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

"""Compatibility types for deployment APIs.

Bridges historical imports from `crackseg.utils.deployment.config.deployment`
and the core manager types, so tests and examples can use either.
"""


# Canonical definitions used across the codebase
@dataclass
class DeploymentConfig:
    artifact_id: str
    target_environment: str = "production"
    deployment_type: str = "container"


class DeploymentStrategy(Enum):
    BLUE_GREEN = "blue-green"
    CANARY = "canary"
    ROLLING = "rolling"
    RECREATE = "recreate"


# Backwards-compat aliases expected by some scripts/tests
ConfigType = DeploymentConfig
StrategyType = DeploymentStrategy

__all__ = [
    "DeploymentConfig",
    "DeploymentStrategy",
    "ConfigType",
    "StrategyType",
]


# (kept for historical context; primary enum declared above)


class DeploymentState(Enum):
    """Deployment states for tracking progress."""

    PENDING = "pending"
    IN_PROGRESS = "in-progress"
    HEALTH_CHECKING = "health-checking"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLING_BACK = "rolling-back"
    ROLLED_BACK = "rolled-back"


@dataclass
class DeploymentMetadata:
    """Metadata for tracking deployment information."""

    deployment_id: str
    artifact_id: str
    strategy: DeploymentStrategy
    state: DeploymentState
    start_time: float
    end_time: float | None = None
    previous_deployment_id: str | None = None
    rollback_reason: str | None = None
    health_check_url: str | None = None
    metrics_url: str | None = None


@dataclass
class DeploymentResult:
    """Result of deployment process."""

    success: bool
    deployment_id: str
    artifact_id: str
    target_environment: str
    message: str = ""

    # Optimization results
    original_size_mb: float = 0.0
    optimized_size_mb: float = 0.0
    compression_ratio: float = 1.0

    # Validation results
    functional_tests_passed: bool = False
    performance_benchmark_score: float = 0.0
    security_scan_passed: bool = False

    # Deployment URLs
    deployment_url: str | None = None
    health_check_url: str | None = None
    monitoring_dashboard_url: str | None = None

    # Error information
    error_message: str | None = None
    metadata: dict[str, Any] | None = None
    duration: float | None = None
