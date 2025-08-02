"""Deployment system for CrackSeg.

This package provides comprehensive deployment capabilities including
artifact selection, environment configuration, optimization, packaging,
validation, monitoring, and orchestration with rollback mechanisms.
"""

from .artifact_optimizer import ArtifactOptimizer
from .artifact_selector import ArtifactSelector, SelectionCriteria
from .config import DeploymentConfig, DeploymentResult
from .deployment_manager import DeploymentManager
from .environment_configurator import EnvironmentConfigurator
from .monitoring_system import DeploymentMonitoringSystem
from .orchestration import (
    DefaultHealthChecker,
    DeploymentMetadata,
    DeploymentOrchestrator,
    DeploymentState,
    DeploymentStrategy,
    HealthChecker,
)
from .packaging_system import PackagingSystem
from .production_readiness_validator import (
    ProductionReadinessCriteria,
    ProductionReadinessResult,
    ProductionReadinessValidator,
)
from .validation_pipeline import ValidationPipeline

__all__ = [
    # Core components
    "DeploymentManager",
    "DeploymentConfig",
    "DeploymentResult",
    # Artifact management
    "ArtifactSelector",
    "SelectionCriteria",
    "ArtifactOptimizer",
    # Environment and packaging
    "EnvironmentConfigurator",
    "PackagingSystem",
    # Validation and monitoring
    "ValidationPipeline",
    "DeploymentMonitoringSystem",
    # Production readiness validation
    "ProductionReadinessValidator",
    "ProductionReadinessCriteria",
    "ProductionReadinessResult",
    # Orchestration and rollback
    "DeploymentOrchestrator",
    "DeploymentStrategy",
    "DeploymentState",
    "DeploymentMetadata",
    "HealthChecker",
    "DefaultHealthChecker",
]
