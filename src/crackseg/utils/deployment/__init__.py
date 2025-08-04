"""Deployment utilities for CrackSeg.

This package provides deployment-related utilities organized into specialized
modules for better maintainability and professional structure.
"""

# Core deployment components
# Artifact management
from .artifacts import (
    ArtifactOptimizer,
    ArtifactSelector,
    OptimizationMetricsCollector,
    OptimizationResult,
    OptimizationStrategy,
    OptimizationStrategyFactory,
    OptimizationValidator,
)

# Configuration management
from .config import (
    AlertHandler,
    ConfigurationResult,
    DeploymentConfig,
    DeploymentResult,
    EnvironmentConfig,
    EnvironmentConfigurator,
    ResourceRequirements,
)
from .core import (
    DeploymentManager,
    DeploymentMetadata,
    DeploymentOrchestrator,
    DeploymentState,
    DeploymentStrategy,
)
from .core import (
    DeploymentResult as DeploymentResultType,
)

# Monitoring system (consolidated)
from .monitoring import (
    AlertThresholds,
    DashboardConfig,
    DeploymentMonitoringSystem,
    HealthCheckConfig,
    HealthChecker,
    MetricsCollector,
    MetricsConfig,
    MonitoringResult,
    PerformanceMonitor,
    ResourceMetrics,
    ResourceMonitor,
)

# Packaging system
from .packaging import (
    DependencyManager,
    DockerComposeGenerator,
    FileGenerator,
    HelmChartGenerator,
    KubernetesManifestGenerator,
    PackagingSystem,
)
from .packaging import (
    MetricsCalculator as PackagingMetricsCollector,
)

# Utility components
from .utils import (
    MultiTargetDeploymentManager,
    ProductionReadinessValidator,
)

# Validation system (unified pipeline + reporting)
from .validation import (
    CompatibilityChecker,
    FunctionalTestRunner,
    PerformanceBenchmarker,
    RiskAnalyzer,
    SecurityScanner,
    ValidationPipeline,
    ValidationReportData,
    ValidationReporter,
    ValidationResult,
    ValidationThresholds,
)

__all__ = [
    # Core deployment types
    "DeploymentConfig",
    "DeploymentResult",
    "DeploymentManager",
    "DeploymentOrchestrator",
    "DeploymentMetadata",
    "DeploymentResultType",
    "DeploymentState",
    "DeploymentStrategy",
    # Configuration
    "EnvironmentConfigurator",
    "EnvironmentConfig",
    "ResourceRequirements",
    "ConfigurationResult",
    "AlertHandler",
    # Artifacts
    "ArtifactOptimizer",
    "OptimizationStrategy",
    "OptimizationResult",
    "OptimizationStrategyFactory",
    "OptimizationValidator",
    "OptimizationMetricsCollector",
    "ArtifactSelector",
    # Validation
    "ValidationPipeline",
    "ValidationResult",
    "ValidationThresholds",
    "FunctionalTestRunner",
    "PerformanceBenchmarker",
    "SecurityScanner",
    "CompatibilityChecker",
    "ValidationReporter",
    "ValidationReportData",
    "RiskAnalyzer",
    # Monitoring
    "DeploymentMonitoringSystem",
    "MonitoringResult",
    "HealthChecker",
    "MetricsCollector",
    "PerformanceMonitor",
    "ResourceMonitor",
    "HealthCheckConfig",
    "MetricsConfig",
    "DashboardConfig",
    "AlertThresholds",
    "ResourceMetrics",
    # Utilities
    "MultiTargetDeploymentManager",
    "ProductionReadinessValidator",
    # Packaging
    "PackagingSystem",
    "DockerComposeGenerator",
    "KubernetesManifestGenerator",
    "HelmChartGenerator",
    "SecurityScanner",
    "DependencyManager",
    "FileGenerator",
    "PackagingMetricsCollector",
]
