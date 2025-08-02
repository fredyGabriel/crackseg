"""Configuration classes for deployment system.

This module contains all configuration dataclasses used by the deployment
system to avoid circular import issues.
"""

from dataclasses import dataclass

from .artifact_selector import SelectionCriteria


@dataclass
class DeploymentConfig:
    """Configuration for deployment."""

    # Artifact and environment
    artifact_id: str
    target_environment: str = (
        "production"  # "production", "staging", "development"
    )
    deployment_type: str = "container"  # "container", "serverless", "edge"

    # Optimization settings
    enable_quantization: bool = True
    enable_pruning: bool = False
    target_format: str = "onnx"  # "pytorch", "onnx", "tensorrt", "torchscript"

    # Validation settings
    run_functional_tests: bool = True
    run_performance_tests: bool = True
    run_security_scan: bool = True

    # Monitoring settings
    enable_health_checks: bool = True
    enable_metrics_collection: bool = True

    # Selection criteria
    selection_criteria: SelectionCriteria | None = None


@dataclass
class DeploymentResult:
    """Result of deployment process."""

    success: bool
    deployment_id: str
    artifact_id: str
    target_environment: str

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
