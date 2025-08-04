"""Configuration classes for deployment system.

This module contains all configuration dataclasses used by the deployment
system to avoid circular import issues.
"""

from dataclasses import dataclass


# Simple SelectionCriteria definition for compatibility
@dataclass
class SelectionCriteria:
    """Criteria for artifact selection."""

    min_accuracy: float = 0.0
    max_inference_time_ms: float = 1000.0
    max_memory_usage_mb: float = 2048.0
    max_model_size_mb: float = 500.0
    preferred_format: str = "pytorch"
    target_environment: str = "production"
    deployment_type: str = "container"


@dataclass
class DeploymentResult:
    """Result of deployment operation."""

    success: bool
    deployment_id: str
    message: str
    error_message: str | None = None


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
