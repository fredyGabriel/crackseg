"""Configuration dataclasses for validation reporting."""

from dataclasses import dataclass


@dataclass
class ValidationReportData:
    """Data structure for comprehensive validation reports."""

    # Basic information
    artifact_id: str
    target_environment: str
    target_format: str
    validation_timestamp: str
    validation_duration: float

    # Performance metrics
    inference_time_ms: float
    memory_usage_mb: float
    throughput_rps: float
    performance_score: float

    # Security metrics
    security_score: float
    vulnerabilities_found: int
    security_scan_passed: bool

    # Compatibility metrics
    compatibility_score: float
    python_compatible: bool
    dependencies_compatible: bool
    environment_compatible: bool

    # Functional test results
    functional_tests_passed: bool
    test_coverage_percentage: float
    critical_tests_passed: bool

    # Resource utilization
    cpu_usage_percent: float
    gpu_usage_percent: float
    disk_usage_mb: float
    network_bandwidth_mbps: float

    # Deployment readiness
    deployment_ready: bool
    risk_level: str  # "low", "medium", "high", "critical"
    estimated_deployment_time: int  # minutes

    # Recommendations
    recommendations: list[str]
    warnings: list[str]
    critical_issues: list[str]
