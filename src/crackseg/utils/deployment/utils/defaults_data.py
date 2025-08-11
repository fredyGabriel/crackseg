"""Default environment configuration data for multi-target deployment.

This module provides plain-Python data (no imports of project classes) to avoid
import cycles. The consumer is responsible for converting these dictionaries
into concrete configuration objects/enums.
"""

from __future__ import annotations

from typing import Any


def get_default_environment_data() -> dict[str, dict[str, Any]]:
    """Return default environment configuration data keyed by environment name.

    Notes:
        - "deployment_strategy" is the enum name as a string (e.g., "RECREATE").
        - Other fields are plain dictionaries/primitives.
    """
    return {
        "development": {
            "deployment_strategy": "RECREATE",
            "health_check_timeout": 10,
            "max_retries": 1,
            "auto_rollback": False,
            "performance_thresholds": {
                "response_time_ms": 1000,
                "memory_usage_mb": 2048,
                "cpu_usage_percent": 80,
            },
            "resource_limits": {
                "memory_mb": 2048,
                "cpu_cores": 2,
                "disk_gb": 10,
            },
            "security_requirements": {
                "ssl_required": False,
                "authentication_required": False,
            },
            "monitoring_config": {
                "check_interval": 60,
                "alert_threshold": 0.8,
            },
        },
        "staging": {
            "deployment_strategy": "BLUE_GREEN",
            "health_check_timeout": 30,
            "max_retries": 2,
            "auto_rollback": True,
            "performance_thresholds": {
                "response_time_ms": 500,
                "memory_usage_mb": 4096,
                "cpu_usage_percent": 70,
            },
            "resource_limits": {
                "memory_mb": 4096,
                "cpu_cores": 4,
                "disk_gb": 20,
            },
            "security_requirements": {
                "ssl_required": True,
                "authentication_required": True,
            },
            "monitoring_config": {
                "check_interval": 30,
                "alert_threshold": 0.9,
            },
        },
        "production": {
            "deployment_strategy": "CANARY",
            "health_check_timeout": 60,
            "max_retries": 3,
            "auto_rollback": True,
            "performance_thresholds": {
                "response_time_ms": 200,
                "memory_usage_mb": 8192,
                "cpu_usage_percent": 60,
            },
            "resource_limits": {
                "memory_mb": 8192,
                "cpu_cores": 8,
                "disk_gb": 50,
            },
            "security_requirements": {
                "ssl_required": True,
                "authentication_required": True,
                "encryption_required": True,
            },
            "monitoring_config": {
                "check_interval": 15,
                "alert_threshold": 0.95,
            },
        },
        "testing": {
            "deployment_strategy": "RECREATE",
            "health_check_timeout": 15,
            "max_retries": 1,
            "auto_rollback": False,
            "performance_thresholds": {
                "response_time_ms": 2000,
                "memory_usage_mb": 1024,
                "cpu_usage_percent": 90,
            },
            "resource_limits": {
                "memory_mb": 1024,
                "cpu_cores": 1,
                "disk_gb": 5,
            },
            "security_requirements": {
                "ssl_required": False,
                "authentication_required": False,
            },
            "monitoring_config": {
                "check_interval": 120,
                "alert_threshold": 0.5,
            },
        },
        "demo": {
            "deployment_strategy": "ROLLING",
            "health_check_timeout": 20,
            "max_retries": 2,
            "auto_rollback": True,
            "performance_thresholds": {
                "response_time_ms": 800,
                "memory_usage_mb": 3072,
                "cpu_usage_percent": 75,
            },
            "resource_limits": {
                "memory_mb": 3072,
                "cpu_cores": 2,
                "disk_gb": 15,
            },
            "security_requirements": {
                "ssl_required": True,
                "authentication_required": False,
            },
            "monitoring_config": {
                "check_interval": 45,
                "alert_threshold": 0.8,
            },
        },
    }
