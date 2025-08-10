"""Environment-related utilities for deployment helpers.

These utilities are extracted to reduce LOC in multi-target deployment
manager while keeping behavior unchanged.
"""

from __future__ import annotations

from typing import Any


def compute_resource_issues(
    resource_limits: dict[str, Any] | None,
) -> tuple[list[str], list[str]]:
    """Compute system resource issues and warnings using psutil if available.

    Returns a tuple of (issues, warnings).
    """
    issues: list[str] = []
    warnings: list[str] = []

    if not resource_limits:
        return issues, warnings

    try:
        import psutil  # type: ignore

        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count() or 0
        disk = psutil.disk_usage("/")

        # Guard against None values before int() to satisfy type checker
        mem_val = resource_limits.get("memory_mb", 0)
        cpu_val = resource_limits.get("cpu_cores", 0)
        disk_val = resource_limits.get("disk_gb", 0)
        req_mem_mb = int(mem_val or 0)
        req_cpu = int(cpu_val or 0)
        req_disk_gb = int(disk_val or 0)

        if req_mem_mb and memory.total < req_mem_mb * 1024 * 1024:
            memory_gb = memory.total / (1024**3)
            issues.append(
                f"Insufficient memory: {memory_gb:.1f}GB available, {req_mem_mb}MB required"
            )

        if req_cpu and cpu_count and cpu_count < req_cpu:
            issues.append(
                f"Insufficient CPU cores: {cpu_count} available, {req_cpu} required"
            )

        if req_disk_gb and disk.free < req_disk_gb * 1024 * 1024 * 1024:
            disk_gb = disk.free / (1024**3)
            issues.append(
                f"Insufficient disk space: {disk_gb:.1f}GB available, {req_disk_gb}GB required"
            )

    except Exception:
        warnings.append("psutil not available, skipping resource validation")

    return issues, warnings


def serialize_environment_configs(
    env_configs: dict[Any, Any],
) -> dict[str, Any]:
    """Serialize environment configs to a JSON-friendly dictionary."""
    out: dict[str, Any] = {}
    for env, config in env_configs.items():
        name = getattr(env, "value", str(env))
        out[name] = {
            "deployment_strategy": getattr(
                config.deployment_strategy,
                "value",
                str(config.deployment_strategy),
            ),
            "health_check_timeout": config.health_check_timeout,
            "max_retries": config.max_retries,
            "auto_rollback": config.auto_rollback,
            "performance_thresholds": config.performance_thresholds,
            "resource_limits": config.resource_limits,
            "security_requirements": config.security_requirements,
            "monitoring_config": config.monitoring_config,
        }
    return out
