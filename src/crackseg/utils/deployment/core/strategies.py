from __future__ import annotations

import time
from collections.abc import Callable

from .manager import DeploymentConfig, DeploymentMetadata, DeploymentResult
from .utils import (
    get_current_replicas,
    health_check_deployment,
    monitor_canary_performance,
    remove_current_deployment,
    remove_old_replica,
    switch_traffic,
    update_traffic_split,
)


def blue_green(
    config: DeploymentConfig,
    deployment_func: Callable[..., DeploymentResult],
    metadata: DeploymentMetadata,
    **kwargs,
) -> DeploymentResult:
    metadata.state = metadata.state.IN_PROGRESS  # type: ignore[attr-defined]
    result = deployment_func(config, **kwargs)
    if not result.success or not getattr(result, "deployment_url", None):
        metadata.state = metadata.state.FAILED  # type: ignore[attr-defined]
        metadata.end_time = time.time()
        if not getattr(result, "deployment_url", None):
            result.success = False
            result.error = "Deployment URL not provided"  # type: ignore[attr-defined]
        return result
    if not health_check_deployment(result.deployment_url):
        metadata.state = metadata.state.FAILED  # type: ignore[attr-defined]
        metadata.end_time = time.time()
        result.success = False
        result.error = "Health check failed"  # type: ignore[attr-defined]
        return result
    switch_traffic(config.environment, result.deployment_url)  # type: ignore[attr-defined]
    if getattr(metadata, "previous_deployment_id", None):
        from .utils import decommission_deployment

        decommission_deployment(metadata.previous_deployment_id)  # type: ignore[arg-type]
    metadata.state = metadata.state.SUCCESS  # type: ignore[attr-defined]
    metadata.end_time = time.time()
    result.success = True
    return result


def canary(
    config: DeploymentConfig,
    deployment_func: Callable[..., DeploymentResult],
    metadata: DeploymentMetadata,
    **kwargs,
) -> DeploymentResult:
    metadata.state = metadata.state.IN_PROGRESS  # type: ignore[attr-defined]
    result = deployment_func(config, **kwargs)
    if not result.success or not getattr(result, "deployment_url", None):
        metadata.state = metadata.state.FAILED  # type: ignore[attr-defined]
        metadata.end_time = time.time()
        if not getattr(result, "deployment_url", None):
            result.success = False
            result.error = "Deployment URL not provided"  # type: ignore[attr-defined]
        return result
    if not health_check_deployment(result.deployment_url):
        metadata.state = metadata.state.FAILED  # type: ignore[attr-defined]
        metadata.end_time = time.time()
        result.success = False
        result.error = "Health check failed"  # type: ignore[attr-defined]
        return result
    for percentage in [10, 25, 50, 75, 100]:
        update_traffic_split(result.deployment_url, percentage)
        time.sleep(60)
        if not monitor_canary_performance(result, **kwargs):
            update_traffic_split(result.deployment_url, 0)
            metadata.state = metadata.state.FAILED  # type: ignore[attr-defined]
            metadata.end_time = time.time()
            result.success = False
            result.error = "Performance degradation detected"  # type: ignore[attr-defined]
            return result
    metadata.state = metadata.state.SUCCESS  # type: ignore[attr-defined]
    metadata.end_time = time.time()
    result.success = True
    return result


def rolling(
    config: DeploymentConfig,
    deployment_func: Callable[..., DeploymentResult],
    metadata: DeploymentMetadata,
    **kwargs,
) -> DeploymentResult:
    current_replicas = get_current_replicas(config.environment)  # type: ignore[attr-defined]
    metadata.state = metadata.state.IN_PROGRESS  # type: ignore[attr-defined]
    for i in range(current_replicas):
        result = deployment_func(config, replica_index=i, **kwargs)
        if not result.success or not getattr(result, "deployment_url", None):
            metadata.state = metadata.state.FAILED  # type: ignore[attr-defined]
            metadata.end_time = time.time()
            if not getattr(result, "deployment_url", None):
                result.success = False
                result.error = "Deployment URL not provided"  # type: ignore[attr-defined]
            return result
        if not health_check_deployment(result.deployment_url):
            metadata.state = metadata.state.FAILED  # type: ignore[attr-defined]
            metadata.end_time = time.time()
            result.success = False
            result.error = "Health check failed"  # type: ignore[attr-defined]
            return result
        remove_old_replica(config.environment, i)  # type: ignore[arg-type]
        time.sleep(30)
    metadata.state = metadata.state.SUCCESS  # type: ignore[attr-defined]
    metadata.end_time = time.time()
    result.success = True
    return result


def recreate(
    config: DeploymentConfig,
    deployment_func: Callable[..., DeploymentResult],
    metadata: DeploymentMetadata,
    **kwargs,
) -> DeploymentResult:
    remove_current_deployment(config.environment)  # type: ignore[attr-defined]
    metadata.state = metadata.state.IN_PROGRESS  # type: ignore[attr-defined]
    result = deployment_func(config, **kwargs)
    if not result.success or not getattr(result, "deployment_url", None):
        metadata.state = metadata.state.FAILED  # type: ignore[attr-defined]
        metadata.end_time = time.time()
        if not getattr(result, "deployment_url", None):
            result.success = False
            result.error = "Deployment URL not provided"  # type: ignore[attr-defined]
        return result
    if not health_check_deployment(result.deployment_url):
        metadata.state = metadata.state.FAILED  # type: ignore[attr-defined]
        metadata.end_time = time.time()
        result.success = False
        result.error = "Health check failed"  # type: ignore[attr-defined]
        return result
    metadata.state = metadata.state.SUCCESS  # type: ignore[attr-defined]
    metadata.end_time = time.time()
    result.success = True
    return result
