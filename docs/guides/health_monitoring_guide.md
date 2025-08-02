# Health Monitoring System Guide

This guide covers the comprehensive health monitoring system integrated with the CrackSeg deployment
orchestration.

## Overview

The health monitoring system provides continuous monitoring of deployed artifacts with:

- **Health Checks**: HTTP-based endpoint monitoring
- **Resource Monitoring**: CPU, memory, disk, and network metrics
- **Alert System**: Configurable thresholds and notifications
- **Integration**: Seamless integration with deployment orchestration

## Architecture

```bash
┌─────────────────────────────────────────────────────────────┐
│                 DeploymentOrchestrator                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Health Monitor  │  │ Performance     │  │ Alert       │ │
│  │                 │  │ Monitor         │  │ Handlers    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              DeploymentHealthMonitor                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Health Checker  │  │ Resource        │  │ Alert       │ │
│  │                 │  │ Monitor         │  │ System      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Health Monitoring

```python
from crackseg.utils.deployment.orchestration import DeploymentOrchestrator

from crackseg.utils.deployment.health_monitoring import DeploymentHealthMonitor


# Initialize orchestrator with health monitoring
orchestrator = DeploymentOrchestrator()

# Add deployment to monitoring
orchestrator.add_deployment_monitoring(
    deployment_id="my-deployment",
    health_check_url="http://localhost:8080/health",
    process_name="my-service",
    check_interval=30,
)

# Start monitoring
orchestrator.start_health_monitoring()

# Get health status
status = orchestrator.get_deployment_health_status("my-deployment")
print(f"Health: {status['health_status']}")

# Stop monitoring
orchestrator.stop_health_monitoring()
```

### Standalone Health Monitor

```python
from crackseg.utils.deployment.health_monitoring import DeploymentHealthMonitor


# Create standalone monitor
monitor = DeploymentHealthMonitor()

# Add multiple deployments
monitor.add_deployment_monitoring(
    deployment_id="web-service",
    health_check_url="http://localhost:8080/health",
    process_name="web-service",
)

monitor.add_deployment_monitoring(
    deployment_id="api-service",
    health_check_url="http://localhost:8081/health",
    process_name="api-service",
)

# Start monitoring
monitor.start_monitoring()

# Get all statuses
all_statuses = monitor.get_all_deployment_statuses()
for deployment_id, status in all_statuses.items():
    print(f"{deployment_id}: {status['health_status']}")
```

## Components

### Health Checker

The `DefaultHealthChecker` provides HTTP-based health checking:

```python
from crackseg.utils.deployment.health_monitoring import DefaultHealthChecker


checker = DefaultHealthChecker()

# Check health
result = checker.check_health("http://localhost:8080/health")
if result.success:
    print(f"Healthy - Response time: {result.response_time_ms}ms")
else:
    print(f"Unhealthy - Error: {result.error_message}")

# Wait for healthy
if checker.wait_for_healthy("http://localhost:8080/health", max_wait=300):
    print("Service is healthy")
else:
    print("Service failed to become healthy")
```

### Resource Monitor

The `DefaultResourceMonitor` provides system and process metrics:

```python
from crackseg.utils.deployment.health_monitoring import DefaultResourceMonitor


monitor = DefaultResourceMonitor()

# Get system metrics
system_metrics = monitor.get_system_metrics()
print(f"CPU: {system_metrics.cpu_usage_percent}%")
print(f"Memory: {system_metrics.memory_usage_mb}MB")
print(f"Disk: {system_metrics.disk_usage_percent}%")
print(f"Network: {system_metrics.network_io_mbps}Mbps")

# Get process metrics
process_metrics = monitor.get_process_metrics("python")
print(f"Process CPU: {process_metrics.cpu_usage_percent}%")
print(f"Process Memory: {process_metrics.memory_usage_mb}MB")
```

### Custom Implementations

You can implement custom health checkers and resource monitors:

```python
from crackseg.utils.deployment.health_monitoring import (
    HealthChecker, ResourceMonitor, HealthCheckResult, ResourceMetrics
)

class CustomHealthChecker(HealthChecker):
    def check_health(self, url: str, timeout: int = 10) -> HealthCheckResult:
        # Custom health check implementation
        pass

    def wait_for_healthy(self, url: str, max_wait: int = 300, interval: int = 5) -> bool:
        # Custom wait implementation
        pass

class CustomResourceMonitor(ResourceMonitor):
    def get_system_metrics(self) -> ResourceMetrics:
        # Custom system metrics implementation
        pass

    def get_process_metrics(self, process_name: str) -> ResourceMetrics:
        # Custom process metrics implementation
        pass

# Use custom implementations
monitor = DeploymentHealthMonitor(
    health_checker=CustomHealthChecker(),
    resource_monitor=CustomResourceMonitor(),
)
```

## Configuration

### Alert Thresholds

Configure alert thresholds for different metrics:

```python
monitor = DeploymentHealthMonitor()

# Customize alert thresholds
monitor.alert_thresholds = {
    "response_time_ms": 500.0,      # Alert if response time > 500ms
    "cpu_usage_percent": 90.0,      # Alert if CPU > 90%
    "memory_usage_mb": 2048.0,      # Alert if memory > 2GB
    "error_rate_percent": 1.0,      # Alert if error rate > 1%
}
```

### Monitoring Intervals

Configure check intervals for different deployments:

```python
# Frequent checks for critical services
orchestrator.add_deployment_monitoring(
    deployment_id="critical-service",
    health_check_url="http://localhost:8080/health",
    check_interval=10,  # Check every 10 seconds
)

# Less frequent checks for non-critical services
orchestrator.add_deployment_monitoring(
    deployment_id="background-service",
    health_check_url="http://localhost:8081/health",
    check_interval=60,  # Check every minute
)
```

## Integration with Deployment Orchestration

### Automatic Health Monitoring

The deployment orchestrator automatically starts health monitoring for successful deployments:

```python
# Deploy with automatic health monitoring
result = orchestrator.deploy_with_strategy(
    config=deployment_config,
    strategy=DeploymentStrategy.BLUE_GREEN,
    deployment_func=deploy_function,
)

# Health monitoring is automatically started if health_check_url is available
if result.success and result.health_check_url:
    # Monitoring is already started
    status = orchestrator.get_deployment_health_status(result.deployment_id)
    print(f"Deployment health: {status['health_status']}")
```

### Manual Health Monitoring

You can also manually control health monitoring:

```python
# Start monitoring for all deployments
orchestrator.start_health_monitoring()

# Get health status of specific deployment
status = orchestrator.get_deployment_health_status("deployment-123")

# Get health status of all deployments
all_statuses = orchestrator.get_all_health_statuses()

# Stop monitoring
orchestrator.stop_health_monitoring()
```

## Alert System

### Built-in Alerts

The system automatically generates alerts for:

- **Health Check Failures**: When health checks fail
- **High Response Time**: When response time exceeds threshold
- **High CPU Usage**: When CPU usage exceeds threshold
- **High Memory Usage**: When memory usage exceeds threshold

### Custom Alert Handlers

You can add custom alert handlers:

```python
from crackseg.utils.deployment.orchestration import AlertHandler


class CustomAlertHandler(AlertHandler):
    def send_alert(self, alert_type: str, metadata: DeploymentMetadata, **kwargs):
        # Custom alert implementation
        print(f"Alert: {alert_type} for {metadata.deployment_id}")

# Add custom handler
orchestrator.add_alert_handler(CustomAlertHandler())
```

## Data Export

### Export Monitoring Data

Export monitoring data for analysis:

```python
from pathlib import Path

# Export monitoring data
export_path = Path("outputs/monitoring_data.json")
monitor.export_monitoring_data(export_path)

# Data includes:
# - Health check history
# - Resource metrics history
# - Alert history
# - Deployment configuration
```

### Monitoring Data Format

The exported data includes:

```json
{
  "timestamp": 1640995200.0,
  "monitored_deployments": {
    "deployment-123": {
      "health_check_url": "http://localhost:8080/health",
      "process_name": "my-service",
      "check_interval": 30,
      "health_history": [
        {
          "success": true,
          "status": "healthy",
          "response_time_ms": 150.5,
          "timestamp": 1640995200.0
        }
      ],
      "resource_history": [
        {
          "cpu_usage_percent": 25.5,
          "memory_usage_mb": 1024.0,
          "disk_usage_percent": 45.2,
          "network_io_mbps": 12.3,
          "timestamp": 1640995200.0
        }
      ]
    }
  }
}
```

## Best Practices

### Health Check Endpoints

Design health check endpoints that:

- **Respond quickly**: Keep response time under 100ms
- **Check dependencies**: Verify database, cache, external services
- **Return meaningful status**: Include service state and version
- **Handle errors gracefully**: Don't crash on dependency failures

Example health check endpoint:

```python
@app.route("/health")
def health_check():
    try:
        # Check database
        db_status = check_database()

        # Check cache
        cache_status = check_cache()

        # Check external services
        external_status = check_external_services()

        if all([db_status, cache_status, external_status]):
            return {"status": "healthy", "version": "1.0.0"}, 200
        else:
            return {"status": "degraded", "issues": ["cache_down"]}, 503

    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}, 500
```

### Monitoring Configuration

Configure monitoring based on service characteristics:

```python
# Critical services: frequent checks, low thresholds
orchestrator.add_deployment_monitoring(
    deployment_id="payment-service",
    health_check_url="http://payment-service/health",
    check_interval=10,
)

# Background services: less frequent checks, higher thresholds
orchestrator.add_deployment_monitoring(
    deployment_id="analytics-service",
    health_check_url="http://analytics-service/health",
    check_interval=60,
)
```

### Resource Monitoring

Monitor resources appropriate to your deployment:

```python
# For CPU-intensive services
monitor.alert_thresholds["cpu_usage_percent"] = 80.0

# For memory-intensive services
monitor.alert_thresholds["memory_usage_mb"] = 4096.0

# For network-intensive services
monitor.alert_thresholds["response_time_ms"] = 200.0
```

## Troubleshooting

### Common Issues

#### Health Checks Failing

```python
# Check if endpoint is accessible
import requests
response = requests.get("http://localhost:8080/health", timeout=5)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
```

#### High Resource Usage

```python
# Get detailed resource metrics
from crackseg.utils.deployment.health_monitoring import DefaultResourceMonitor


monitor = DefaultResourceMonitor()
metrics = monitor.get_system_metrics()

print(f"CPU: {metrics.cpu_usage_percent}%")
print(f"Memory: {metrics.memory_usage_mb}MB")
print(f"Disk: {metrics.disk_usage_percent}%")
```

#### Monitoring Not Starting

```python
# Check if monitoring is active
print(f"Monitoring active: {monitor.monitoring_active}")
print(f"Monitored deployments: {len(monitor.monitored_deployments)}")

# Check for errors in logs
import logging
logging.getLogger("crackseg.utils.deployment.health_monitoring").setLevel(logging.DEBUG)
```

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Monitor with debug output
monitor = DeploymentHealthMonitor()
monitor.add_deployment_monitoring(
    deployment_id="debug-service",
    health_check_url="http://localhost:8080/health",
    check_interval=5,
)
monitor.start_monitoring()
```

## Examples

### Complete Monitoring Setup

```python
from crackseg.utils.deployment.orchestration import (
    DeploymentOrchestrator, LoggingAlertHandler
)
from crackseg.utils.deployment.health_monitoring import DeploymentHealthMonitor


# Initialize orchestrator
orchestrator = DeploymentOrchestrator()

# Add alert handler
orchestrator.add_alert_handler(LoggingAlertHandler())

# Configure monitoring for different services
services = [
    {
        "id": "web-frontend",
        "url": "http://web-frontend:8080/health",
        "process": "web-frontend",
        "interval": 15,
    },
    {
        "id": "api-backend",
        "url": "http://api-backend:8081/health",
        "process": "api-backend",
        "interval": 30,
    },
    {
        "id": "model-service",
        "url": "http://model-service:8082/health",
        "process": "model-service",
        "interval": 60,
    },
]

# Add all services to monitoring
for service in services:
    orchestrator.add_deployment_monitoring(
        deployment_id=service["id"],
        health_check_url=service["url"],
        process_name=service["process"],
        check_interval=service["interval"],
    )

# Start monitoring
orchestrator.start_health_monitoring()

# Monitor for a period
import time
time.sleep(300)  # Monitor for 5 minutes

# Get status report
all_statuses = orchestrator.get_all_health_statuses()
for deployment_id, status in all_statuses.items():
    print(f"{deployment_id}: {status['health_status']}")

# Stop monitoring
orchestrator.stop_health_monitoring()
```

This comprehensive health monitoring system provides robust monitoring capabilities for deployed
artifacts with configurable alerts, resource tracking, and seamless integration with the deployment
orchestration system.
