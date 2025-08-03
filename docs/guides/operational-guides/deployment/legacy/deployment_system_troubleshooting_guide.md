# Deployment System Troubleshooting Guide

**Subtask 5.10**: Document System and Provide User Guides

## Overview

This troubleshooting guide provides solutions for common issues encountered when using the CrackSeg
Deployment System. It covers diagnostic procedures, error resolution, and preventive measures.

## Table of Contents

1. [Diagnostic Tools](#diagnostic-tools)
2. [Common Issues](#common-issues)
3. [Error Resolution](#error-resolution)
4. [Performance Issues](#performance-issues)
5. [Health Monitoring Issues](#health-monitoring-issues)
6. [Security Issues](#security-issues)
7. [Network Issues](#network-issues)
8. [Preventive Measures](#preventive-measures)

## Diagnostic Tools

### System Diagnostics

```python
from crackseg.utils.deployment.orchestration import DeploymentOrchestrator


# Run comprehensive system diagnostics
orchestrator = DeploymentOrchestrator()
diagnostics = orchestrator.run_diagnostics()

print("System Diagnostics:")
for component, status in diagnostics.items():
    print(f"  {component}: {'✅' if status['healthy'] else ''}")
    if not status['healthy']:
        print(f"    Issues: {status['issues']}")
```

### Health Check Diagnostics

```python
from crackseg.utils.deployment.health_monitoring import DeploymentHealthMonitor


# Check deployment health
monitor = DeploymentHealthMonitor()
health_status = monitor.check_deployment_health("deployment-123")

print(f"Health Status: {health_status['status']}")
print(f"Response Time: {health_status['response_time_ms']}ms")
print(f"Error Rate: {health_status['error_rate']}")
```

### Resource Monitoring

```python
# Get resource metrics
metrics = monitor.get_resource_metrics("deployment-123")

print("Resource Metrics:")
print(f"  CPU Usage: {metrics['cpu_percent']}%")
print(f"  Memory Usage: {metrics['memory_percent']}%")
print(f"  Disk Usage: {metrics['disk_percent']}%")
print(f"  Network I/O: {metrics['network_io']} MB/s")
```

### Log Analysis

```python
# Get deployment logs
logs = orchestrator.get_deployment_logs("deployment-123")

print("Recent Logs:")
for log_entry in logs[-10:]:  # Last 10 log entries
    print(f"  [{log_entry['timestamp']}] {log_entry['level']}: {log_entry['message']}")
```

## Common Issues

### 1. Deployment Validation Failures

**Symptoms:**

- Deployment fails during validation phase
- Performance thresholds not met
- Resource requirements exceeded

**Diagnosis:**

```python
# Check validation details
validation = orchestrator.validate_deployment(artifact)
print(f"Validation Ready: {validation.ready}")
print(f"Validation Issues: {validation.issues}")
print(f"Performance Checks: {validation.performance_checks}")
print(f"Resource Checks: {validation.resource_checks}")
```

**Solutions:**

1. **Adjust Performance Thresholds:**

   ```python
   # Increase response time threshold
   config.performance_thresholds = {
       "response_time_ms": 1000.0,  # Increase from 500ms
       "memory_usage_mb": 4096.0    # Increase from 2048MB
   }
   ```

2. **Optimize Artifact:**

   ```python
   # Optimize for resource constraints
   optimized_artifact = orchestrator.optimize_artifact(
       artifact,
       target_memory_mb=2048,
       target_cpu_cores=2,
       enable_quantization=True
   )
   ```

3. **Scale Resources:**

   ```python
   # Increase resource limits
   config.resource_limits = {
       "memory_mb": 8192,    # Increase memory
       "cpu_cores": 8,       # Increase CPU cores
       "disk_gb": 200        # Increase disk space
   }
   ```

### 2. Health Check Failures

**Symptoms:**

- Health checks return unhealthy status
- High response times
- Increased error rates

**Diagnosis:**

```python
# Check health check configuration
health_config = monitor.get_health_config("deployment-123")
print(f"Health Check URL: {health_config['url']}")
print(f"Timeout: {health_config['timeout']}s")
print(f"Retries: {health_config['retries']}")
```

**Solutions:**

1. **Increase Timeout:**

   ```python
   # Increase health check timeout
   monitor.update_health_config(
       deployment_id="deployment-123",
       timeout=120,  # Increase from 60s
       retries=5     # Increase from 3
   )
   ```

2. **Check Application Health:**

   ```python
   # Verify application is responding
   import requests

   try:
       response = requests.get("http://deployment-url/health", timeout=30)
       print(f"Health Check Response: {response.status_code}")
       print(f"Response Time: {response.elapsed.total_seconds()}s")
   except Exception as e:
       print(f"Health Check Failed: {e}")
   ```

3. **Restart Application:**

   ```python
   # Restart deployment
   orchestrator.restart_deployment("deployment-123")
   ```

### 3. Resource Exhaustion

**Symptoms:**

- High CPU/memory usage
- Out of memory errors
- Slow response times

**Diagnosis:**

```python
# Check resource usage
metrics = monitor.get_resource_metrics("deployment-123")

if metrics['cpu_percent'] > 80:
    print("⚠️ High CPU usage detected")
if metrics['memory_percent'] > 80:
    print("⚠️ High memory usage detected")
if metrics['disk_percent'] > 90:
    print("⚠️ High disk usage detected")
```

**Solutions:**

1. **Scale Resources:**

   ```python
   # Scale up deployment
   orchestrator.scale_deployment(
       deployment_id="deployment-123",
       cpu_cores=8,      # Increase CPU
       memory_mb=8192    # Increase memory
   )
   ```

2. **Optimize Application:**

   ```python
   # Optimize for resource efficiency
   optimized_artifact = orchestrator.optimize_artifact(
       artifact,
       enable_quantization=True,
       enable_pruning=True,
       target_memory_mb=2048
   )
   ```

3. **Add Resource Limits:**

   ```python
   # Set resource limits
   config.resource_limits = {
       "memory_mb": 4096,
       "cpu_cores": 4,
       "disk_gb": 100
   }
   ```

### 4. Rollback Failures

**Symptoms:**

- Rollback procedure fails
- Previous deployment not available
- Rollback timeout

**Diagnosis:**

```python
# Check rollback status
rollback_status = orchestrator.get_rollback_status("deployment-123")
print(f"Rollback Status: {rollback_status['status']}")
print(f"Previous Deployment: {rollback_status['previous_deployment']}")
print(f"Rollback Issues: {rollback_status['issues']}")
```

**Solutions:**

1. **Force Rollback:**

   ```python
   # Force rollback if needed
   success = orchestrator.force_rollback("deployment-123", force=True)
   print(f"Force Rollback: {'✅' if success else ''}")
   ```

2. **Check Previous Deployment:**

   ```python
   # Verify previous deployment exists
   previous = orchestrator.get_previous_deployment("deployment-123")
   if previous:
       print(f"Previous Deployment: {previous['deployment_id']}")
       print(f"Previous Status: {previous['status']}")
   ```

3. **Manual Rollback:**

   ```python
   # Manual rollback to specific deployment
   orchestrator.rollback_to_deployment("deployment-123", "previous-deployment-id")
   ```

## Error Resolution

### Deployment Errors

**Common Error**: "Artifact not found"

```python
# Check artifact availability
artifact = orchestrator.get_artifact("crackseg-model-v1")
if not artifact:
    print(" Artifact not found")
    # Rebuild or restore artifact
    artifact = orchestrator.build_artifact("crackseg-model-v1")
```

**Common Error**: "Insufficient resources"

```python
# Check available resources
resources = orchestrator.get_available_resources()
print(f"Available CPU: {resources['cpu_cores']}")
print(f"Available Memory: {resources['memory_mb']}MB")

# Scale down requirements or scale up infrastructure
config.resource_limits = {
    "memory_mb": min(2048, resources['memory_mb']),
    "cpu_cores": min(2, resources['cpu_cores'])
}
```

**Common Error**: "Health check timeout"

```python
# Increase health check timeout
monitor.update_health_config(
    deployment_id="deployment-123",
    timeout=180,  # 3 minutes
    retries=5
)

# Check if application is starting slowly
startup_logs = orchestrator.get_startup_logs("deployment-123")
for log in startup_logs:
    print(f"Startup: {log['message']}")
```

### Validation Errors

**Performance Validation Failures:**

```python
# Check performance metrics
performance = orchestrator.get_performance_metrics("deployment-123")
print(f"Response Time: {performance['response_time_ms']}ms")
print(f"Throughput: {performance['throughput_rps']} RPS")

# Adjust thresholds or optimize
if performance['response_time_ms'] > 500:
    config.performance_thresholds["response_time_ms"] = 1000.0
```

**Resource Validation Failures:**

```python
# Check resource usage
usage = orchestrator.get_resource_usage("deployment-123")
print(f"Memory Usage: {usage['memory_mb']}MB")
print(f"CPU Usage: {usage['cpu_percent']}%")

# Optimize or scale
if usage['memory_mb'] > 2048:
    optimized = orchestrator.optimize_artifact(artifact, target_memory_mb=1024)
```

## Performance Issues

### High Response Times

**Diagnosis:**

```python
# Monitor response times
response_times = monitor.get_response_time_history("deployment-123")
print(f"Average Response Time: {sum(response_times) / len(response_times)}ms")
print(f"95th Percentile: {sorted(response_times)[int(len(response_times) * 0.95)]}ms")
```

**Solutions:**

1. **Optimize Model:**

   ```python
   # Enable model optimization
   optimized = orchestrator.optimize_artifact(
       artifact,
       enable_quantization=True,
       enable_pruning=True,
       target_latency_ms=200
   )
   ```

2. **Scale Resources:**

   ```python
   # Increase CPU cores
   orchestrator.scale_deployment(
       deployment_id="deployment-123",
       cpu_cores=8
   )
   ```

3. **Add Caching:**

   ```python
   # Enable caching
   config.cache_enabled = True
   config.cache_size_mb = 1024
   ```

### Memory Leaks

**Diagnosis:**

```python
# Monitor memory usage over time
memory_history = monitor.get_memory_usage_history("deployment-123")
print(f"Memory Trend: {memory_history}")

# Check for memory leaks
if memory_history[-1] > memory_history[0] * 1.5:
    print("⚠️ Potential memory leak detected")
```

**Solutions:**

1. **Restart Deployment:**

   ```python
   # Restart to clear memory
   orchestrator.restart_deployment("deployment-123")
   ```

2. **Increase Memory:**

   ```python
   # Increase memory allocation
   orchestrator.scale_deployment(
       deployment_id="deployment-123",
       memory_mb=8192
   )
   ```

3. **Optimize Code:**

   ```python
   # Enable garbage collection optimization
   config.gc_optimization = True
   config.memory_pool_size = 512
   ```

## Health Monitoring Issues

### False Positives

**Symptoms:**

- Health checks fail intermittently
- Application is actually healthy
- Network issues causing failures

**Diagnosis:**

```python
# Check health check reliability
reliability = monitor.get_health_check_reliability("deployment-123")
print(f"Health Check Reliability: {reliability['success_rate']}%")
print(f"False Positives: {reliability['false_positives']}")
```

**Solutions:**

1. **Adjust Health Check Parameters:**

   ```python
   # Increase retries and timeout
   monitor.update_health_config(
       deployment_id="deployment-123",
       timeout=120,
       retries=5,
       success_threshold=2  # Require 2 successful checks
   )
   ```

2. **Custom Health Check:**

   ```python
   # Implement custom health check
   class ReliableHealthChecker(HealthChecker):
       def check_health(self, deployment_id: str) -> dict[str, Any]:
           # Implement more reliable health check
           return {"status": "healthy", "confidence": 0.95}
   ```

### Alert Fatigue

**Symptoms:**

- Too many alerts
- Alerts for minor issues
- Important alerts missed

**Solutions:**

1. **Adjust Alert Thresholds:**

   ```python
   # Increase alert thresholds
   config.alert_thresholds = {
       "critical": {"cpu_usage": 0.95, "memory_usage": 0.98},
       "warning": {"cpu_usage": 0.8, "memory_usage": 0.85}
   }
   ```

2. **Implement Alert Cooldown:**

   ```python
   # Add alert cooldown
   config.alert_cooldown_minutes = 30
   config.alert_grouping = True
   ```

## Security Issues

### Authentication Failures

**Symptoms:**

- Deployment fails due to authentication
- Access denied errors
- Token expiration

**Diagnosis:**

```python
# Check authentication status
auth_status = orchestrator.check_authentication()
print(f"Authentication Status: {auth_status['valid']}")
print(f"Token Expiry: {auth_status['expires_at']}")
```

**Solutions:**

1. **Refresh Authentication:**

   ```python
   # Refresh authentication token
   orchestrator.refresh_authentication()
   ```

2. **Update Credentials:**

   ```python
   # Update deployment credentials
   orchestrator.update_credentials(
       username="new_username",
       password="new_password"
   )
   ```

### SSL/TLS Issues

**Symptoms:**

- SSL certificate errors
- TLS handshake failures
- Security warnings

**Diagnosis:**

```python
# Check SSL/TLS configuration
ssl_status = orchestrator.check_ssl_configuration()
print(f"SSL Valid: {ssl_status['valid']}")
print(f"Certificate Expiry: {ssl_status['expires_at']}")
```

**Solutions:**

1. **Update Certificates:**

   ```python
   # Update SSL certificates
   orchestrator.update_ssl_certificates(
       certificate_path="/path/to/new/cert.pem",
       private_key_path="/path/to/new/key.pem"
   )
   ```

2. **Disable SSL Verification (Development):**

   ```python
   # Disable SSL verification for development
   config.ssl_verify = False
   ```

## Network Issues

### Connectivity Problems

**Symptoms:**

- Network timeouts
- Connection refused errors
- Slow network performance

**Diagnosis:**

```python
# Check network connectivity
network_status = orchestrator.check_network_connectivity()
print(f"Network Status: {network_status['status']}")
print(f"Latency: {network_status['latency_ms']}ms")
print(f"Bandwidth: {network_status['bandwidth_mbps']}Mbps")
```

**Solutions:**

1. **Increase Timeouts:**

   ```python
   # Increase network timeouts
   config.network_timeout = 300  # 5 minutes
   config.connection_timeout = 60  # 1 minute
   ```

2. **Retry Logic:**

   ```python
   # Enable retry logic
   config.max_retries = 5
   config.retry_delay_seconds = 10
   ```

### DNS Resolution Issues

**Symptoms:**

- Hostname resolution failures
- Service discovery problems

**Solutions:**

1. **Use IP Addresses:**

   ```python
   # Use IP addresses instead of hostnames
   config.deployment_url = "http://192.168.1.100:8080"
   ```

2. **Update DNS Configuration:**

   ```python
   # Update DNS settings
   config.dns_servers = ["8.8.8.8", "8.8.4.4"]
   ```

## Preventive Measures

### Regular Maintenance

1. **Scheduled Health Checks:**

   ```python
   # Schedule regular health checks
   monitor.schedule_health_checks(
       deployment_id="deployment-123",
       interval_minutes=30
   )
   ```

2. **Resource Monitoring:**

   ```python
   # Set up resource monitoring
   monitor.setup_resource_monitoring(
       deployment_id="deployment-123",
       alert_threshold=0.8
   )
   ```

3. **Backup Procedures:**

   ```python
   # Create deployment backups
   orchestrator.create_deployment_backup("deployment-123")
   ```

### Performance Optimization

1. **Regular Optimization:**

   ```python
   # Schedule regular optimization
   orchestrator.schedule_optimization(
       artifact_id="crackseg-model-v1",
       interval_days=7
   )
   ```

2. **Resource Scaling:**

   ```python
   # Enable auto-scaling
   config.auto_scaling = True
   config.scale_up_threshold = 0.7
   config.scale_down_threshold = 0.3
   ```

### Security Hardening

1. **Regular Security Audits:**

   ```python
   # Schedule security audits
   orchestrator.schedule_security_audit(
       deployment_id="deployment-123",
       interval_days=30
   )
   ```

2. **Access Control:**

   ```python
   # Implement strict access control
   config.access_control = {
       "required_roles": ["admin", "deployer"],
       "ip_whitelist": ["192.168.1.0/24"],
       "session_timeout_minutes": 30
   }
   ```

---

**Status**: ✅ **COMPLETED**

This troubleshooting guide provides comprehensive solutions for common deployment system issues,
enabling users to quickly diagnose and resolve problems.
