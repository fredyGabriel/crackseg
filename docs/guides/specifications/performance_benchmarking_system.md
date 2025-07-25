# Performance Benchmarking System Guide

## Overview

The CrackSeg performance benchmarking system provides comprehensive performance testing, monitoring,
and validation for the entire E2E testing pipeline. This system ensures reliability, performance
standards, and resource efficiency across all components.

## System Architecture

### Core Components

1. **Benchmark Suite** (`tests/e2e/performance/benchmark_suite.py`)
   - Orchestrates all performance tests
   - Manages test configurations and execution
   - Validates results against thresholds

2. **Resource Monitor** (`src/utils/monitoring/resource_monitor.py`)
   - Real-time CPU, memory, and GPU tracking
   - Docker container monitoring
   - Resource leak detection

3. **Cleanup Validation** (`tests/e2e/cleanup/`)
   - Automated resource cleanup
   - Post-test validation
   - Environment state restoration

4. **CI/CD Integration** (`.github/workflows/performance-ci.yml`)
   - Automated performance gates
   - Regression detection
   - Performance report generation

5. **Maintenance Automation** (`scripts/performance_maintenance.py`)
   - System health checks
   - Baseline updates
   - Automated maintenance protocols

## Quick Start

### Running Basic Performance Tests

```bash
# Validate performance system
python scripts/check_test_files.py --performance-check

# Run smoke tests (quick validation)
python -m pytest tests/e2e/performance/ -k "smoke" -v

# Run complete benchmark suite
python -m pytest tests/e2e/performance/test_benchmark_runner.py -v

# Run system health check
python scripts/performance_maintenance.py --health-check
```

### Manual Benchmark Execution

```python
from tests.e2e.performance.benchmark_suite import BenchmarkSuite
from tests.e2e.config.performance_thresholds import PerformanceThresholds

# Initialize benchmark suite
thresholds = PerformanceThresholds.from_file("configs/testing/performance_thresholds.yaml")
suite = BenchmarkSuite(thresholds)

# Run specific benchmark
config = BenchmarkConfig(
    duration_seconds=30,
    concurrent_users=10,
    ramp_up_seconds=5,
    iterations=100
)

result = await suite.run_benchmark("load", config)
print(f"Success rate: {result.success_rate:.1f}%")
```

## Performance Benchmarks

### Available Benchmark Types

1. **Load Testing** (`load_test.py`)
   - Normal operational load simulation
   - Response time validation
   - Throughput measurement

2. **Stress Testing** (`stress_test.py`)
   - Beyond-capacity testing
   - System breaking point identification
   - Resource exhaustion scenarios

3. **Endurance Testing** (`endurance_test.py`)
   - Extended duration monitoring
   - Memory leak detection
   - Performance degradation tracking

### Benchmark Configuration

```yaml
# configs/testing/performance_thresholds.yaml
web_interface:
  page_load_time_ms: 2000
  response_time_95th_percentile_ms: 500
  memory_usage_mb: 512

model_processing:
  inference_time_ms: 100
  batch_processing_time_ms: 1000
  gpu_memory_usage_mb: 4096

system_resources:
  cpu_usage_percent: 80
  memory_usage_percent: 85
  disk_io_mb_per_second: 100
```

## Resource Monitoring

### Automatic Monitoring

The system automatically monitors resources during test execution:

- **CPU Usage**: Per-core utilization tracking
- **Memory Usage**: RAM and swap monitoring
- **GPU Resources**: VRAM usage (RTX 3070 Ti optimized)
- **Network I/O**: Bandwidth and connection monitoring
- **Disk I/O**: Read/write performance tracking

### Manual Resource Monitoring

```python
from crackseg.utils.monitoring.resource_monitor import ResourceMonitor

# Start monitoring
monitor = ResourceMonitor(interval=1.0)
monitor.start()

# Your test code here
# ...

# Stop and get results
results = monitor.stop()
print(f"Peak memory usage: {results.peak_memory_mb} MB")
print(f"Average CPU usage: {results.avg_cpu_percent:.1f}%")
```

## Cleanup Validation

### Automated Cleanup

The system automatically validates cleanup after test execution:

1. **Process Cleanup**: Ensures no test processes remain running
2. **Container Cleanup**: Validates Docker containers are removed
3. **File Cleanup**: Confirms temporary files are deleted
4. **Network Cleanup**: Verifies ports and connections are released

### Manual Cleanup Validation

```bash
# Run cleanup validation
python -m pytest tests/e2e/cleanup/test_validation_system.py -v

# Validate specific resource type
python -m pytest tests/e2e/cleanup/ -k "container" -v
```

## CI/CD Integration

### GitHub Actions Workflow

The performance CI/CD pipeline automatically:

1. **Validates Environment**: Checks all components are available
2. **Runs Benchmarks**: Executes performance test suite
3. **Validates Thresholds**: Ensures performance standards are met
4. **Generates Reports**: Creates detailed performance reports
5. **Triggers Alerts**: Notifies on performance regressions

### Manual CI/CD Testing

```bash
# Simulate CI/CD environment
export PERFORMANCE_CI_MODE=true
export PERFORMANCE_GATES_ENABLED=true

# Run performance validation
python -m pytest tests/e2e/performance/ --performance-ci -v
```

## Performance Thresholds

### Threshold Categories

1. **Web Interface Thresholds**
   - Page load times < 2000ms
   - Response times < 500ms
   - Memory usage < 512MB

2. **Model Processing Thresholds**
   - Inference time < 100ms
   - Batch processing < 1000ms
   - GPU memory < 4096MB

3. **System Resource Thresholds**
   - CPU usage < 80%
   - Memory usage < 85%
   - Disk I/O < 100MB/s

### Threshold Validation

```python
from tests.e2e.config.performance_thresholds import PerformanceThresholds

# Load thresholds
thresholds = PerformanceThresholds.from_file("configs/testing/performance_thresholds.yaml")

# Validate specific metric
violations = thresholds.validate_web_interface({
    "page_load_time_ms": 1500,
    "response_time_95th_percentile_ms": 300,
    "memory_usage_mb": 256
})

if violations:
    print(f"Threshold violations: {violations}")
```

## Maintenance Protocols

### Automated Maintenance System

The performance benchmarking system includes comprehensive maintenance automation:

#### 1. System Health Checks

```bash
# Run comprehensive health check
python scripts/performance_maintenance.py --health-check

# Check specific components
python scripts/check_test_files.py --performance-check
```

**Health Check Coverage:**

- Performance system structure validation
- Configuration file integrity
- CI/CD pipeline integration
- Resource monitoring functionality
- Cleanup validation system

#### 2. Performance Baseline Updates

```bash
# Update performance baselines
python scripts/performance_maintenance.py --update-baselines

# Force baseline regeneration
python scripts/performance_maintenance.py --update-baselines --force
```

**Baseline Update Process:**

- Analyzes recent performance data
- Identifies performance improvements
- Updates threshold configurations
- Validates new baseline performance

#### 3. Cleanup System Validation

```bash
# Validate cleanup system
python scripts/performance_maintenance.py --validate-cleanup

# Test cleanup with resource leak simulation
python scripts/performance_maintenance.py --validate-cleanup --simulate-leaks
```

**Cleanup Validation Coverage:**

- Docker container cleanup
- Process termination validation
- File system cleanup
- Network resource release
- Memory leak detection

#### 4. Full Maintenance Cycle

```bash
# Run complete maintenance cycle
python scripts/performance_maintenance.py --full-maintenance

# Generate detailed maintenance report
python scripts/performance_maintenance.py --full-maintenance --generate-report
```

### Manual Maintenance Procedures

#### Weekly Maintenance Tasks

1. **Performance Trend Analysis**

   ```bash
   # Review performance trends
   python scripts/performance_maintenance.py --health-check

   # Check for performance regressions
   python -m pytest tests/e2e/performance/ --regression-check
   ```

2. **Resource Usage Review**

   ```bash
   # Analyze resource usage patterns
   python -c "
   from crackseg.utils.monitoring.resource_monitor import ResourceMonitor
   monitor = ResourceMonitor()
   report = monitor.generate_usage_report()
   print(report)
   "
   ```

3. **Cleanup System Verification**

   ```bash
   # Verify cleanup system integrity
   python scripts/performance_maintenance.py --validate-cleanup
   ```

#### Monthly Maintenance Tasks

1. **Baseline Updates**

   ```bash
   # Update performance baselines
   python scripts/performance_maintenance.py --update-baselines

   # Review threshold configurations
   vim configs/testing/performance_thresholds.yaml
   ```

2. **System Component Validation**

   ```bash
   # Validate all system components
   python scripts/check_test_files.py --performance-check

   # Test CI/CD integration
   gh workflow run performance-ci.yml
   ```

3. **Documentation Updates**

   ```bash
   # Update performance documentation
   python scripts/performance_maintenance.py --generate-maintenance-report
   ```

#### Quarterly Maintenance Tasks

1. **Comprehensive System Review**

   ```bash
   # Full system audit
   python scripts/performance_maintenance.py --full-maintenance

   # Performance regression analysis
   python -m pytest tests/e2e/performance/ --comprehensive-analysis
   ```

2. **Infrastructure Optimization**

   ```bash
   # Optimize monitoring infrastructure
   python scripts/performance_maintenance.py --optimize-monitoring

   # Update automation scripts
   python scripts/performance_maintenance.py --update-automation
   ```

### Maintenance Scheduling

#### Automated Scheduling (CI/CD)

The system automatically performs maintenance tasks:

- **Daily**: Basic health checks during CI/CD runs
- **Weekly**: Comprehensive validation on main branch pushes
- **Monthly**: Baseline updates and system optimization

#### Manual Scheduling

Set up cron jobs for regular maintenance:

```bash
# Add to crontab
# Weekly health check (Sundays at 2 AM)
0 2 * * 0 cd /path/to/crackseg && python scripts/performance_maintenance.py --health-check

# Monthly baseline update (1st of month at 3 AM)
0 3 1 * * cd /path/to/crackseg && python scripts/performance_maintenance.py --update-baselines

# Quarterly full maintenance (1st of quarter at 1 AM)
0 1 1 1,4,7,10 * cd /path/to/crackseg && python scripts/performance_maintenance.py --full-maintenance
```

## Troubleshooting

### Common Issues

#### 1. Performance Test Failures

**Symptoms**: Tests fail with timeout or threshold violations

**Diagnostic Steps**:

```bash
# Check system health
python scripts/performance_maintenance.py --health-check

# Validate system components
python scripts/check_test_files.py --performance-check

# Run maintenance cycle
python scripts/performance_maintenance.py --full-maintenance
```

**Solutions**:

```bash
# Check system resources
python scripts/check_test_files.py --performance-check

# Reduce test load
# Edit configs/testing/performance_thresholds.yaml
# Increase threshold values temporarily

# Run individual components
python -m pytest tests/e2e/performance/test_load_test.py -v
```

#### 2. Resource Monitoring Issues

**Symptoms**: ResourceMonitor fails to start or reports incorrect values

**Diagnostic Steps**:

```bash
# Test resource monitoring
python scripts/performance_maintenance.py --validate-cleanup

# Check monitoring dependencies
python -c "
from crackseg.utils.monitoring.resource_monitor import ResourceMonitor
monitor = ResourceMonitor()
print('Resource monitoring OK')
"
```

**Solutions**:

```bash
# Check system permissions
# Linux/macOS: May need sudo for some metrics
# Windows: Run as administrator

# Verify monitoring dependencies
pip install psutil GPUtil

# Test monitoring manually
python -c "from crackseg.utils.monitoring.resource_monitor import ResourceMonitor; print('OK')"
```

#### 3. Cleanup Validation Failures

**Symptoms**: Cleanup validation reports resource leaks

**Diagnostic Steps**:

```bash
# Run detailed cleanup validation
python scripts/performance_maintenance.py --validate-cleanup

# Check for stuck processes
python -c "
import psutil
for proc in psutil.process_iter():
    if 'crackseg' in proc.name():
        print(f'Process: {proc.name()} PID: {proc.pid}')
"
```

**Solutions**:

```bash
# Run cleanup validation with verbose output
python -m pytest tests/e2e/cleanup/ -v -s

# Check for stuck processes
ps aux | grep crackseg

# Clean up manually
docker container prune -f
docker volume prune -f
```

#### 4. CI/CD Pipeline Issues

**Symptoms**: Performance CI fails in GitHub Actions

**Diagnostic Steps**:

```bash
# Test CI/CD integration locally
export PERFORMANCE_CI_MODE=true
python scripts/performance_maintenance.py --health-check

# Validate workflow configuration
gh workflow list
gh workflow run performance-ci.yml
```

**Solutions**:

- Check workflow permissions in GitHub
- Verify all required files are committed
- Review CI logs for specific error messages
- Test locally with CI environment variables

#### 5. Maintenance Script Issues

**Symptoms**: Maintenance scripts fail or produce incorrect results

**Diagnostic Steps**:

```bash
# Test maintenance script components
python scripts/performance_maintenance.py --health-check

# Check script dependencies
python -c "
import sys
print('Python version:', sys.version)
import subprocess
print('Subprocess module OK')
"
```

**Solutions**:

- Verify Python environment and dependencies
- Check file permissions for scripts
- Review log files for detailed error messages
- Update maintenance scripts if needed

### Debug Mode

Enable debug mode for detailed logging:

```bash
export PERFORMANCE_DEBUG=true
export LOGGING_LEVEL=DEBUG

python -m pytest tests/e2e/performance/ -v -s --log-level=DEBUG
python scripts/performance_maintenance.py --health-check --debug
```

## Integration with Existing Systems

### Task Master Integration

The performance benchmarking system integrates with Task Master for project management:

```bash
# Update performance-related tasks
task-master update-subtask --id=16.11 --prompt="Performance system integration completed"

# Set task status
task-master set-status --id=16.11 --status=done
```

### Git Hooks Integration

Add performance validation to pre-commit hooks:

```bash
# .git/hooks/pre-commit
#!/bin/bash
echo "Running performance system validation..."
python scripts/check_test_files.py --performance-check
if [ $? -ne 0 ]; then
    echo "Performance system validation failed"
    exit 1
fi

echo "Running maintenance health check..."
python scripts/performance_maintenance.py --health-check
if [ $? -ne 0 ]; then
    echo "Performance maintenance check failed"
    exit 1
fi
```

### IDE Integration

Configure IDE to run performance checks:

```json
// .vscode/tasks.json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Validate Performance System",
            "type": "shell",
            "command": "python",
            "args": ["scripts/check_test_files.py", "--performance-check"],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Performance Health Check",
            "type": "shell",
            "command": "python",
            "args": ["scripts/performance_maintenance.py", "--health-check"],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Full Performance Maintenance",
            "type": "shell",
            "command": "python",
            "args": ["scripts/performance_maintenance.py", "--full-maintenance"],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        }
    ]
}
```

### Automation Scripts Integration

The system integrates with existing automation scripts:

1. **Test Validation**: `scripts/check_test_files.py` includes performance checks
2. **Maintenance Automation**: `scripts/performance_maintenance.py` provides comprehensive maintenance
3. **CI/CD Integration**: `.github/workflows/performance-ci.yml` automates validation

## API Reference

### Key Classes

- `BenchmarkSuite`: Main orchestrator for performance tests
- `BenchmarkConfig`: Configuration for benchmark execution
- `BenchmarkResult`: Results container with metrics
- `ResourceMonitor`: Real-time resource monitoring
- `PerformanceThresholds`: Threshold validation and management
- `PerformanceMaintenanceManager`: Automated maintenance operations

### Environment Variables

- `PERFORMANCE_CI_MODE`: Enable CI/CD mode
- `PERFORMANCE_GATES_ENABLED`: Enable performance gates
- `PERFORMANCE_DEBUG`: Enable debug logging
- `RESOURCE_MONITORING_INTERVAL`: Monitoring interval in seconds

### Command Line Tools

#### Performance Validation

```bash
# Basic validation
python scripts/check_test_files.py --performance-check

# Comprehensive validation
python scripts/performance_maintenance.py --health-check
```

#### Maintenance Operations

```bash
# Update baselines
python scripts/performance_maintenance.py --update-baselines

# Validate cleanup
python scripts/performance_maintenance.py --validate-cleanup

# Full maintenance
python scripts/performance_maintenance.py --full-maintenance
```

#### Testing and Benchmarking

```bash
# Run smoke tests
python -m pytest tests/e2e/performance/ -k "smoke"

# Run full benchmark suite
python -m pytest tests/e2e/performance/test_benchmark_runner.py

# Run specific benchmark type
python -m pytest tests/e2e/performance/test_load_test.py
```

## Best Practices

1. **Regular Monitoring**: Run performance checks with every significant change
2. **Baseline Maintenance**: Update baselines after system improvements
3. **Resource Awareness**: Monitor resource usage during development
4. **Cleanup Discipline**: Always validate cleanup after testing
5. **CI/CD Integration**: Ensure performance gates are active in pipelines
6. **Automated Maintenance**: Use scheduled maintenance for system health
7. **Documentation Updates**: Keep performance documentation current
8. **Threshold Management**: Regularly review and adjust performance thresholds

## Support

For issues or questions:

1. Check troubleshooting section above
2. Review GitHub Issues for similar problems
3. Run system validation: `python scripts/check_test_files.py --performance-check`
4. Run maintenance check: `python scripts/performance_maintenance.py --health-check`
5. Enable debug mode for detailed diagnostics
6. Check CI/CD pipeline logs for automated validation results

---

*This documentation covers the complete performance benchmarking system. For specific*
*implementation details, refer to the source code and inline documentation.*
