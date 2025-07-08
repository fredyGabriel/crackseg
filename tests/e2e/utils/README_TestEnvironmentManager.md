# TestEnvironmentManager - Test Environment Setup System

**Subtask 16.1 Implementation**: Centralized test environment setup with consistent hardware
specifications, environment isolation, and baseline network conditions.

## Overview

The TestEnvironmentManager provides a unified interface for configuring and managing test
environments with:

- **Consistent Hardware Specifications**: RTX 3070 Ti optimized configurations with VRAM constraints
- **Environment Isolation**: Process-level isolation preventing test interference
- **Baseline Network Conditions**: Standardized network configurations for reliable testing
- **Resource Management**: Automated resource allocation and cleanup
- **Integration**: Seamless integration with existing E2E infrastructure

## Architecture

```txt
TestEnvironmentManager
├── HardwareSpecification      # CPU, memory, GPU constraints
├── NetworkConditions          # Latency, bandwidth, timeout settings
├── EnvironmentIsolation       # Process isolation, port allocation
├── SystemCapabilityChecker    # Hardware validation
└── Integration Components
    ├── ResourceManager        # Port and memory allocation
    ├── BrowserConfigManager   # Browser configuration
    └── EnvironmentManager     # Environment variables
```

## Key Features

### 1. Hardware Specification Management

```python
from tests.e2e.utils.test_environment_manager import HardwareSpecification

# RTX 3070 Ti optimized specification
rtx_3070_ti_spec = HardwareSpecification(
    cpu_cores=4,
    memory_mb=8192,
    gpu_memory_mb=8192,        # RTX 3070 Ti VRAM limit
    max_model_batch_size=16,   # Optimized for 8GB VRAM
    browser_instances=3,
    concurrent_tests=4,
)
```

### 2. Environment Isolation

```python
from tests.e2e.utils.test_environment_manager import EnvironmentIsolation

isolation = EnvironmentIsolation(
    process_isolation=True,     # Separate process spaces
    network_isolation=True,     # Isolated network conditions
    filesystem_isolation=True,  # Separate temp directories
    port_range=(8600, 8699),    # Allocated port range
    cleanup_on_exit=True,       # Automatic cleanup
    memory_limit_mb=4096,       # Resource limits
)
```

### 3. Network Conditions

```python
from tests.e2e.utils.test_environment_manager import NetworkConditions

# Baseline network conditions
baseline_network = NetworkConditions(
    latency_ms=50,
    bandwidth_mbps=100,
    packet_loss_percent=0.0,
    connection_timeout_sec=30,
    retry_attempts=3,
)
```

## Usage Examples

### Basic Usage with Context Manager

```python
from tests.e2e.utils.test_environment_manager import TestEnvironmentManager

# Create manager with default configuration
manager = TestEnvironmentManager()

# Setup isolated environment
with manager.setup_test_environment() as env_info:
    # Environment is fully configured and isolated
    print(f"Environment ID: {env_info['environment_id']}")
    print(f"Allocated ports: {env_info['resource_allocation'].allocated_ports}")

    # Run your tests here
    # All resources are automatically managed

# Environment is automatically cleaned up
```

### Using Pytest Fixtures

```python
import pytest

@pytest.mark.isolated_environment
def test_with_isolated_environment(isolated_test_environment):
    """Test using isolated environment fixture."""
    env_info = isolated_test_environment

    # Environment is pre-configured with consistent specifications
    assert env_info["validation_results"]["valid"] is True

    # Access environment details
    hardware = env_info["hardware_specs"]
    network = env_info["network_conditions"]

    # Run tests with guaranteed environment consistency

@pytest.mark.performance_environment
def test_with_performance_environment(performance_test_environment):
    """Test using performance-optimized environment."""
    env_info = performance_test_environment

    # High-performance configuration automatically applied
    hardware = env_info["hardware_specs"]
    assert hardware.cpu_cores >= 4
    assert hardware.memory_mb >= 8192
```

### Custom Configuration

```python
from tests.e2e.utils.test_environment_manager import (
    TestEnvironmentManager,
    TestEnvironmentConfig,
    HardwareSpecification,
    NetworkConditions,
)

# Create custom configuration
custom_config = TestEnvironmentConfig(
    hardware=HardwareSpecification(
        cpu_cores=8,
        memory_mb=16384,
        gpu_memory_mb=8192,  # RTX 3070 Ti constraint
        concurrent_tests=6,
    ),
    network=NetworkConditions(
        latency_ms=25,       # Low latency for performance tests
        bandwidth_mbps=1000, # High bandwidth
    ),
)

# Use custom configuration
manager = TestEnvironmentManager(config=custom_config)
```

## Available Fixtures

### Environment Configuration Fixtures

- `hardware_spec_default`: Standard RTX 3070 Ti optimized specification
- `hardware_spec_high_performance`: High-performance configuration
- `network_conditions_default`: Standard network conditions
- `network_conditions_slow`: Degraded network conditions for edge case testing
- `environment_isolation_default`: Standard isolation configuration
- `environment_isolation_strict`: Maximum isolation configuration

### Environment Manager Fixtures

- `test_environment_manager`: Standard environment manager
- `test_environment_manager_performance`: Performance-optimized manager
- `isolated_test_environment`: Complete isolated environment with cleanup
- `performance_test_environment`: Performance-optimized environment
- `environment_validation_results`: Pre-computed validation results

### Usage in Tests

```python
def test_with_custom_hardware(hardware_spec_high_performance):
    """Test with high-performance hardware specification."""
    assert hardware_spec_high_performance.cpu_cores >= 4
    assert hardware_spec_high_performance.max_model_batch_size == 16

def test_with_isolated_environment(isolated_test_environment):
    """Test with complete isolated environment setup."""
    env = isolated_test_environment

    # Environment is validated and ready
    assert env["validation_results"]["valid"] is True

    # Resources are allocated and isolated
    allocation = env["resource_allocation"]
    assert len(allocation.allocated_ports) > 0
```

## Environment Validation

The system automatically validates:

### Hardware Compatibility

- CPU core availability
- Memory availability
- Disk space availability
- GPU memory constraints (RTX 3070 Ti)
- Platform compatibility

### Network Conditions

- Network reachability
- Basic connectivity tests
- Timeout configuration validation

### Resource Availability

- Port range availability
- Memory allocation limits
- Process isolation capabilities

## Integration with Existing Infrastructure

### Resource Manager Integration

```python
# Automatic integration with existing ResourceManager
with manager.setup_test_environment() as env_info:
    allocation = env_info["resource_allocation"]

    # Ports are automatically allocated from ResourceManager
    ports = allocation.allocated_ports

    # Memory limits are enforced
    memory_limit = allocation.memory_limit_mb
```

### Browser Configuration Integration

```python
# Automatic browser matrix configuration
with manager.setup_test_environment() as env_info:
    browser_matrix = env_info["browser_matrix"]

    # Browsers configured for current hardware specs
    browsers = browser_matrix.browsers  # ["chrome", "firefox"]
    window_sizes = browser_matrix.window_sizes  # [(1920, 1080), (1366, 768)]
```

### Environment Variable Integration

```python
# Automatic environment variable setup
with manager.setup_test_environment() as env_info:
    # Environment variables are automatically configured:
    # - TEST_WORKER_ID
    # - CRACKSEG_ENV=test
    # - Memory and CPU limits
    # - Selenium configuration

    os.environ["TEST_WORKER_ID"]  # Automatically set
```

## Pytest Markers

The system registers custom pytest markers:

- `@pytest.mark.isolated_environment`: Tests requiring isolated environments
- `@pytest.mark.performance_environment`: Tests requiring high-performance setup
- `@pytest.mark.hardware_validation`: Tests validating hardware compatibility
- `@pytest.mark.network_validation`: Tests validating network conditions

## Configuration Options

### Hardware Specification Options

```python
HardwareSpecification(
    cpu_cores=4,                    # CPU cores required
    memory_mb=8192,                 # System memory in MB
    disk_space_mb=20480,            # Free disk space in MB
    network_bandwidth_mbps=1000,    # Network bandwidth
    browser_instances=4,            # Concurrent browser instances
    concurrent_tests=6,             # Parallel test processes
    gpu_memory_mb=8192,             # GPU memory (RTX 3070 Ti)
    max_model_batch_size=16,        # Maximum ML model batch size
)
```

### Network Conditions Options

```python
NetworkConditions(
    latency_ms=50,                  # Network latency
    bandwidth_mbps=100,             # Bandwidth limit
    packet_loss_percent=0.0,        # Packet loss simulation
    jitter_ms=10,                   # Network jitter
    connection_timeout_sec=30,      # Connection timeout
    retry_attempts=3,               # Retry attempts
)
```

### Environment Isolation Options

```python
EnvironmentIsolation(
    process_isolation=True,         # Enable process isolation
    network_isolation=True,         # Enable network isolation
    filesystem_isolation=True,      # Enable filesystem isolation
    port_range=(8600, 8699),        # Port allocation range
    temp_dir_prefix="crackseg_test_", # Temp directory prefix
    cleanup_on_exit=True,           # Automatic cleanup
    memory_limit_mb=4096,           # Memory limit per test
    cpu_limit_percent=80,           # CPU usage limit
    disk_limit_mb=10240,            # Disk usage limit
)
```

## Error Handling and Troubleshooting

### Common Issues

1. **Hardware Compatibility Errors**

   ```python
   # Check system compatibility
   manager = TestEnvironmentManager()
   validation = manager.validate_environment()

   if not validation["valid"]:
       print("Validation errors:", validation["errors"])
       print("Warnings:", validation["warnings"])
   ```

2. **Resource Allocation Failures**

   ```python
   # Monitor resource allocation
   with manager.setup_test_environment() as env_info:
       allocation = env_info["resource_allocation"]
       print(f"Allocated memory: {allocation.memory_limit_mb}MB")
       print(f"Allocated ports: {allocation.allocated_ports}")
   ```

3. **Network Connectivity Issues**

   ```python
   # Test network conditions
   checker = SystemCapabilityChecker()
   network_results = checker.check_network_conditions(
       NetworkConditions(connection_timeout_sec=10)
   )

   if not network_results["reachable"]:
       print("Network issues detected")
   ```

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# TestEnvironmentManager will log detailed information
manager = TestEnvironmentManager()
```

### Recovery Strategies

```python
# Reset environment on issues
manager = TestEnvironmentManager()
manager.reset_environment()

# Check status
status = manager.get_environment_status()
print(f"Environment setup: {status['setup_complete']}")
```

## Performance Considerations

### RTX 3070 Ti Optimizations

- **VRAM Limit**: 8GB maximum, batch sizes optimized accordingly
- **Memory Management**: Conservative memory allocation to prevent OOM
- **Concurrent Testing**: Limited browser instances to prevent resource contention

### Resource Efficiency

- **Port Allocation**: Efficient port range management prevents conflicts
- **Process Isolation**: Prevents resource leaks between test runs
- **Automatic Cleanup**: Ensures resources are freed after tests

## Examples

See `tests/e2e/test_environment_setup_demo.py` for comprehensive usage examples and validation tests.

## Related Components

- `tests/e2e/config/resource_manager.py`: Resource allocation and management
- `tests/e2e/config/browser_config_manager.py`: Browser configuration management
- `tests/docker/env_manager.py`: Environment variable management
- `tests/e2e/conftest.py`: Existing E2E testing fixtures
