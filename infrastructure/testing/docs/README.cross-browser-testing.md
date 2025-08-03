# Cross-Browser Testing Infrastructure

## Overview

This document describes the cross-browser testing infrastructure implemented for the CrackSeg
project as part of **Subtask 13.10 - Setup Cross-Browser Support Infrastructure**. The system
provides dynamic browser container management with support for multiple browser versions, mobile
emulation, and configurable test matrices.

## Architecture

### Core Components

1. **Browser Manager** (`scripts/browser-manager.sh`)
   - Dynamic browser container creation and management
   - Browser capability matrix configuration
   - Mobile device emulation support
   - Container lifecycle management

2. **Browser Capabilities** (`browser-capabilities.json`)
   - Centralized browser configuration management
   - Version-specific settings and resources
   - Mobile device profiles and capabilities
   - Test matrix definitions

3. **E2E Test Orchestrator** (Enhanced `scripts/e2e-test-orchestrator.sh`)
   - Cross-browser test execution coordination
   - Parallel test execution across browser matrices
   - Dynamic browser setup and teardown
   - Result aggregation and reporting

4. **Stack Manager** (Enhanced `scripts/docker-stack-manager.sh`)
   - Integration with existing Docker infrastructure
   - Browser matrix management commands
   - Cross-browser test orchestration

### Browser Support Matrix

#### Desktop Browsers

- **Chrome**: latest, stable, beta versions
- **Firefox**: latest, stable, beta versions
- **Edge**: latest, stable versions
- **Safari**: WebKit alternative (Linux compatible)

#### Mobile Emulation

- **Chrome Mobile**: Pixel 5, iPhone 12, Galaxy S21 Ultra
- **Firefox Mobile**: Responsive design mode
- **Edge Mobile**: Surface Duo emulation

## Usage Guide

### Quick Start

#### 1. List Available Browser Configurations

```bash
# Using browser manager directly
./scripts/browser-manager.sh browsers

# Using stack manager
./scripts/docker-stack-manager.sh browsers
```

#### 2. Create Browser Matrix

```bash
# Create smoke test matrix (Chrome + Firefox latest)
./scripts/browser-manager.sh matrix smoke_test

# Create compatibility test matrix (multiple versions)
./scripts/browser-manager.sh matrix compatibility_test

# Create mobile test matrix
./scripts/browser-manager.sh matrix mobile_test

# Create full matrix (all browsers + mobile)
./scripts/browser-manager.sh matrix full_matrix
```

#### 3. Create Individual Browser Containers

```bash
# Desktop browsers
./scripts/browser-manager.sh create chrome:latest
./scripts/browser-manager.sh create firefox:beta
./scripts/browser-manager.sh create edge:stable

# Mobile emulation
./scripts/browser-manager.sh create chrome:pixel_5
./scripts/browser-manager.sh create firefox:responsive
./scripts/browser-manager.sh create edge:surface_duo
```

#### 4. Run Cross-Browser Tests

```bash
# Run with specific matrix
./scripts/docker-stack-manager.sh test-cross-browser smoke_test

# Run with custom test patterns
./scripts/docker-stack-manager.sh test-cross-browser mobile_test tests/e2e/

# Force browser recreation
./scripts/docker-stack-manager.sh test-cross-browser full_matrix tests/e2e/ true
```

### Advanced Usage

#### Browser Manager Commands

```bash
# Validate environment
./scripts/browser-manager.sh validate

# List dynamic containers
./scripts/browser-manager.sh list

# Show browser configuration
./scripts/browser-manager.sh config chrome:beta

# Stop all dynamic browsers
./scripts/browser-manager.sh stop

# Remove all dynamic browsers
./scripts/browser-manager.sh remove

# Complete cleanup
./scripts/browser-manager.sh cleanup
```

#### Stack Manager Browser Commands

```bash
# Browser management
./scripts/docker-stack-manager.sh browsers
./scripts/docker-stack-manager.sh browser-matrix smoke_test
./scripts/docker-stack-manager.sh browser-create firefox:latest
./scripts/docker-stack-manager.sh browser-status
./scripts/docker-stack-manager.sh browser-cleanup

# Help for browser commands
./scripts/docker-stack-manager.sh browser-help
```

## Configuration

### Browser Capabilities File

The `browser-capabilities.json` file defines:

- **Browser Versions**: Image specifications and capabilities
- **Mobile Devices**: Device-specific emulation settings
- **Resource Allocation**: Memory and CPU limits per browser type
- **Test Matrices**: Predefined browser combinations
- **Selenium Grid**: Hub connection settings

### Test Matrices

#### Available Matrices

1. **smoke_test**
   - Browsers: chrome:latest, firefox:latest
   - Purpose: Basic functionality verification
   - Mobile: No

2. **compatibility_test**
   - Browsers: chrome:latest, chrome:stable, firefox:latest, firefox:stable, edge:latest
   - Purpose: Multi-version compatibility testing
   - Mobile: No

3. **mobile_test**
   - Browsers: chrome:pixel_5, chrome:iphone_12, firefox:responsive
   - Purpose: Mobile-specific testing
   - Mobile: Yes

4. **full_matrix**
   - Browsers: All desktop versions + all mobile emulations
   - Purpose: Comprehensive cross-browser testing
   - Mobile: Yes

### Mobile Device Configuration

The `mobile-browser-config.json` file provides:

- **Device Profiles**: Screen dimensions, pixel ratios, user agents
- **Network Conditions**: 3G, 4G, offline simulation
- **Geolocation Profiles**: GPS coordinates for testing
- **Browser Features**: Touch, camera, microphone capabilities
- **Test Scenarios**: Predefined mobile testing workflows

## Network Architecture

### Container Networks

- **Frontend Network** (172.20.0.0/24): Streamlit app access
- **Backend Network** (172.21.0.0/24): Selenium Grid and browser nodes
- **Management Network** (172.22.0.0/24): Health checks and monitoring

### Port Allocation

- **Base Port Range**: 5555-5600 for dynamic browser nodes
- **Selenium Hub**: 4444 (Grid API), 4442/4443 (Event Bus)
- **VNC Access**: 5900 per browser container
- **Node Ports**: Dynamic allocation based on instance ID

## Resource Management

### Memory Allocation

- **Desktop Browsers**: 2G limit, 1G reservation
- **Mobile Emulation**: 1.5G limit, 0.8G reservation
- **Selenium Hub**: 2.5G limit, 1G reservation

### CPU Allocation

- **Desktop Browsers**: 1.0 CPU limit, 0.5 reservation
- **Mobile Emulation**: 0.8 CPU limit, 0.4 reservation
- **Selenium Hub**: 1.5 CPU limit, 0.5 reservation

### Container Labels

Dynamic browser containers are tagged with:

- `crackseg.service=browser-node`
- `crackseg.dynamic=true`
- `crackseg.browser=<browser_name>`
- `crackseg.variant=<variant>`
- `crackseg.mobile=<true/false>`

## Integration with Existing Infrastructure

### Selenium Grid Integration

- Automatic registration with existing Selenium Hub
- Health check validation for browser node availability
- Dynamic capability negotiation
- Session management and timeout handling

### Docker Compose Integration

- Compatible with existing `docker-compose.test.yml`
- Uses existing volumes (selenium-videos, /dev/shm)
- Integrates with existing health check system
- Respects resource limits and restart policies

### Test Framework Integration

- Environment variables for browser selection
- Selenium Grid URL configuration
- Test result directory management
- Artifact collection and storage

## Monitoring and Debugging

### Health Checks

```bash
# Check Selenium Hub status
curl http://localhost:4444/grid/api/hub/status

# List registered nodes
curl http://localhost:4444/grid/api/hub | jq '.value.nodes'

# Browser-specific status
docker ps --filter "label=crackseg.dynamic=true"
```

### Logs and Debugging

```bash
# Browser container logs
docker logs crackseg-chrome-latest-1

# Selenium Hub logs
docker logs crackseg-selenium-hub

# Test execution logs
tail -f infrastructure/testing/test-results/*/logs/*.log
```

### VNC Access

Each browser container provides VNC access on port 5900:

```bash
# Connect to browser container
vncviewer localhost:5900
```

## Troubleshooting

### Common Issues

#### Browser Registration Failures

- Check Selenium Hub health
- Verify network connectivity
- Review browser container logs
- Validate capabilities configuration

#### Resource Constraints

- Monitor memory and CPU usage
- Adjust resource limits in capabilities file
- Check Docker daemon resources
- Review system performance

#### Container Conflicts

- Clean up existing dynamic browsers
- Check port availability
- Verify network configuration
- Review container naming conflicts

### Recovery Procedures

```bash
# Complete cleanup and restart
./scripts/browser-manager.sh cleanup
./scripts/docker-stack-manager.sh restart

# Reset Selenium Grid
docker restart crackseg-selenium-hub

# Recreate browser matrix
./scripts/browser-manager.sh matrix smoke_test true
```

## Performance Optimization

### Resource Tuning

1. **Parallel Execution**: Adjust worker count based on available CPU cores
2. **Memory Management**: Monitor container memory usage and adjust limits
3. **Network Optimization**: Use local image caching for faster startup
4. **Container Reuse**: Implement container pooling for frequently used browsers

### Testing Strategy

1. **Progressive Testing**: Start with smoke tests, expand to full matrix
2. **Targeted Testing**: Use specific browser matrices for focused testing
3. **Conditional Execution**: Skip mobile tests for desktop-only features
4. **Result Caching**: Cache successful test results to avoid retesting

## Extensions and Customization

### Adding New Browsers

1. Add browser configuration to `browser-capabilities.json`
2. Define version-specific capabilities and resources
3. Update test matrices as needed
4. Test browser registration and execution

### Custom Device Profiles

1. Add device specifications to `mobile-browser-config.json`
2. Include screen dimensions, user agents, and capabilities
3. Update browser capabilities for device support
4. Test mobile emulation functionality

### Integration with CI/CD

```yaml
# Example GitHub Actions integration
- name: Run Cross-Browser Tests
  run: |
    cd infrastructure/testing
    ./scripts/docker-stack-manager.sh start
    ./scripts/docker-stack-manager.sh test-cross-browser compatibility_test
    ./scripts/docker-stack-manager.sh browser-cleanup
```

## Security Considerations

### Network Isolation

- Browser containers isolated in backend network
- No direct external access to browser containers
- Selenium Grid acts as secure proxy
- Management network for monitoring only

### Resource Limits

- Strict memory and CPU limits per container
- Prevention of resource exhaustion
- Isolation between browser instances
- Container-level security policies

### Data Protection

- Temporary container file systems
- No persistent data storage in browser containers
- Secure cleanup of test artifacts
- Network traffic isolation

## Version Compatibility

### Selenium Grid Version

- Compatible with Selenium Grid 4.27.0+
- Supports latest WebDriver protocols
- Backward compatible with existing tests

### Browser Versions

- Chrome: Selenium node 4.27.0 (latest), 4.26.0 (stable)
- Firefox: Selenium node 4.27.0 (latest), 4.26.0 (stable)
- Edge: Selenium node 4.27.0 (latest), 4.26.0 (stable)

### Docker Requirements

- Docker Engine 20.10+
- Docker Compose 2.0+
- Available memory: 8GB+ recommended
- Available CPU: 4+ cores recommended

---

**Implementation Version**: 1.0 (Subtask 13.10)
**Last Updated**: December 2024
**Compatibility**: CrackSeg v1.2+, Docker Infrastructure Task 13
