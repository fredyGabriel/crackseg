# CrackSeg Selenium Grid Infrastructure Guide

## Overview

This document provides comprehensive guidance for the Selenium Grid infrastructure implemented in
subtask 13.3. The grid enables distributed, multi-browser testing for the CrackSeg Streamlit application.

## Architecture

### Grid Topology

```txt
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │  Selenium Hub   │    │  Grid Console   │
│  Application    │    │   (Coordinator) │    │   (Monitor)     │
│  172.20.0.10    │    │   172.20.0.20   │    │  172.20.0.70    │
│    Port 8501    │    │   Port 4444     │    │   Port 4446     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌────────┴────────┐             │
         │              │                 │             │
         └──────────────▼─────────────────▼─────────────┘
                        │
              ┌─────────┼─────────┐
              │         │         │
    ┌─────────▼───┐ ┌───▼────┐ ┌──▼──────┐
    │ Chrome Node │ │Firefox │ │Edge Node│
    │172.20.0.30  │ │Node    │ │172.20.0.32│
    │  Port 5555  │ │172.20. │ │ Port 5555 │
    │             │ │0.31    │ │           │
    │             │ │Port    │ │           │
    │             │ │5555    │ │           │
    └─────────────┘ └────────┘ └─────────┘
```

### Key Components

1. **Selenium Hub**: Central coordinator managing browser nodes and test sessions
2. **Browser Nodes**: Chrome, Firefox, and Edge containers running browser instances
3. **Grid Console**: Monitoring interface for grid status and session management
4. **Test Runner**: Orchestrates test execution across multiple browsers
5. **Support Services**: Video recording, noVNC debugging, artifact management

## Configuration Details

### Selenium Hub Configuration

```yaml
environment:
  # Core Hub Settings
  - SE_HUB_HOST=selenium-hub
  - SE_HUB_PORT=4444
  - SE_GRID_MAX_SESSION=6
  - SE_SESSION_REQUEST_TIMEOUT=300

  # Grid Scaling Parameters
  - SE_GRID_HEARTBEAT_PERIOD=60000
  - SE_GRID_CLEAN_UP_CYCLE=5000
  - SE_NEW_SESSION_THREAD_POOL_SIZE=8

  # Resource Management
  - JAVA_OPTS=-Xmx2g -XX:+UseG1GC
```

### Browser Node Capabilities

#### Chrome Node

- **Max Instances**: 2 sessions per node
- **VNC Access**: Port 5900 (when debugging enabled)
- **Special Capabilities**: Hardware acceleration disabled, sandbox disabled
- **Resource Limits**: 2GB RAM, 1 CPU core

#### Firefox Node

- **Max Instances**: 2 sessions per node
- **VNC Access**: Port 5900 (when debugging enabled)
- **Special Capabilities**: Media disabled, notifications disabled
- **Resource Limits**: 2GB RAM, 1 CPU core

#### Edge Node (Optional)

- **Profile**: `edge` (requires `--profile edge` to start)
- **Max Instances**: 2 sessions per node
- **Platform**: Linux-based Microsoft Edge
- **Resource Limits**: 2GB RAM, 1 CPU core

## Management Scripts

### Grid Management Script

Use `tests/docker/scripts/manage-grid.sh` for comprehensive grid management:

```bash
# Start basic grid (Chrome + Firefox + Streamlit)
./tests/docker/scripts/manage-grid.sh start

# Start with debugging (includes noVNC)
./tests/docker/scripts/manage-grid.sh start debug

# Start with video recording
./tests/docker/scripts/manage-grid.sh start recording

# Start with Edge browser support
./tests/docker/scripts/manage-grid.sh start edge

# Monitor grid status in real-time
./tests/docker/scripts/manage-grid.sh monitor

# Scale Chrome nodes to 3 instances
./tests/docker/scripts/manage-grid.sh scale chrome 3

# Validate grid connectivity
./tests/docker/scripts/manage-grid.sh validate

# View hub logs
./tests/docker/scripts/manage-grid.sh logs selenium-hub

# Stop all services
./tests/docker/scripts/manage-grid.sh stop

# Clean up all resources
./tests/docker/scripts/manage-grid.sh cleanup
```

### Docker Compose Profiles

```bash
# Basic grid (default)
docker-compose -f tests/docker/docker-compose.test.yml up -d

# Grid with debugging
docker-compose -f tests/docker/docker-compose.test.yml --profile debug up -d

# Grid with video recording
docker-compose -f tests/docker/docker-compose.test.yml --profile recording up -d

# Grid with monitoring console
docker-compose -f tests/docker/docker-compose.test.yml --profile monitoring up -d

# Grid with Edge browser
docker-compose -f tests/docker/docker-compose.test.yml --profile edge up -d

# Combined profiles
docker-compose -f tests/docker/docker-compose.test.yml --profile debug --profile recording up -d
```

## Monitoring and Access URLs

### Primary Interfaces

| Service | URL | Purpose |
|---------|-----|---------|
| **Selenium Hub** | <http://localhost:4444> | Grid API endpoint |
| **Grid Console** | <http://localhost:4446> | Grid monitoring UI |
| **Streamlit App** | <http://localhost:8501> | Application under test |
| **noVNC Chrome** | <http://localhost:7900> | Debug Chrome sessions |
| **noVNC Firefox** | <http://localhost:7901> | Debug Firefox sessions |

### Grid Status Endpoints

```bash
# Grid status
curl http://localhost:4444/wd/hub/status

# Active sessions
curl http://localhost:4444/wd/hub/sessions

# Grid configuration
curl http://localhost:4444/wd/hub/config
```

## Browser Capabilities Configuration

### Chrome Configuration

```json
{
  "browserName": "chrome",
  "browserVersion": "latest",
  "platformName": "linux",
  "goog:chromeOptions": {
    "args": [
      "--no-sandbox",
      "--disable-dev-shm-usage",
      "--disable-gpu",
      "--disable-web-security"
    ],
    "prefs": {
      "profile.default_content_setting_values.notifications": 2,
      "profile.default_content_settings.popups": 0
    }
  }
}
```

### Firefox Configuration

```json
{
  "browserName": "firefox",
  "browserVersion": "latest",
  "platformName": "linux",
  "moz:firefoxOptions": {
    "args": ["--headless", "--width=1920", "--height=1080"],
    "prefs": {
      "dom.webnotifications.enabled": false,
      "media.volume_scale": "0.0"
    }
  }
}
```

### Edge Configuration

```json
{
  "browserName": "MicrosoftEdge",
  "browserVersion": "latest",
  "platformName": "linux",
  "ms:edgeOptions": {
    "args": [
      "--no-sandbox",
      "--disable-dev-shm-usage",
      "--disable-gpu"
    ]
  }
}
```

## Test Execution

### Multi-Browser Testing

The grid supports parallel execution across multiple browser types:

```python
# Example test configuration
import pytest
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

@pytest.fixture(params=['chrome', 'firefox'])
def browser(request):
    """Parametrized fixture for multi-browser testing."""
    browser_name = request.param

    if browser_name == 'chrome':
        options = webdriver.ChromeOptions()
        caps = DesiredCapabilities.CHROME.copy()
    elif browser_name == 'firefox':
        options = webdriver.FirefoxOptions()
        caps = DesiredCapabilities.FIREFOX.copy()

    # Grid-specific settings
    caps['se:screenResolution'] = '1920x1080x24'
    caps['se:vncEnabled'] = True

    driver = webdriver.Remote(
        command_executor='http://localhost:4444/wd/hub',
        desired_capabilities=caps,
        options=options
    )

    yield driver
    driver.quit()
```

### Test Profiles

#### Smoke Testing

- **Browsers**: Chrome only
- **Parallel**: Disabled
- **Timeout**: 60 seconds
- **Use Case**: Quick validation

#### Regression Testing

- **Browsers**: Chrome + Firefox
- **Parallel**: Enabled
- **Timeout**: 300 seconds
- **Use Case**: Standard CI/CD pipeline

#### Full Testing

- **Browsers**: Chrome + Firefox + Edge
- **Parallel**: Enabled
- **Timeout**: 600 seconds
- **Use Case**: Release validation

## Network Configuration

### Custom Network

- **Name**: `crackseg-test-network`
- **Subnet**: `172.20.0.0/16`
- **Driver**: Bridge

### Static IP Assignments

```yaml
services:
  streamlit-app:    172.20.0.10
  selenium-hub:     172.20.0.20
  chrome-node:      172.20.0.30
  firefox-node:     172.20.0.31
  edge-node:        172.20.0.32
  test-runner:      172.20.0.40
  video-recorder:   172.20.0.50
  novnc:           172.20.0.60
  grid-console:     172.20.0.70
```

## Volume Management

### Shared Volumes

- **test-results**: Test artifacts, reports, screenshots
- **test-data**: Test input data and fixtures
- **selenium-videos**: Video recordings of test sessions

### Volume Mounting

```bash
# Access test results
docker volume inspect crackseg-test-results

# Copy artifacts to host
docker cp crackseg-test-runner:/app/test-results ./local-results

# Clean up volumes
docker volume rm crackseg-test-results crackseg-test-data crackseg-selenium-videos
```

## Troubleshooting

### Common Issues

#### Grid Not Starting

```bash
# Check Docker daemon
docker info

# Verify compose file
docker-compose -f tests/docker/docker-compose.test.yml config

# Check port conflicts
netstat -tulpn | grep -E '4444|8501'
```

#### Browser Nodes Not Connecting

```bash
# Check hub logs
docker logs crackseg-selenium-hub

# Check node logs
docker logs crackseg-chrome-node

# Verify network connectivity
docker exec crackseg-chrome-node ping selenium-hub
```

#### Session Creation Failures

```bash
# Check grid status
curl http://localhost:4444/wd/hub/status

# Verify node availability
curl http://localhost:4444/wd/hub/status | jq '.value.nodes'

# Test browser connectivity
./tests/docker/scripts/manage-grid.sh validate
```

### Debug Mode

Enable debug mode for detailed troubleshooting:

```bash
# Start with debugging
./tests/docker/scripts/manage-grid.sh start debug

# Access browser sessions via noVNC
# Chrome: http://localhost:7900
# Firefox: http://localhost:7901

# Enable script debugging
DEBUG=true ./tests/docker/scripts/manage-grid.sh status
```

### Performance Tuning

#### Resource Optimization

```yaml
# Hub tuning
environment:
  - JAVA_OPTS=-Xmx4g -XX:+UseG1GC -XX:+UseStringDeduplication
  - SE_GRID_MAX_SESSION=8

# Node tuning
deploy:
  resources:
    limits:
      memory: 3G
      cpus: '1.5'
```

#### Network Optimization

```yaml
# Increase shared memory
volumes:
  - /dev/shm:/dev/shm:rw

# Tune timeouts
environment:
  - SE_SESSION_REQUEST_TIMEOUT=600
  - SE_SESSION_RETRY_INTERVAL=5
```

## Security Considerations

### Network Isolation

- Services run in isolated Docker network
- No direct host network access
- Internal service discovery only

### Resource Limits

- Memory limits prevent resource exhaustion
- CPU limits ensure fair resource sharing
- Disk space monitoring via volume management

### Access Control

- No external authentication required (test environment)
- VNC access protected by profile requirements
- Grid console access limited to monitoring profile

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: E2E Testing
on: [push, pull_request]

jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Start Selenium Grid
        run: ./tests/docker/scripts/manage-grid.sh start
      - name: Run E2E Tests
        run: |
          docker-compose -f tests/docker/docker-compose.test.yml \
            run --rm test-runner
      - name: Collect Artifacts
        if: always()
        run: |
          docker cp crackseg-test-runner:/app/test-results ./artifacts
      - name: Upload Test Results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-results
          path: ./artifacts
```

### Jenkins Pipeline Example

```groovy
pipeline {
    agent any
    stages {
        stage('Setup') {
            steps {
                script {
                    sh './tests/docker/scripts/manage-grid.sh start'
                }
            }
        }
        stage('Test') {
            steps {
                script {
                    sh 'docker-compose -f tests/docker/docker-compose.test.yml run --rm test-runner'
                }
            }
        }
        stage('Collect Results') {
            steps {
                script {
                    sh 'docker cp crackseg-test-runner:/app/test-results ./test-artifacts'
                    archiveArtifacts artifacts: 'test-artifacts/**/*'
                }
            }
        }
    }
    post {
        always {
            sh './tests/docker/scripts/manage-grid.sh cleanup'
        }
    }
}
```

## Configuration Reference

### Complete Configuration Files

- **Grid Configuration**: `tests/docker/grid-config.json`
- **Docker Compose**: `tests/docker/docker-compose.test.yml`
- **Environment Variables**: `tests/docker/env-test.yml`
- **Management Script**: `tests/docker/scripts/manage-grid.sh`

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SE_HUB_HOST` | selenium-hub | Hub hostname |
| `SE_HUB_PORT` | 4444 | Hub port |
| `SE_GRID_MAX_SESSION` | 6 | Maximum concurrent sessions |
| `SE_SESSION_REQUEST_TIMEOUT` | 300 | Session request timeout (seconds) |
| `SE_NODE_MAX_INSTANCES` | 2 | Max instances per node |
| `SELENIUM_GRID_URL` | <http://selenium-hub:4444/wd/hub> | Grid endpoint URL |

## Conclusion

The Selenium Grid infrastructure provides a robust, scalable foundation for multi-browser E2E
testing of the CrackSeg Streamlit application. Key benefits include:

- **Multi-browser Support**: Chrome, Firefox, and Edge browsers
- **Parallel Execution**: Distributed test execution for faster feedback
- **Monitoring & Debugging**: Comprehensive monitoring and debugging capabilities
- **Resource Management**: Controlled resource allocation and cleanup
- **CI/CD Integration**: Ready for automated pipeline integration

The implementation fulfills all requirements for subtask 13.3, providing grid console access,
monitoring capabilities, and advanced browser node configurations for distributed test execution.
