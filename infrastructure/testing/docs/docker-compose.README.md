# CrackSeg Docker Testing Infrastructure - Services Configuration

## Overview

This directory contains comprehensive Docker Compose configuration for end-to-end testing of the
CrackSeg Streamlit application using Selenium Grid infrastructure.

## Architecture

```txt
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Test Runner   │────│  Selenium Hub   │────│   Browser Nodes │
│                 │    │                 │    │ (Chrome/Firefox)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                   ┌─────────────────┐
                   │ Streamlit App   │
                   │                 │
                   └─────────────────┘
```

## Services

### Core Services

#### 1. **streamlit-app**

- **Purpose**: Main application under test
- **Image**: Built from `Dockerfile.streamlit`
- **Ports**: `8501:8501`
- **Health Check**: `curl -f http://localhost:8501/_stcore/health`
- **Networks**: `crackseg-test-network` (172.20.0.10)

#### 2. **selenium-hub**

- **Purpose**: Central hub for coordinating browser nodes
- **Image**: `selenium/hub:4.27.0`
- **Ports**:
  - `4444:4444` - Grid console and WebDriver API
  - `4442:4442` - Event bus publish port
  - `4443:4443` - Event bus subscribe port
- **Networks**: `crackseg-test-network` (172.20.0.20)

#### 3. **chrome-node**

- **Purpose**: Chrome browser instance for test execution
- **Image**: `selenium/node-chrome:4.27.0`
- **Resources**: 2GB RAM, 1 CPU core
- **Sessions**: Up to 2 concurrent sessions
- **Networks**: `crackseg-test-network` (172.20.0.30)

#### 4. **firefox-node**

- **Purpose**: Firefox browser instance for test execution
- **Image**: `selenium/node-firefox:4.27.0`
- **Resources**: 2GB RAM, 1 CPU core
- **Sessions**: Up to 2 concurrent sessions
- **Networks**: `crackseg-test-network` (172.20.0.31)

#### 5. **test-runner**

- **Purpose**: Executes pytest E2E test suites
- **Image**: Built from `Dockerfile.streamlit`
- **Mode**: Run once and exit
- **Dependencies**: All other services must be healthy
- **Networks**: `crackseg-test-network` (172.20.0.40)

### Optional Services (Profiles)

#### 6. **video-recorder** (Profile: `recording`)

- **Purpose**: Records browser sessions for debugging
- **Image**: `selenium/video:ffmpeg-7.1.0-20241206`
- **Output**: MP4 videos in `/videos` volume
- **Networks**: `crackseg-test-network` (172.20.0.50)

#### 7. **noVNC** (Profile: `debug`)

- **Purpose**: VNC access to browser for debugging
- **Image**: `selenium/standalone-chrome:4.27.0`
- **Ports**: `7900:7900` - Web-based VNC interface
- **Networks**: `crackseg-test-network` (172.20.0.60)

## Network Configuration

- **Network Name**: `crackseg-test-network`
- **Subnet**: `172.20.0.0/16`
- **Driver**: `bridge`
- **Service Discovery**: Hostname-based (e.g., `selenium-hub:4444`)

## Volume Management

### Named Volumes

| Volume | Purpose | Mount Point |
|--------|---------|-------------|
| `crackseg-test-results` | Test reports, screenshots, coverage | `/app/test-results` |
| `crackseg-test-data` | Test input data and fixtures | `/app/test-data` |
| `crackseg-selenium-videos` | Recorded browser sessions | `/videos` |

### Bind Mounts

- **Source Code**: `../..:/app:ro` - Read-only mount of project root
- **Shared Memory**: `/dev/shm:/dev/shm` - Browser performance optimization

## Usage Examples

### Basic Usage

```bash
# Start all core services
docker compose -f docker-compose.test.yml up

# Start in detached mode
docker compose -f docker-compose.test.yml up -d

# Stop and remove containers
docker compose -f docker-compose.test.yml down
```

### Using Profiles

```bash
# Start with video recording
docker compose -f docker-compose.test.yml --profile recording up

# Start with debugging support
docker compose -f docker-compose.test.yml --profile debug up -d

# Start with both recording and debugging
docker compose -f docker-compose.test.yml --profile recording --profile debug up
```

### Orchestration Scripts

```bash
# Start test environment with management script
./scripts/start-test-env.sh --mode standard --detach

# Run E2E tests
./scripts/run-e2e-tests.sh --browser chrome --headless

# Run with video recording
./scripts/run-e2e-tests.sh --browser both --record --verbose
```

## Service Configuration

### Environment Variables

#### Selenium Hub

```yaml
SE_HUB_HOST: selenium-hub
SE_HUB_PORT: 4444
SE_GRID_MAX_SESSION: 4
SE_GRID_BROWSER_TIMEOUT: 60
SE_SESSION_REQUEST_TIMEOUT: 300
```

#### Browser Nodes

```yaml
SE_NODE_MAX_INSTANCES: 2
SE_NODE_MAX_SESSIONS: 2
SE_SCREEN_WIDTH: 1920
SE_SCREEN_HEIGHT: 1080
SE_VNC_NO_PASSWORD: 1
```

#### Test Runner

```yaml
TEST_MODE: e2e
BROWSER: chrome
HEADLESS: true
SELENIUM_HUB_HOST: selenium-hub
STREAMLIT_URL: http://streamlit-app:8501
```

### Health Checks

All services include comprehensive health checks:

| Service | Health Check | Interval | Timeout | Retries |
|---------|-------------|----------|---------|---------|
| streamlit-app | Streamlit health endpoint | 30s | 10s | 3 |
| selenium-hub | Grid status endpoint | 30s | 10s | 3 |
| chrome-node | Node status endpoint | 30s | 10s | 3 |
| firefox-node | Node status endpoint | 30s | 10s | 3 |

### Resource Limits

Browser nodes are configured with resource limits to prevent resource exhaustion:

```yaml
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
    reservations:
      memory: 1G
      cpus: '0.5'
```

## Troubleshooting

### Common Issues

#### Services Not Starting

```bash
# Check service status
docker compose -f docker-compose.test.yml ps

# View service logs
docker compose -f docker-compose.test.yml logs selenium-hub

# Check health status
docker inspect crackseg-selenium-hub --format='{{.State.Health.Status}}'
```

#### Network Connectivity Issues

```bash
# Test network connectivity between services
docker exec crackseg-test-runner ping selenium-hub

# Check network configuration
docker network inspect crackseg-test-network
```

#### Browser Node Connection Problems

```bash
# Check if nodes are registered with hub
curl -s http://localhost:4444/grid/api/hub | jq '.slotCounts'

# View grid console
open http://localhost:4444/grid/console
```

#### Test Execution Failures

```bash
# Run tests with verbose output
docker compose -f docker-compose.test.yml run --rm test-runner \
  python -m pytest tests/e2e/ --verbose -s

# Check test artifacts
ls -la test-results/
```

### Debug Mode

When using the debug profile, you can access browser sessions via noVNC:

1. Start services with debug profile:

   ```bash
   docker compose -f docker-compose.test.yml --profile debug up -d
   ```

2. Open noVNC in browser:

   ```url
   http://localhost:7900
   ```

3. Password: `secret` (if prompted)

### Performance Tuning

#### Browser Performance

- Shared memory mount (`/dev/shm`) improves browser performance
- Adjust `SE_NODE_MAX_SESSIONS` based on system resources
- Configure screen resolution for optimal rendering

#### Parallel Execution

```yaml
# Test runner configuration for parallel execution
environment:
  - PARALLEL_WORKERS=auto  # Automatically detect optimal worker count
  - PARALLEL_WORKERS=2     # Specific worker count
```

#### Resource Allocation

```bash
# Monitor resource usage
docker stats

# Adjust memory limits in compose file
services:
  chrome-node:
    deploy:
      resources:
        limits:
          memory: 4G  # Increase if needed
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: E2E Tests

on: [push, pull_request]

jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Start test environment
        run: |
          cd tests/docker
          ./scripts/start-test-env.sh --mode standard --detach

      - name: Run E2E tests
        run: |
          cd tests/docker
          ./scripts/run-e2e-tests.sh --browser both --format junit

      - name: Upload test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: test-results
          path: test-results/
```

### Jenkins Pipeline Example

```groovy
pipeline {
    agent any

    stages {
        stage('E2E Tests') {
            steps {
                script {
                    sh '''
                        cd tests/docker
                        ./scripts/start-test-env.sh --mode standard --detach
                        ./scripts/run-e2e-tests.sh --browser chrome --format junit
                    '''
                }
            }
            post {
                always {
                    publishTestResults testResultsPattern: 'test-results/junit.xml'
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'test-results',
                        reportFiles: 'report.html',
                        reportName: 'E2E Test Report'
                    ])
                }
            }
        }
    }
}
```

## Security Considerations

### Container Security

- All containers run as non-root users
- Source code mounted read-only
- Network isolation through custom network
- No privileged mode enabled

### Secret Management

```yaml
# Example for production environments
services:
  test-runner:
    environment:
      - API_KEY_FILE=/run/secrets/api_key
    secrets:
      - api_key

secrets:
  api_key:
    external: true
```

## File Structure

```txt
tests/docker/
├── docker-compose.test.yml     # Main compose configuration
├── env-test.yml               # Environment variables
├── Dockerfile.streamlit       # Multi-stage Dockerfile
├── docker-entrypoint.sh      # Container initialization
├── pytest.ini               # Pytest configuration
├── .dockerignore            # Build context optimization
├── scripts/
│   ├── start-test-env.sh    # Environment startup
│   └── run-e2e-tests.sh     # Test execution
└── docker-compose.README.md # This documentation
```

## Best Practices

### Development Workflow

1. Use `start-test-env.sh` for consistent environment setup
2. Run tests frequently with `run-e2e-tests.sh`
3. Use debug profile for troubleshooting
4. Collect artifacts for analysis

### Testing Strategy

- Start with headless mode for faster execution
- Use GUI mode for debugging test failures
- Enable video recording for complex scenarios
- Implement retry mechanisms for flaky tests

### Maintenance

- Regularly update Selenium images
- Monitor resource usage and adjust limits
- Clean up volumes periodically
- Review and update documentation

---

For additional support and advanced configuration options, refer to the
[Selenium Grid documentation](https://selenium-grid.readthedocs.io/) and [Docker Compose reference](https://docs.docker.com/compose/).
