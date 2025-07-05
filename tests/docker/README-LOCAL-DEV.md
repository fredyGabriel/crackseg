# CrackSeg Docker Testing Infrastructure - Local Development Guide

> **Focused guide for local development environment setup and daily workflows**

## Quick Setup

### Prerequisites

- Docker 20.10+ with Compose V2
- 8GB RAM (16GB recommended)
- 10GB free disk space
- Windows 10/11, Ubuntu 20.04+, or macOS 11+

### One-Command Setup

```bash
cd tests/docker
./setup-local-dev.sh
```

### Manual Setup Steps

```bash
# 1. Validate system requirements
./setup-local-dev.sh --validate

# 2. Initialize environment
./setup-local-dev.sh

# 3. Start infrastructure
./scripts/docker-stack-manager.sh start

# 4. Verify setup
./scripts/docker-stack-manager.sh health
```

## Daily Development Workflow

### Starting Work

```bash
# Quick start
./scripts/dev/start-testing.sh

# Or manual start
cd tests/docker
./scripts/docker-stack-manager.sh start
```

### Running Tests

```bash
# Basic test run
./scripts/dev/run-tests.sh

# Run specific tests
./scripts/run-test-runner.sh run tests/e2e/test_basic.py

# Run with specific browser
./scripts/run-test-runner.sh run --browser firefox

# Run with video recording for debugging
./scripts/run-test-runner.sh run --videos --debug
```

### Monitoring and Debugging

```bash
# Check system health
./scripts/docker-stack-manager.sh health

# View service logs
./scripts/docker-stack-manager.sh logs streamlit-app

# Access browser for debugging
# Chrome: http://localhost:7900
# Firefox: http://localhost:7901
# Selenium Grid Console: http://localhost:4444
```

### Stopping Work

```bash
# Quick stop
./scripts/dev/stop-testing.sh

# Or manual stop
./scripts/docker-stack-manager.sh stop
```

## Development Environment Configuration

### Environment Variables

The setup script creates `.env.local` with development-optimized settings:

```bash
# Test configuration
TEST_BROWSER=chrome
TEST_PARALLEL_WORKERS=auto
TEST_TIMEOUT=300
TEST_HEADLESS=true

# Feature flags
COVERAGE_ENABLED=true
VIDEO_RECORDING_ENABLED=false  # Enable for debugging
MONITORING_ENABLED=false       # Enable for performance analysis

# Performance tuning
SELENIUM_MEMORY=2g
STREAMLIT_MEMORY=1g
```

### Docker Compose Overrides

The setup creates `docker-compose.override.yml` for development:

- Source code mounted for live reload
- Exposed ports for direct access
- Development-friendly resource limits
- Extended logging and debugging options

## Common Development Tasks

### Code Changes

When you modify source code, the changes are automatically reflected due to volume mounts:

```bash
# No restart needed for most changes to:
# - src/ (source code)
# - tests/ (test files)
# - configs/ (configuration files)

# Restart needed for:
# - Dockerfile changes
# - Environment variable changes
# - Dependency changes
```

### Adding New Tests

```bash
# 1. Create test file
touch tests/e2e/test_new_feature.py

# 2. Run the new test
./scripts/run-test-runner.sh run tests/e2e/test_new_feature.py

# 3. Debug if needed
./scripts/run-test-runner.sh run tests/e2e/test_new_feature.py --videos --debug
```

### Browser Testing

```bash
# Test with different browsers
./scripts/run-test-runner.sh run --browser chrome
./scripts/run-test-runner.sh run --browser firefox
./scripts/run-test-runner.sh run --browser edge

# Cross-browser testing
./scripts/docker-stack-manager.sh test-cross-browser smoke_test
```

### Performance Testing

```bash
# Enable monitoring
export MONITORING_ENABLED=true
./scripts/docker-stack-manager.sh restart

# Run performance tests
./scripts/run-test-runner.sh run -m performance --monitoring

# View performance dashboard
./scripts/system-monitor.sh dashboard
```

## Troubleshooting

### Common Issues

**Services won't start:**

```bash
# Check resources
./scripts/system-monitor.sh resources

# Check Docker daemon
docker info

# Restart Docker if needed
sudo systemctl restart docker  # Linux
```

**Tests are failing:**

```bash
# Run with debug output
./scripts/run-test-runner.sh run --debug --verbose

# Check service health
./scripts/docker-stack-manager.sh health

# Access browser manually
open http://localhost:7900  # Chrome noVNC
```

**Performance issues:**

```bash
# Reduce resource usage
export SELENIUM_MEMORY=1g
export STREAMLIT_MEMORY=512m
./scripts/docker-stack-manager.sh restart

# Use fewer parallel workers
./scripts/run-test-runner.sh run --parallel-workers 1
```

### Getting Help

1. Check logs: `./scripts/docker-stack-manager.sh logs`
2. Run diagnostics: `./scripts/system-monitor.sh diagnose`
3. Review [README-TROUBLESHOOTING.md](README-TROUBLESHOOTING.md)

## Service URLs

When running locally, access these services:

- **Streamlit App**: <http://localhost:8501>
- **Selenium Grid Console**: <http://localhost:4444>
- **Chrome noVNC**: <http://localhost:7900> (password: secret)
- **Firefox noVNC**: <http://localhost:7901> (password: secret)
- **Edge noVNC**: <http://localhost:7902> (password: secret)

## Generated Files

The setup script creates:

- `.env.local` - Local environment configuration
- `docker-compose.override.yml` - Development overrides
- `scripts/dev/` - Convenience scripts
- Project directories for artifacts and results

---

**Local Development Guide Version**: 1.0.0
**Compatibility**: CrackSeg v1.2+, Docker 20.10+
