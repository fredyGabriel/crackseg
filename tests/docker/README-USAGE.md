# CrackSeg Docker Testing Infrastructure - Usage Guide

> **Complete guide for daily development and testing workflows**
>
> This document provides step-by-step instructions for using the CrackSeg Docker testing environment
> for development, testing, and CI/CD integration.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Operations](#basic-operations)
3. [Development Workflows](#development-workflows)
4. [Testing Scenarios](#testing-scenarios)
5. [Advanced Usage](#advanced-usage)
6. [Performance Optimization](#performance-optimization)
7. [CI/CD Integration](#cicd-integration)
8. [Monitoring and Debugging](#monitoring-and-debugging)

## Quick Start

### First Time Setup

```bash
# 1. Setup local development environment
cd tests/docker
./setup-local-dev.sh

# 2. Start the infrastructure
./scripts/docker-stack-manager.sh start

# 3. Run your first test
./scripts/run-test-runner.sh run --browser chrome

# 4. Access services
# - Streamlit: http://localhost:8501
# - Selenium Grid: http://localhost:4444
```

### Daily Development Flow

```bash
# Start working
./scripts/dev/start-testing.sh

# Run tests during development
./scripts/dev/run-tests.sh

# Stop when done
./scripts/dev/stop-testing.sh
```

## Basic Operations

### Infrastructure Management

#### Starting Services

```bash
# Start all services (recommended for development)
./scripts/docker-stack-manager.sh start

# Start with specific profile
./scripts/docker-stack-manager.sh start --profile chrome,firefox

# Start in background mode
./scripts/docker-stack-manager.sh start --detach

# Start with monitoring
./scripts/docker-stack-manager.sh start --monitoring
```

#### Stopping Services

```bash
# Stop all services
./scripts/docker-stack-manager.sh stop

# Stop and clean volumes
./scripts/docker-stack-manager.sh stop --volumes

# Emergency stop (force)
./scripts/docker-stack-manager.sh stop --force
```

#### Health and Status

```bash
# Check overall health
./scripts/docker-stack-manager.sh health

# View service status
./scripts/docker-stack-manager.sh status

# Detailed diagnostics
./scripts/system-monitor.sh diagnose

# Real-time monitoring
./scripts/system-monitor.sh dashboard
```

### Test Execution

#### Basic Test Runs

```bash
# Run all tests with default settings
./scripts/run-test-runner.sh run

# Run with specific browser
./scripts/run-test-runner.sh run --browser firefox

# Run with coverage
./scripts/run-test-runner.sh run --coverage

# Run with video recording
./scripts/run-test-runner.sh run --videos
```

#### Test Pattern Selection

```bash
# Run specific test pattern
./scripts/run-test-runner.sh run tests/e2e/test_basic_functionality.py

# Run tests matching pattern
./scripts/run-test-runner.sh run -k "streamlit and ui"

# Run with markers
./scripts/run-test-runner.sh run -m "smoke or regression"

# Exclude slow tests
./scripts/run-test-runner.sh run -m "not slow"
```

#### Report Generation

```bash
# Generate HTML report
./scripts/run-test-runner.sh run --html-report

# Generate JSON report
./scripts/run-test-runner.sh run --json-report

# Generate both reports
./scripts/run-test-runner.sh run --html-report --json-report

# Custom report location
./scripts/run-test-runner.sh run --html-report --report-dir custom-reports
```

## Development Workflows

### Feature Development Workflow

```bash
# 1. Setup development environment
./setup-local-dev.sh

# 2. Start services
./scripts/docker-stack-manager.sh start

# 3. Develop your feature
# (Edit code in src/, scripts/, etc.)

# 4. Run tests continuously
./scripts/run-test-runner.sh run --browser chrome --watch

# 5. Debug failing tests with video
./scripts/run-test-runner.sh run --videos --debug

# 6. Generate final report
./scripts/run-test-runner.sh run --coverage --html-report --json-report
```

### Bug Investigation Workflow

```bash
# 1. Reproduce the issue
./scripts/run-test-runner.sh run --browser chrome --videos --verbose

# 2. Use debug mode for detailed logging
./scripts/run-test-runner.sh run --debug --headless false

# 3. Access browser via noVNC (if needed)
# Open: http://localhost:7900 (Chrome), 7901 (Firefox), 7902 (Edge)

# 4. Collect debugging artifacts
./scripts/artifact-manager.sh collect --include-videos --include-logs

# 5. Generate debug report
./scripts/artifact-manager.sh report debug
```

### Cross-Browser Testing Workflow

```bash
# 1. Create browser matrix
./scripts/browser-manager.sh matrix smoke_test

# 2. Run smoke tests across browsers
./scripts/docker-stack-manager.sh test-cross-browser smoke_test

# 3. Run compatibility tests
./scripts/browser-manager.sh matrix compatibility_test
./scripts/docker-stack-manager.sh test-cross-browser compatibility_test

# 4. Run mobile tests
./scripts/browser-manager.sh matrix mobile_test
./scripts/docker-stack-manager.sh test-cross-browser mobile_test

# 5. Run full matrix (all browsers + mobile)
./scripts/docker-stack-manager.sh test-cross-browser full_matrix
```

## Testing Scenarios

### Smoke Testing

**Purpose**: Quick validation of core functionality

```bash
# Basic smoke test
./scripts/run-test-runner.sh run -m smoke --browser chrome

# Cross-browser smoke test
./scripts/docker-stack-manager.sh test-cross-browser smoke_test

# Smoke test with monitoring
./scripts/run-test-runner.sh run -m smoke --monitoring --timeout 120
```

### Regression Testing

**Purpose**: Comprehensive testing after changes

```bash
# Full regression suite
./scripts/run-test-runner.sh run -m regression --coverage --html-report

# Regression with cross-browser
./scripts/docker-stack-manager.sh test-cross-browser compatibility_test \
    tests/e2e/ --coverage

# Parallel regression testing
./scripts/run-test-runner.sh run -m regression --parallel-workers 4
```

### Performance Testing

**Purpose**: Validate application performance

```bash
# Performance test suite
./scripts/run-test-runner.sh run -m performance --monitoring

# Load testing with monitoring
./scripts/run-test-runner.sh run -m load --monitoring --timeout 600

# Memory profiling
./scripts/system-monitor.sh profile-memory tests/e2e/test_memory_intensive.py
```

### Mobile Testing

**Purpose**: Mobile responsiveness and functionality

```bash
# Mobile emulation tests
./scripts/browser-manager.sh matrix mobile_test
./scripts/docker-stack-manager.sh test-cross-browser mobile_test

# Specific device testing
./scripts/browser-manager.sh create chrome:pixel_5
./scripts/run-test-runner.sh run --browser chrome:pixel_5

# Mobile performance testing
./scripts/run-test-runner.sh run -m mobile --monitoring --device pixel_5
```

## Advanced Usage

### Custom Test Configurations

#### Environment-Specific Testing

```bash
# Local development testing
export TEST_ENV=local
./scripts/run-test-runner.sh run --config local

# Staging environment testing
export TEST_ENV=staging
./scripts/run-test-runner.sh run --config staging --timeout 600

# Production-like testing
export TEST_ENV=production
./scripts/run-test-runner.sh run --config production --strict
```

#### Custom Browser Configurations

```bash
# Create custom browser
./scripts/browser-manager.sh create chrome:custom \
    --memory 4g \
    --viewport 1920x1080 \
    --options "--disable-web-security,--allow-running-insecure-content"

# Use custom configuration
./scripts/run-test-runner.sh run --browser chrome:custom
```

### Parallel Execution

#### Multi-Worker Testing

```bash
# Auto-detect worker count
./scripts/run-test-runner.sh run --parallel-workers auto

# Specific worker count
./scripts/run-test-runner.sh run --parallel-workers 4

# Distributed across browsers
./scripts/run-test-runner.sh run \
    --browser chrome,firefox,edge \
    --parallel-workers 3
```

#### Browser-Specific Parallelization

```bash
# Parallel within each browser
./scripts/docker-stack-manager.sh test-cross-browser full_matrix \
    --parallel-per-browser 2

# Mixed parallel execution
./scripts/e2e-test-orchestrator.sh run \
    --browsers chrome:2,firefox:2,edge:1 \
    --parallel-workers 5
```

### Artifact Management

#### Collection and Archiving

```bash
# Collect all artifacts
./scripts/artifact-manager.sh collect

# Collect specific types
./scripts/artifact-manager.sh collect --type reports,videos,screenshots

# Archive with timestamp
./scripts/artifact-manager.sh archive --timestamp

# Clean old artifacts
./scripts/artifact-manager.sh clean --older-than 7d
```

#### Report Generation

```bash
# Generate summary report
./scripts/artifact-manager.sh report summary

# Generate detailed analysis
./scripts/artifact-manager.sh report detailed --include-performance

# Compare test runs
./scripts/artifact-manager.sh compare run1 run2

# Export for CI/CD
./scripts/artifact-manager.sh export junit --output ci-reports/
```

## Performance Optimization

### Resource Optimization

#### Memory Management

```bash
# Optimize for low memory
export SELENIUM_MEMORY=1g
export STREAMLIT_MEMORY=512m
./scripts/docker-stack-manager.sh start --memory-optimized

# High performance mode
export SELENIUM_MEMORY=4g
export STREAMLIT_MEMORY=2g
./scripts/docker-stack-manager.sh start --high-performance
```

#### Network Optimization

```bash
# Enable network optimization
./scripts/network-manager.sh optimize

# Monitor network performance
./scripts/system-monitor.sh network-stats

# Tune network settings
./scripts/network-manager.sh tune --bandwidth 1000 --latency low
```

### Test Execution Optimization

#### Selective Testing

```bash
# Run only changed tests
./scripts/run-test-runner.sh run --changed-only

# Skip slow tests in development
./scripts/run-test-runner.sh run --fast-only

# Incremental testing
./scripts/run-test-runner.sh run --incremental --baseline main
```

#### Caching and Reuse

```bash
# Use test result cache
./scripts/run-test-runner.sh run --cache-results

# Reuse browser sessions
./scripts/run-test-runner.sh run --reuse-sessions

# Cache Docker images
./scripts/docker-stack-manager.sh cache-images
```

## CI/CD Integration

### GitHub Actions Integration

```yaml
# .github/workflows/e2e-tests.yml
name: E2E Tests
on: [push, pull_request]

jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Docker testing environment
        run: |
          cd tests/docker
          ./setup-local-dev.sh --validate --skip-docker

      - name: Run E2E tests
        run: |
          cd tests/docker
          ./scripts/e2e-test-orchestrator.sh full-suite \
            --browsers chrome,firefox \
            --parallel-workers 2 \
            --coverage \
            --artifacts-collection

      - name: Upload test artifacts
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: e2e-test-results
          path: |
            test-results/
            test-artifacts/
            selenium-videos/
```

### Jenkins Integration

```groovy
pipeline {
    agent any

    stages {
        stage('Setup') {
            steps {
                dir('tests/docker') {
                    sh './setup-local-dev.sh --validate'
                }
            }
        }

        stage('E2E Tests') {
            parallel {
                stage('Chrome Tests') {
                    steps {
                        dir('tests/docker') {
                            sh './scripts/run-test-runner.sh run --browser chrome --coverage'
                        }
                    }
                }
                stage('Firefox Tests') {
                    steps {
                        dir('tests/docker') {
                            sh './scripts/run-test-runner.sh run --browser firefox --coverage'
                        }
                    }
                }
            }
        }

        stage('Artifacts') {
            steps {
                dir('tests/docker') {
                    sh './scripts/artifact-manager.sh collect --archive'
                }

                publishHTML([
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: 'test-results/reports',
                    reportFiles: 'report.html',
                    reportName: 'E2E Test Report'
                ])
            }
        }
    }

    post {
        always {
            dir('tests/docker') {
                sh './scripts/docker-stack-manager.sh cleanup'
            }
        }
    }
}
```

### GitLab CI Integration

```yaml
# .gitlab-ci.yml
stages:
  - test

e2e-tests:
  stage: test
  image: docker:latest
  services:
    - docker:dind

  before_script:
    - cd tests/docker
    - ./setup-local-dev.sh --validate --skip-docker

  script:
    - ./scripts/e2e-test-orchestrator.sh ci-suite
      --browsers chrome,firefox
      --parallel-workers 2
      --coverage
      --artifacts-collection

  artifacts:
    when: always
    reports:
      junit: test-results/junit.xml
      coverage_report:
        coverage_format: cobertura
        path: test-results/coverage/coverage.xml
    paths:
      - test-results/
      - test-artifacts/
    expire_in: 1 week

  after_script:
    - cd tests/docker
    - ./scripts/docker-stack-manager.sh cleanup
```

## Monitoring and Debugging

### Real-Time Monitoring

#### System Dashboard

```bash
# Launch monitoring dashboard
./scripts/system-monitor.sh dashboard

# Monitor specific service
./scripts/system-monitor.sh monitor selenium-hub

# Resource monitoring
./scripts/system-monitor.sh resources --interval 5
```

#### Log Monitoring

```bash
# View all logs
./scripts/docker-stack-manager.sh logs

# Follow specific service logs
./scripts/docker-stack-manager.sh logs selenium-hub --follow

# Filter logs by level
./scripts/docker-stack-manager.sh logs --level ERROR
```

### Debugging Tools

#### Browser Debugging

```bash
# Access browser via noVNC
# Chrome: http://localhost:7900
# Firefox: http://localhost:7901
# Edge: http://localhost:7902

# Enable debug mode
./scripts/run-test-runner.sh run --debug --headless false

# Interactive debugging
./scripts/run-test-runner.sh run --interactive --pdb
```

#### Network Debugging

```bash
# Monitor network traffic
./scripts/network-manager.sh monitor

# Analyze network performance
./scripts/network-manager.sh analyze

# Test network connectivity
./scripts/network-manager.sh test-connectivity
```

### Performance Analysis

#### Test Performance

```bash
# Profile test execution
./scripts/system-monitor.sh profile tests/e2e/

# Memory usage analysis
./scripts/system-monitor.sh memory-analysis

# CPU usage monitoring
./scripts/system-monitor.sh cpu-monitor --duration 300
```

#### System Performance

```bash
# Generate performance report
./scripts/system-monitor.sh report performance

# Resource utilization analysis
./scripts/system-monitor.sh analyze-resources

# Benchmark system performance
./scripts/system-monitor.sh benchmark
```

## Common Commands Reference

### Essential Daily Commands

```bash
# Quick setup and start
./setup-local-dev.sh && ./scripts/docker-stack-manager.sh start

# Standard test run
./scripts/run-test-runner.sh run --browser chrome --coverage

# Health check
./scripts/docker-stack-manager.sh health

# Stop and clean
./scripts/docker-stack-manager.sh stop --volumes
```

### Troubleshooting Commands

```bash
# Diagnose issues
./scripts/system-monitor.sh diagnose

# Restart services
./scripts/docker-stack-manager.sh restart

# Clean and rebuild
./scripts/docker-stack-manager.sh cleanup --rebuild

# View detailed logs
./scripts/docker-stack-manager.sh logs --verbose
```

### Performance Commands

```bash
# Monitor resources
./scripts/system-monitor.sh dashboard

# Optimize performance
./scripts/docker-stack-manager.sh optimize

# Profile execution
./scripts/system-monitor.sh profile
```

---

**Usage Guide Version**: 1.0.0 - Comprehensive Docker Testing Infrastructure
**Last Updated**: Subtask 13.11 - Documentation and Local Development Setup
**Compatibility**: CrackSeg v1.2+, Docker 20.10+, Python 3.12+
