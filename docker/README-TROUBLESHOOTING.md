# CrackSeg Docker Testing Infrastructure - Troubleshooting Guide

> **Complete troubleshooting reference for common issues and solutions**
>
> This document provides systematic approaches to diagnose and resolve issues with the CrackSeg
> Docker testing environment.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Common Issues](#common-issues)
3. [Service-Specific Issues](#service-specific-issues)
4. [Network Issues](#network-issues)
5. [Performance Issues](#performance-issues)
6. [Browser Issues](#browser-issues)
7. [Test Execution Issues](#test-execution-issues)
8. [Artifact Issues](#artifact-issues)
9. [Environment Issues](#environment-issues)
10. [Advanced Troubleshooting](#advanced-troubleshooting)

## Quick Diagnostics

### First Steps

```bash
# 1. Check overall system health
cd tests/docker
./scripts/docker-stack-manager.sh health

# 2. Check Docker system
docker system info
docker compose ps

# 3. Check service logs
./scripts/docker-stack-manager.sh logs --tail 50

# 4. Run system diagnostics
./scripts/system-monitor.sh diagnose
```

### Health Check Commands

```bash
# Infrastructure health
./scripts/docker-stack-manager.sh health

# Individual service health
docker compose exec selenium-hub curl -f http://localhost:4444/wd/hub/status
docker compose exec streamlit-app curl -f http://localhost:8501/_stcore/health

# Network connectivity
./scripts/network-manager.sh test-connectivity

# Resource usage
./scripts/system-monitor.sh resources
```

### Log Collection

```bash
# Collect all logs
./scripts/docker-stack-manager.sh logs > debug-logs.txt

# Service-specific logs
docker compose logs selenium-hub
docker compose logs streamlit-app
docker compose logs test-runner

# System logs (Linux)
journalctl -u docker.service --since "1 hour ago"
```

## Common Issues

### 1. Services Won't Start

#### Symptoms

- Containers fail to start
- Health checks fail
- Services crash immediately

#### Diagnosis

```bash
# Check container status
docker compose ps

# Check container logs
docker compose logs [service-name]

# Check resource usage
docker system df
free -h
```

#### Solutions

**Insufficient Resources**:

```bash
# Check available resources
./scripts/system-monitor.sh resources

# Reduce resource allocation
export SELENIUM_MEMORY=1g
export STREAMLIT_MEMORY=512m
./scripts/docker-stack-manager.sh restart
```

**Port Conflicts**:

```bash
# Check port usage
netstat -tulpn | grep :8501
netstat -tulpn | grep :4444

# Kill conflicting processes
sudo lsof -ti:8501 | xargs kill -9
sudo lsof -ti:4444 | xargs kill -9
```

**Docker Daemon Issues**:

```bash
# Restart Docker daemon
sudo systemctl restart docker

# Check Docker daemon logs
journalctl -u docker.service --since "10 minutes ago"
```

### 2. Tests Are Failing

#### Symptoms

- Tests timeout
- WebDriver errors
- Application not responding

#### Diagnosis

```bash
# Run tests with debug output
./scripts/run-test-runner.sh run --debug --verbose

# Check browser node status
./scripts/browser-manager.sh list
./scripts/browser-manager.sh validate

# Check application logs
docker compose logs streamlit-app --follow
```

#### Solutions

**Browser Node Issues**:

```bash
# Restart browser nodes
./scripts/browser-manager.sh restart

# Check browser capabilities
./scripts/browser-manager.sh config chrome:latest

# Increase browser timeout
export TEST_TIMEOUT=600
./scripts/run-test-runner.sh run
```

**Application Issues**:

```bash
# Restart Streamlit app
docker compose restart streamlit-app

# Check application health
curl -f http://localhost:8501/_stcore/health

# Access application manually
open http://localhost:8501
```

### 3. Performance Issues

#### Symptoms

- Slow test execution
- High resource usage
- Timeouts

#### Diagnosis

```bash
# Monitor resources
./scripts/system-monitor.sh dashboard

# Profile test execution
./scripts/system-monitor.sh profile tests/e2e/

# Check container stats
docker stats
```

#### Solutions

**Resource Optimization**:

```bash
# Enable performance mode
./scripts/docker-stack-manager.sh start --high-performance

# Adjust parallel workers
./scripts/run-test-runner.sh run --parallel-workers 2

# Use memory optimization
./scripts/docker-stack-manager.sh start --memory-optimized
```

**Network Optimization**:

```bash
# Optimize network settings
./scripts/network-manager.sh optimize

# Check network performance
./scripts/network-manager.sh analyze
```

## Service-Specific Issues

### Selenium Hub Issues

#### Common Problems

**Hub Not Responding**:

```bash
# Check hub health
curl -f http://localhost:4444/wd/hub/status

# Restart hub
docker compose restart selenium-hub

# Check hub logs
docker compose logs selenium-hub --tail 100
```

**Nodes Not Registering**:

```bash
# Check node registration
curl -s http://localhost:4444/grid/api/hub | jq '.value.ready'

# Restart nodes
docker compose restart chrome-node firefox-node edge-node

# Check network connectivity
./scripts/network-manager.sh test-connectivity
```

**Grid Console Not Accessible**:

```bash
# Check port forwarding
docker compose port selenium-hub 4444

# Verify network configuration
docker network inspect crackseg-backend-network
```

### Streamlit App Issues

#### Common Problems

**App Won't Start**:

```bash
# Check startup logs
docker compose logs streamlit-app

# Check Python dependencies
docker compose exec streamlit-app pip list

# Test minimal configuration
docker compose exec streamlit-app streamlit hello
```

**App Crashes**:

```bash
# Check memory usage
docker stats streamlit-app

# Increase memory limit
export STREAMLIT_MEMORY=2g
docker compose up -d streamlit-app

# Check for Python errors
docker compose logs streamlit-app | grep -i error
```

**App Not Accessible**:

```bash
# Check port binding
docker compose port streamlit-app 8501

# Test internal connectivity
docker compose exec test-runner curl -f http://streamlit:8501/_stcore/health

# Check firewall settings
sudo ufw status
```

### Test Runner Issues

#### Common Problems

**Runner Won't Execute Tests**:

```bash
# Check runner environment
docker compose exec test-runner env | grep TEST_

# Verify test files
docker compose exec test-runner ls -la /app/tests/e2e/

# Run minimal test
docker compose exec test-runner python -m pytest --version
```

**Import Errors**:

```bash
# Check Python path
docker compose exec test-runner python -c "import sys; print(sys.path)"

# Install missing dependencies
docker compose exec test-runner pip install -r requirements-testing.txt

# Rebuild container
docker compose build test-runner
```

## Network Issues

### Connection Problems

#### Services Can't Communicate

**Diagnosis**:

```bash
# Test network connectivity
./scripts/network-manager.sh test-connectivity

# Check network configuration
docker network ls
docker network inspect crackseg-frontend-network
docker network inspect crackseg-backend-network
```

**Solutions**:

```bash
# Recreate networks
./scripts/docker-stack-manager.sh cleanup --networks
./scripts/docker-stack-manager.sh start

# Reset network configuration
./scripts/network-manager.sh reset
```

#### DNS Resolution Issues

**Diagnosis**:

```bash
# Test DNS resolution
docker compose exec test-runner nslookup selenium-hub
docker compose exec test-runner nslookup streamlit

# Check network aliases
docker network inspect crackseg-backend-network | jq '.[].Containers'
```

**Solutions**:

```bash
# Use IP addresses temporarily
docker compose exec test-runner curl -f http://172.21.0.10:4444/wd/hub/status

# Restart DNS resolver
./scripts/network-manager.sh restart-dns
```

### Port Conflicts

#### External Port Issues

**Diagnosis**:

```bash
# Check port usage
netstat -tulpn | grep :8501
netstat -tulpn | grep :4444

# Check Docker port mapping
docker compose port streamlit-app 8501
docker compose port selenium-hub 4444
```

**Solutions**:

```bash
# Kill conflicting processes
sudo lsof -ti:8501 | xargs kill -9

# Use alternative ports
export STREAMLIT_PORT=8502
export SELENIUM_PORT=4445
docker compose up -d
```

## Performance Issues

### Slow Test Execution

#### Resource Constraints

**Diagnosis**:

```bash
# Monitor resource usage
./scripts/system-monitor.sh resources --interval 5

# Check container limits
docker inspect selenium-hub | jq '.[].HostConfig.Memory'
```

**Solutions**:

```bash
# Increase resource limits
export SELENIUM_MEMORY=4g
export CHROME_MEMORY=3g
./scripts/docker-stack-manager.sh restart

# Optimize parallel execution
./scripts/run-test-runner.sh run --parallel-workers auto
```

#### Network Latency

**Diagnosis**:

```bash
# Test network performance
./scripts/network-manager.sh benchmark

# Monitor network I/O
docker stats --format "table {{.Container}}\t{{.NetIO}}"
```

**Solutions**:

```bash
# Enable network optimization
./scripts/network-manager.sh optimize

# Use local DNS caching
./scripts/network-manager.sh enable-dns-cache
```

### Memory Issues

#### Out of Memory Errors

**Diagnosis**:

```bash
# Check system memory
free -h

# Check container memory usage
docker stats --no-stream

# Check for OOM kills
dmesg | grep -i "killed process"
```

**Solutions**:

```bash
# Increase swap space (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Reduce memory usage
export SELENIUM_MEMORY=1g
export CHROME_MEMORY=2g
./scripts/docker-stack-manager.sh restart --memory-optimized
```

## Browser Issues

### Chrome Node Issues

#### Chrome Crashes

**Diagnosis**:

```bash
# Check Chrome logs
docker compose logs chrome-node

# Check shared memory
docker exec chrome-node df -h /dev/shm

# Test Chrome directly
docker exec chrome-node google-chrome --version
```

**Solutions**:

```bash
# Increase shared memory
docker compose down
docker run --rm -it --shm-size=2g selenium/node-chrome:4.27.0

# Use Chrome options
export CHROME_OPTIONS="--no-sandbox --disable-dev-shm-usage"
./scripts/browser-manager.sh create chrome:latest
```

#### Firefox Node Issues

**Diagnosis**:

```bash
# Check Firefox logs
docker compose logs firefox-node

# Test Firefox directly
docker exec firefox-node firefox --version
```

**Solutions**:

```bash
# Restart Firefox node
docker compose restart firefox-node

# Clear Firefox profile
docker exec firefox-node rm -rf /tmp/.mozilla
```

### Browser-Specific Configuration

#### Custom Browser Settings

```bash
# Create custom Chrome configuration
./scripts/browser-manager.sh create chrome:custom \
    --memory 4g \
    --options "--disable-web-security,--allow-running-insecure-content"

# Create custom Firefox configuration
./scripts/browser-manager.sh create firefox:custom \
    --memory 3g \
    --options "-width=1920 -height=1080"
```

## Test Execution Issues

### Test Timeouts

#### WebDriver Timeouts

**Diagnosis**:

```bash
# Check test timeout configuration
docker compose exec test-runner env | grep TIMEOUT

# Run with extended timeout
./scripts/run-test-runner.sh run --timeout 600
```

**Solutions**:

```bash
# Increase global timeout
export TEST_TIMEOUT=900
./scripts/run-test-runner.sh run

# Use per-test timeouts
./scripts/run-test-runner.sh run --timeout 300 --slow-timeout 600
```

#### Page Load Timeouts

**Solutions**:

```bash
# Increase page load timeout
export PAGE_LOAD_TIMEOUT=60
./scripts/run-test-runner.sh run

# Use explicit waits
./scripts/run-test-runner.sh run --explicit-waits
```

### WebDriver Errors

#### Session Creation Failures

**Diagnosis**:

```bash
# Check available sessions
curl -s http://localhost:4444/grid/api/hub | jq '.value.ready'

# Check node capacity
./scripts/browser-manager.sh status
```

**Solutions**:

```bash
# Increase node capacity
export CHROME_MAX_SESSIONS=5
docker compose restart chrome-node

# Add more nodes
./scripts/browser-manager.sh scale chrome 3
```

## Artifact Issues

### Video Recording Problems

#### Videos Not Generated

**Diagnosis**:

```bash
# Check video recorder status
./scripts/artifact-manager.sh status video

# Check video directory
ls -la selenium-videos/

# Check recorder configuration
docker compose logs video-recorder
```

**Solutions**:

```bash
# Enable video recording
export VIDEO_RECORDING_ENABLED=true
./scripts/run-test-runner.sh run --videos

# Restart video recorder
docker compose restart video-recorder
```

### Report Generation Issues

#### Missing Reports

**Diagnosis**:

```bash
# Check report directory
ls -la test-results/reports/

# Check report generation logs
./scripts/artifact-manager.sh logs report-generator
```

**Solutions**:

```bash
# Regenerate reports
./scripts/artifact-manager.sh generate-report

# Use alternative report format
./scripts/run-test-runner.sh run --json-report
```

## Environment Issues

### Docker Environment

#### Docker Daemon Issues

**Diagnosis**:

```bash
# Check Docker daemon status
systemctl status docker

# Check Docker daemon logs
journalctl -u docker.service --since "1 hour ago"

# Test Docker connectivity
docker version
```

**Solutions**:

```bash
# Restart Docker daemon
sudo systemctl restart docker

# Reset Docker to defaults
docker system prune -a --volumes
```

#### Docker Compose Issues

**Diagnosis**:

```bash
# Check Compose version
docker compose version

# Validate Compose file
docker compose config

# Check Compose logs
docker compose logs
```

**Solutions**:

```bash
# Update Docker Compose
pip install docker-compose --upgrade

# Recreate services
docker compose down
docker compose up --build
```

### System Environment

#### Insufficient Resources

**Diagnosis**:

```bash
# Check system resources
./scripts/system-monitor.sh resources

# Check disk space
df -h

# Check memory usage
free -h
```

**Solutions**:

```bash
# Free up disk space
docker system prune -a
./scripts/artifact-manager.sh clean --older-than 7d

# Close unnecessary applications
# Add more RAM/swap if possible
```

#### Permission Issues

**Diagnosis**:

```bash
# Check file permissions
ls -la tests/docker/

# Check Docker socket permissions
ls -la /var/run/docker.sock

# Check user groups
groups $USER
```

**Solutions**:

```bash
# Add user to Docker group
sudo usermod -aG docker $USER
newgrp docker

# Fix file permissions
chmod +x tests/docker/scripts/*.sh
```

## Advanced Troubleshooting

### Debug Mode Execution

#### Enable Comprehensive Debugging

```bash
# Enable all debug options
export DEBUG=true
export VERBOSE=true
export TEST_DEBUG=true
./scripts/run-test-runner.sh run --debug --verbose --headless false
```

#### Interactive Debugging

```bash
# Access browser via noVNC
# Chrome: http://localhost:7900
# Firefox: http://localhost:7901
# Edge: http://localhost:7902

# Enable test debugging
./scripts/run-test-runner.sh run --interactive --pdb
```

### System State Analysis

#### Container State Analysis

```bash
# Comprehensive container inspection
./scripts/system-monitor.sh inspect-containers > container-state.json

# Network state analysis
./scripts/network-manager.sh inspect > network-state.json

# Volume state analysis
docker volume ls --format "table {{.Driver}}\t{{.Name}}\t{{.Scope}}"
```

#### Log Analysis

```bash
# Centralized log collection
./scripts/artifact-manager.sh collect-logs --all > full-system.log

# Error pattern analysis
grep -i error full-system.log | sort | uniq -c

# Performance pattern analysis
./scripts/system-monitor.sh analyze-logs full-system.log
```

### Recovery Procedures

#### Complete System Reset

```bash
# Nuclear option - complete reset
./scripts/docker-stack-manager.sh reset --force

# Clean Docker system
docker system prune -a --volumes

# Rebuild from scratch
./setup-local-dev.sh --force
./scripts/docker-stack-manager.sh start
```

#### Partial Recovery

```bash
# Service-specific recovery
./scripts/docker-stack-manager.sh recover selenium-hub

# Network recovery
./scripts/network-manager.sh recover

# Volume recovery
./scripts/artifact-manager.sh recover-volumes
```

### Performance Profiling

#### System Performance Profiling

```bash
# Profile system performance
./scripts/system-monitor.sh profile --duration 300

# Generate performance report
./scripts/system-monitor.sh report performance

# Benchmark against baseline
./scripts/system-monitor.sh benchmark --baseline
```

#### Test Performance Profiling

```bash
# Profile test execution
./scripts/run-test-runner.sh run --profile

# Analyze test performance
./scripts/artifact-manager.sh analyze-performance

# Compare performance across runs
./scripts/artifact-manager.sh compare-performance run1 run2
```

## Getting Help

### Support Channels

1. **Documentation**: Check other README files for specific topics
2. **Logs**: Always collect logs before asking for help
3. **System Info**: Include system information and Docker version
4. **Reproducible Steps**: Provide exact steps to reproduce the issue

### Information to Collect

```bash
# System information
./scripts/system-monitor.sh info > system-info.txt

# Docker information
docker version >> system-info.txt
docker compose version >> system-info.txt

# Service logs
./scripts/docker-stack-manager.sh logs > service-logs.txt

# Configuration
cat tests/docker/.env.local > config-info.txt (remove sensitive data)
```

### Emergency Contacts

- **Critical Production Issues**: Use immediate escalation procedures
- **Development Issues**: Create detailed bug reports
- **Performance Issues**: Include profiling data

---

**Troubleshooting Guide Version**: 1.0.0 - Comprehensive Issue Resolution
**Last Updated**: Subtask 13.11 - Documentation and Local Development Setup
**Compatibility**: CrackSeg v1.2+, Docker 20.10+, All supported platforms
