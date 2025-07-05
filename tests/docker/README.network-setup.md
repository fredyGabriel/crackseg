# CrackSeg Multi-Network Docker Architecture

## Overview

This document describes the enhanced multi-network Docker architecture implemented for the CrackSeg
E2E testing infrastructure. The architecture separates services into logical network zones for
improved security, isolation, and service discovery.

## Architecture Design (Subtask 13.9)

### Network Topology

```txt
┌─────────────────────────────────────────────────────────────────┐
│                CrackSeg Multi-Network Architecture               │
└─────────────────────────────────────────────────────────────────┘

┌─ Frontend Network (172.20.0.0/24) ─────────────────────────────┐
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  Streamlit App  │    │  Test Runner    │                    │
│  │  172.20.0.10    │    │  172.20.0.40    │                    │
│  └─────────────────┘    └─────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ Bridge Connection
                                │
┌─ Backend Network (172.21.0.0/24) ──────────────────────────────┐
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  Selenium Hub   │    │  Test Runner    │                    │
│  │  172.21.0.10    │    │  172.21.0.40    │                    │
│  └─────────────────┘    └─────────────────┘                    │
│           │                                                     │
│           ├─ Chrome Node (172.21.0.30)                         │
│           ├─ Firefox Node (172.21.0.31)                        │
│           ├─ Edge Node (172.21.0.32)                           │
│           └─ Video Recorder (172.21.0.50)                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ Management Access
                                │
┌─ Management Network (172.22.0.0/24) ───────────────────────────┐
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  Grid Console   │    │  Health Checks  │                    │
│  │  172.22.0.70    │    │  All Services   │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  noVNC Debug    │    │  Video Mgmt     │                    │
│  │  172.22.0.60    │    │  172.22.0.50    │                    │
│  └─────────────────┘    └─────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

## Network Definitions

### 1. Frontend Network (`crackseg-frontend-network`)

- **Purpose**: Public-facing services accessible to external clients
- **Subnet**: `172.20.0.0/24`
- **Gateway**: `172.20.0.1`
- **Security Zone**: Public
- **Services**:
  - Streamlit App (`172.20.0.10`)
  - Test Runner (frontend interface) (`172.20.0.40`)

**Configuration**:

```yaml
crackseg-frontend-network:
  driver: bridge
  ipam:
    config:
      - subnet: 172.20.0.0/24
        gateway: 172.20.0.1
  driver_opts:
    com.docker.network.bridge.enable_icc: "true"
    com.docker.network.bridge.enable_ip_masquerade: "true"
```

### 2. Backend Network (`crackseg-backend-network`)

- **Purpose**: Internal service communication and processing
- **Subnet**: `172.21.0.0/24`
- **Gateway**: `172.21.0.1`
- **Security Zone**: Internal
- **Services**:
  - Selenium Hub (`172.21.0.10`)
  - Browser Nodes:
    - Chrome (`172.21.0.30`)
    - Firefox (`172.21.0.31`)
    - Edge (`172.21.0.32`)
  - Test Runner (backend interface) (`172.21.0.40`)
  - Video Recorder (`172.21.0.50`)

**Configuration**:

```yaml
crackseg-backend-network:
  driver: bridge
  ipam:
    config:
      - subnet: 172.21.0.0/24
        gateway: 172.21.0.1
  driver_opts:
    com.docker.network.bridge.enable_icc: "true"
    com.docker.network.bridge.enable_ip_masquerade: "false"
```

### 3. Management Network (`crackseg-management-network`)

- **Purpose**: Administrative, monitoring, and debugging services
- **Subnet**: `172.22.0.0/24`
- **Gateway**: `172.22.0.1`
- **Security Zone**: Administrative
- **Services**:
  - All services (management interfaces)
  - Grid Console (`172.22.0.70`)
  - noVNC Debug (`172.22.0.60`)
  - Health Check System

**Configuration**:

```yaml
crackseg-management-network:
  driver: bridge
  ipam:
    config:
      - subnet: 172.22.0.0/24
        gateway: 172.22.0.1
  driver_opts:
    com.docker.network.bridge.enable_icc: "true"
    com.docker.network.bridge.enable_ip_masquerade: "false"
```

## Service Discovery

### Network Aliases

Each service has multiple network aliases for different access patterns:

#### Streamlit App

- **Frontend**: `streamlit`, `app`
- **Management**: `streamlit-mgmt`

#### Selenium Hub

- **Backend**: `hub`, `selenium`, `grid-hub`
- **Management**: `hub-mgmt`, `selenium-hub-mgmt`

#### Browser Nodes

- **Chrome**: `chrome`, `chrome-browser` (backend), `chrome-mgmt` (management)
- **Firefox**: `firefox`, `firefox-browser` (backend), `firefox-mgmt` (management)
- **Edge**: `edge`, `edge-browser` (backend), `edge-mgmt` (management)

#### Test Runner

- **Frontend**: `test-runner-frontend`
- **Backend**: `test-runner`, `runner`
- **Management**: `test-runner-mgmt`

### Service Discovery Examples

```bash
# Test Runner accessing Streamlit (frontend network)
curl http://streamlit:8501/_stcore/health

# Test Runner accessing Selenium Hub (backend network)
curl http://hub:4444/wd/hub/status

# Chrome Node connecting to Selenium Hub (backend network)
curl http://selenium:4444/grid/api/hub

# Management accessing any service
curl http://streamlit-mgmt:8501/_stcore/health
curl http://hub-mgmt:4444/wd/hub/status
```

## Security Features

### Network Isolation

1. **Frontend Isolation**: Only Streamlit and Test Runner frontend interface exposed
2. **Backend Isolation**: Selenium Grid services isolated from external access
3. **Management Isolation**: Administrative services on separate network
4. **No IP Masquerading**: Backend and management networks don't route to external

### Access Control

- **Frontend → Backend**: Test Runner bridges networks for E2E testing
- **Backend ↔ Backend**: Full communication between Selenium services
- **Management → All**: Monitoring and health checks across all networks
- **External → Frontend**: Only Streamlit port 8501 exposed to host

### Port Restrictions

```yaml
# Only essential ports exposed to host
ports:
  - "8501:8501"    # Streamlit (frontend only)
  - "4444:4444"    # Selenium Hub (backend only)
  - "7900:7900"    # noVNC (debug profile only)
```

## Network Management

### Using the Network Manager Script

The `network-manager.sh` script provides comprehensive network management:

```bash
# Create all networks
./tests/docker/scripts/network-manager.sh start

# Check network status
./tests/docker/scripts/network-manager.sh status

# Test connectivity and service discovery
./tests/docker/scripts/network-manager.sh test

# Monitor network health
./tests/docker/scripts/network-manager.sh monitor

# Apply security policies
./tests/docker/scripts/network-manager.sh policies

# Clean up networks
./tests/docker/scripts/network-manager.sh stop
```

### Network Health Monitoring

The system includes automated network health monitoring:

1. **Container Connectivity**: Ping tests between containers
2. **Service Discovery**: DNS resolution validation
3. **Port Accessibility**: Service endpoint availability
4. **Security Compliance**: Policy validation

## Troubleshooting

### Common Network Issues

#### Service Discovery Failures

```bash
# Check if service is accessible by name
docker exec test-runner nslookup streamlit
docker exec test-runner nslookup hub

# Test port connectivity
docker exec test-runner nc -z streamlit 8501
docker exec test-runner nc -z hub 4444
```

#### Network Connectivity Problems

```bash
# Inspect network configuration
docker network inspect crackseg-frontend-network
docker network inspect crackseg-backend-network
docker network inspect crackseg-management-network

# Check container network assignments
docker inspect streamlit-app | jq '.NetworkSettings.Networks'
```

#### Container Communication Issues

```bash
# Test basic connectivity
docker exec streamlit-app ping hub
docker exec chrome-node ping selenium-hub

# Check network routes
docker exec test-runner ip route
```

### Network Diagnostics

#### Network Manager Diagnostics

```bash
# Comprehensive network test
./tests/docker/scripts/network-manager.sh test

# Continuous monitoring
./tests/docker/scripts/network-manager.sh monitor 5

# Security compliance check
./tests/docker/scripts/network-manager.sh compliance
```

#### Manual Diagnostics

```bash
# Check all networks
docker network ls | grep crackseg

# Inspect specific network
docker network inspect crackseg-backend-network --format='{{json .Containers}}'

# Test inter-service communication
docker compose -f tests/docker/docker-compose.test.yml exec test-runner \
  curl -f http://streamlit:8501/_stcore/health
```

## Migration from Legacy Network

### Backward Compatibility

The legacy `crackseg-test-network` is maintained for backward compatibility but marked as deprecated:

```yaml
crackseg-test-network:
  name: crackseg-test-network-legacy
  labels:
    - "crackseg.deprecated=true"
    - "crackseg.migration-target=multi-network"
```

### Migration Steps

1. **Update Service References**: Change hardcoded IPs to service names
2. **Test Service Discovery**: Verify all services can communicate via aliases
3. **Update Health Checks**: Use new network-aware endpoints
4. **Remove Legacy Dependencies**: Phase out legacy network references

## Performance Considerations

### Network Optimization

1. **MTU Configuration**: Set to 1500 for optimal performance
2. **Inter-Container Communication**: Enabled within each network
3. **Bridge Networks**: Optimized for container-to-container communication
4. **DNS Caching**: Docker's built-in DNS resolver with caching

### Resource Usage

- **Network Overhead**: Minimal additional overhead from multiple networks
- **Memory Usage**: Each network uses ~1-2MB additional memory
- **CPU Impact**: Negligible impact on CPU usage
- **Storage**: Network configuration stored in Docker daemon

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: E2E Testing with Multi-Network
on: [push, pull_request]

jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Multi-Network Architecture
        run: |
          ./tests/docker/scripts/network-manager.sh start

      - name: Start Services
        run: |
          docker compose -f tests/docker/docker-compose.test.yml up -d

      - name: Wait for Service Health
        run: |
          ./tests/docker/scripts/network-manager.sh test

      - name: Run E2E Tests
        run: |
          docker compose -f tests/docker/docker-compose.test.yml \
            run --rm test-runner pytest tests/e2e/ -v

      - name: Cleanup
        if: always()
        run: |
          ./tests/docker/scripts/network-manager.sh stop
```

## Security Best Practices

### Network Security Checklist

- [ ] **Network Segmentation**: Services grouped by function and security requirements
- [ ] **Minimal Exposure**: Only necessary ports exposed to host
- [ ] **Service Discovery**: Using service names instead of IP addresses
- [ ] **Access Control**: Appropriate network access between service tiers
- [ ] **Monitoring**: Network health and connectivity monitoring enabled
- [ ] **Documentation**: Network topology and access patterns documented
- [ ] **Backup**: Network configuration backed up regularly

### Compliance Validation

```bash
# Run security compliance check
./tests/docker/scripts/network-manager.sh compliance

# Check for exposed ports
docker compose -f tests/docker/docker-compose.test.yml ps

# Verify network isolation
./tests/docker/scripts/network-manager.sh test
```

## Advanced Configuration

### Custom Network Drivers

For specialized use cases, alternative network drivers can be configured:

```yaml
# Example: Overlay network for multi-host deployment
crackseg-overlay-network:
  driver: overlay
  driver_opts:
    encrypted: "true"
  ipam:
    config:
      - subnet: 10.0.0.0/24
```

### Network Plugins

Docker network plugins can extend functionality:

- **Weave**: Advanced networking features
- **Flannel**: Kubernetes-style networking
- **Calico**: Network policy enforcement

## Future Enhancements

### Planned Improvements

1. **Network Policies**: Implement fine-grained access control
2. **Service Mesh**: Consider Istio/Linkerd integration
3. **Monitoring**: Enhanced network metrics collection
4. **Automation**: Automated network configuration management
5. **Multi-Environment**: Support for staging/production variants

### Integration Opportunities

- **Kubernetes Migration**: Network design compatible with K8s networking
- **Service Discovery**: Integration with external service discovery systems
- **Load Balancing**: Network-aware load balancing strategies
- **Security**: Integration with network security tools

---

**Version**: 1.0 (Subtask 13.9)
**Author**: CrackSeg Project
**Last Updated**: Implementation of Multi-Network Architecture
**Next Review**: After subtask 13.11 completion
