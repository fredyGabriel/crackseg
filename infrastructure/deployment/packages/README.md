# CrackSeg Deployment System

## Overview

The `infrastructure/deployment/packages/` directory contains production-ready deployment packages
for the CrackSeg pavement
crack segmentation system. This directory serves as the central hub for all deployment artifacts,
configurations, and deployment-specific resources.

## Purpose

The deployment system transforms trained ML models into production-ready applications that can be
deployed across various environments:

- **Docker Containers**: For containerized deployments
- **Kubernetes Clusters**: For orchestrated container management
- **Cloud Platforms**: For serverless and managed services
- **Edge Devices**: For on-device inference

## Directory Structure

```bash
infrastructure/deployment/packages/
├── test/                    # Test deployment configurations
├── test-package/           # Package testing environment
│   └── package/           # Deployment package template
│       ├── app/           # Application code
│       ├── config/        # Environment configurations
│       ├── docs/          # Deployment documentation
│       ├── helm/          # Helm charts for Kubernetes
│       ├── k8s/           # Kubernetes manifests
│       ├── scripts/       # Deployment scripts
│       ├── tests/         # Deployment tests
│       ├── Dockerfile     # Container definition
│       ├── docker-compose.yml  # Multi-service setup
│       └── requirements.txt    # Python dependencies
└── test-crackseg-model/   # Model-specific deployment
    └── package/           # CrackSeg model package
        ├── app/           # Model serving application
        ├── config/        # Model configurations
        ├── docs/          # Model documentation
        ├── scripts/       # Model deployment scripts
        └── tests/         # Model-specific tests
```

## Deployment Types

### 1. Test Deployments (`test/`)

**Purpose**: Development and testing environments for deployment validation.

**Use Cases**:

- Local development testing
- CI/CD pipeline validation
- Deployment script testing
- Environment configuration testing

### 2. Package Template (`test-package/`)

**Purpose**: Standardized deployment package template with comprehensive deployment configurations.

**Components**:

- **Docker Support**: Multi-stage Dockerfile with optimized base images
- **Kubernetes Support**: Complete K8s manifests (deployment, service, ingress, HPA)
- **Helm Charts**: Templated Kubernetes deployments
- **Application Code**: FastAPI and Streamlit application templates
- **Configuration Management**: Environment-specific configs
- **Deployment Scripts**: Automated deployment and rollback scripts

### 3. Model-Specific Deployments (`test-crackseg-model/`)

**Purpose**: Specialized deployments for the CrackSeg ML model.

**Features**:

- Model serving endpoints
- Inference optimization
- Model versioning
- Performance monitoring
- Health checks

## Deployment Pipeline

### 1. Artifact Selection

The deployment system automatically selects the appropriate model artifacts based on:

- Model performance metrics
- Training completion status
- Version compatibility
- Environment requirements

### 2. Package Generation

Each deployment package includes:

- **Application Code**: Model serving and API endpoints
- **Dependencies**: Optimized requirements for production
- **Configuration**: Environment-specific settings
- **Documentation**: Deployment and usage guides
- **Tests**: Validation and health check scripts

### 3. Validation

Comprehensive validation ensures:

- **Functional Testing**: Model loading and inference
- **Performance Testing**: Response times and throughput
- **Security Scanning**: Vulnerability assessment
- **Compatibility Checks**: Environment compatibility

### 4. Deployment

Supported deployment targets:

- **Docker**: Containerized applications
- **Kubernetes**: Orchestrated deployments
- **Cloud Platforms**: AWS, GCP, Azure
- **Edge Devices**: IoT and mobile deployments

## Usage

### Local Development

```bash
# Build and run test deployment
cd infrastructure/deployment/packages/test-package/package
docker build -t crackseg-test .
docker run -p 8501:8501 crackseg-test
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes cluster
cd infrastructure/deployment/packages/test-package/package
kubectl apply -f k8s/
kubectl wait --for=condition=available deployment/crackseg
```

### Production Deployment

```bash
# Deploy production model
cd infrastructure/deployment/packages/test-crackseg-model/package
./scripts/deploy_production.sh
```

## Configuration

### Environment Variables

Key environment variables for deployment:

```bash
# Model Configuration
MODEL_PATH=/path/to/model.pth
MODEL_VERSION=v1.0.0
INFERENCE_DEVICE=cuda

# Application Configuration
APP_PORT=8501
APP_HOST=0.0.0.0
LOG_LEVEL=INFO

# Performance Configuration
BATCH_SIZE=1
MAX_WORKERS=4
MEMORY_LIMIT=2048MB
```

### Deployment Configurations

Each deployment package includes:

- **app_config.json**: Application settings
- **environment.json**: Environment-specific variables
- **values.yaml**: Helm chart values
- **deployment.yaml**: Kubernetes deployment spec

## Monitoring and Health Checks

### Health Check Endpoints

- `/health`: Application health status
- `/ready`: Readiness for traffic
- `/metrics`: Performance metrics
- `/model/info`: Model information

### Monitoring Features

- **Performance Metrics**: Inference time, throughput, memory usage
- **Error Tracking**: Exception monitoring and alerting
- **Resource Monitoring**: CPU, memory, GPU utilization
- **Custom Metrics**: Business-specific KPIs

## Security Considerations

### Container Security

- **Base Image Security**: Regular security updates
- **Dependency Scanning**: Automated vulnerability assessment
- **Runtime Security**: Non-root user execution
- **Network Security**: Minimal port exposure

### Model Security

- **Model Encryption**: Secure model storage and transfer
- **Input Validation**: Robust input sanitization
- **Access Control**: Authentication and authorization
- **Audit Logging**: Comprehensive activity logging

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   - Verify model file path and permissions
   - Check model format compatibility
   - Validate model dependencies

2. **Performance Issues**
   - Monitor resource usage
   - Optimize batch processing
   - Check GPU memory allocation

3. **Deployment Failures**
   - Validate configuration files
   - Check network connectivity
   - Verify resource availability

### Debug Commands

```bash
# Check container logs
docker logs crackseg-container

# Inspect Kubernetes pods
kubectl describe pod crackseg-pod
kubectl logs crackseg-pod

# Test health endpoints
curl http://localhost:8501/health
curl http://localhost:8501/ready
```

## Best Practices

### Deployment Best Practices

1. **Version Control**: Always version deployment packages
2. **Environment Isolation**: Separate dev, staging, and production
3. **Rollback Strategy**: Maintain rollback capabilities
4. **Monitoring**: Implement comprehensive monitoring
5. **Security**: Regular security updates and scanning

### Performance Optimization

1. **Model Optimization**: Quantization and pruning
2. **Resource Management**: Efficient memory and CPU usage
3. **Caching**: Implement result caching where appropriate
4. **Load Balancing**: Distribute load across instances

## Related Documentation

- [Deployment Pipeline Architecture](docs/guides/deployment/DEPLOYMENT_PIPELINE_ARCHITECTURE.md)
- [CI/CD Integration Guide](docs/guides/cicd/ci_cd_integration_guide.md)
- [Model Serving Guide](docs/guides/deployment/MODEL_SERVING_GUIDE.md)
- [Production Deployment Guide](docs/guides/deployment/PRODUCTION_DEPLOYMENT_GUIDE.md)

## Contributing

When adding new deployment configurations:

1. Follow the existing package structure
2. Include comprehensive documentation
3. Add appropriate tests and validation
4. Update this README with new information
5. Ensure security and performance best practices

---

**Note**: This deployment system is designed to be extensible and maintainable. All deployments
follow the same patterns and conventions to ensure consistency across environments.
