# Deployment Scripts

This directory contains scripts for deploying and packaging the CrackSeg project for production environments.

## Structure

- **examples/**: Example deployment scripts demonstrating various deployment strategies
  - `packaging_example.py`: Advanced packaging and containerization examples
  - `deployment_example.py`: Basic deployment pipeline demonstration
  - `orchestration_example.py`: Blue-green, canary, and rolling deployment strategies
  - `artifact_selection_example.py`: Artifact selection and environment configuration

## Usage

### Basic Deployment

```bash
python scripts/deployment/examples/deployment_example.py
```

### Advanced Orchestration

```bash
python scripts/deployment/examples/orchestration_example.py
```

### Artifact Selection

```bash
python scripts/deployment/examples/artifact_selection_example.py
```

## Features

- **Multi-environment support**: Production, staging, development
- **Deployment strategies**: Blue-green, canary, rolling updates
- **Artifact optimization**: Quantization, pruning, compression
- **Health monitoring**: Automated health checks and rollback
- **Security scanning**: Container security validation

## Best Practices

1. Always test deployments in staging environment first
2. Use appropriate deployment strategy for your use case
3. Monitor health checks and performance metrics
4. Keep deployment configurations in version control
5. Document environment-specific requirements

## Integration

These scripts integrate with:

- `src/crackseg/utils/deployment/`: Core deployment utilities
- `src/crackseg/utils/traceability/`: Artifact tracking
- `infrastructure/deployment/`: Infrastructure configurations
