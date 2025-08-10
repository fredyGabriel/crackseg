# Environment Variable Management System

**Subtask 13.6 Implementation** - Comprehensive environment variable handling for CrackSeg Docker
testing infrastructure.

## Overview

This system provides robust environment variable management for different deployment environments
(local, staging, production, test) with validation, default value handling, and security features.

## Architecture

### Components

```txt
tests/docker/
├── env_manager.py              # Core environment management system
├── scripts/setup-env.sh        # Setup and validation script
├── env.*.template             # Environment-specific templates
├── docker-compose.test.yml    # Updated with env var support
└── README.environment-management.md
```

### Environment Types

- **local**: Development environment with debug features
- **staging**: CI/staging environment with production-like settings
- **production**: Production environment with security and performance optimizations
- **test**: Automated testing environment with isolation

## Quick Start

### 1. Setup Environment

```bash
# Setup local development environment
cd tests/docker
./scripts/setup-env.sh local --apply

# Setup test environment with validation
./scripts/setup-env.sh test --validate --apply

# Export configuration for review
./scripts/setup-env.sh staging --export staging-config.json
```

### 2. Use with Docker Compose

```bash
# Load environment and start services
source .env-test.sh
docker-compose -f docker-compose.test.yml up

# Or specify environment file directly
docker-compose -f docker-compose.test.yml --env-file .env.test up
```

## Environment Configuration

### Template System

Each environment has a template file that defines the default configuration:

- `env.local.template` - Local development defaults
- `env.staging.template` - Staging/CI defaults
- `env.production.template` - Production defaults
- `env.test.template` - Testing defaults

### Creating Environment Files

1. **From Template** (Recommended):

   ```bash
   ./scripts/setup-env.sh local  # Creates .env.local from template
   ```

2. **Manual Creation**:

   ```bash
   cp env.local.template .env.local
   # Edit .env.local with your specific values
   ```

### Environment Variables Structure

#### Core Environment Variables

```bash
# Environment Identification
NODE_ENV=local|staging|production|test
CRACKSEG_ENV=local|staging|production|test
PROJECT_NAME=crackseg

# Application Configuration
STREAMLIT_SERVER_HEADLESS=true|false
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
DEBUG=true|false
LOG_LEVEL=DEBUG|INFO|WARNING|ERROR

# Testing Configuration
TEST_BROWSER=chrome|firefox|chrome,firefox
TEST_PARALLEL_WORKERS=auto|1|2|4
TEST_TIMEOUT=300
TEST_HEADLESS=true|false
COVERAGE_ENABLED=true|false

# Service Endpoints
SELENIUM_HUB_HOST=localhost|selenium-hub
SELENIUM_HUB_PORT=4444
STREAMLIT_HOST=localhost|streamlit-app

# ML/Training Configuration
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
CUDA_VISIBLE_DEVICES=0|0,1,2,3
MODEL_CACHE_DIR=./cache/models
DATASET_CACHE_DIR=./cache/datasets

# Feature Flags
FEATURE_ADVANCED_METRICS=true|false
FEATURE_TENSORBOARD=true|false
FEATURE_MODEL_COMPARISON=true|false
FEATURE_EXPERIMENT_TRACKING=true|false
```

#### Security Variables (Handle Separately)

```bash
# Secrets (Never commit these)
API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here
DATABASE_PASSWORD=your_db_password_here
```

## Python API Usage

### Basic Usage

```python
from env_manager import EnvironmentManager, Environment

# Initialize manager
manager = EnvironmentManager()

# Detect current environment
env = manager.detect_environment()

# Load configuration
config = manager.create_config_from_env(env)

# Apply to current process
manager.apply_configuration(config)

# Export for Docker Compose
docker_env = manager.export_to_docker_compose(config)
```

### Validation and Error Handling

```python
from env_manager import EnvironmentConfig

try:
    config = EnvironmentConfig(
        streamlit_server_port=8501,
        test_browser="chrome",
        log_level="INFO"
    )
    print("Configuration is valid")
except ValueError as e:
    print(f"Configuration error: {e}")
```

### Configuration Customization

```python
# Create custom configuration
config = EnvironmentConfig(
    node_env="test",
    debug=False,
    test_browser="chrome,firefox",
    feature_flags={
        "FEATURE_ADVANCED_METRICS": True,
        "FEATURE_TENSORBOARD": False
    }
)

# Save to file
manager.save_config_to_file(config, Path("my-config.json"))
```

## Command Line Interface

### Setup Script Usage

```bash
# Basic syntax
./scripts/setup-env.sh [ENVIRONMENT] [OPTIONS]

# Examples
./scripts/setup-env.sh local --apply                    # Setup local development
./scripts/setup-env.sh staging --validate               # Validate staging config
./scripts/setup-env.sh test --export test-config.json   # Export test configuration
./scripts/setup-env.sh production --apply --verbose     # Setup production with details
./scripts/setup-env.sh local --force                    # Force overwrite existing files
```

### Python Module CLI

```bash
# Validate configuration
python env_manager.py --env test --validate

# Export configuration
python env_manager.py --env staging --export config.json

# Apply configuration (sets environment variables)
python env_manager.py --env local --apply
```

## Environment-Specific Configurations

### Local Development (local)

Optimized for development with debugging features:

```bash
DEBUG=true
LOG_LEVEL=DEBUG
STREAMLIT_SERVER_HEADLESS=false
TEST_HEADLESS=false
HOT_RELOAD_ENABLED=true
DEVELOPMENT_MODE=true
```

### Staging/CI (staging)

Production-like with CI optimizations:

```bash
DEBUG=false
LOG_LEVEL=INFO
STREAMLIT_SERVER_HEADLESS=true
TEST_HEADLESS=true
CI=true
TEST_PARALLEL_WORKERS=auto
```

### Production (production)

Security and performance optimized:

```bash
DEBUG=false
LOG_LEVEL=WARNING
ENABLE_SSL=true
RATE_LIMITING=true
AUTHENTICATION_REQUIRED=true
AUDIT_LOGGING=true
```

### Testing (test)

Isolated testing environment:

```bash
DEBUG=false
LOG_LEVEL=INFO
TEST_HEADLESS=true
COVERAGE_ENABLED=true
USE_MOCK_DATA=false
CLEAN_TEST_DATA=true
```

## Docker Integration

### Docker Compose Integration

The system integrates seamlessly with Docker Compose through environment variable substitution:

```yaml
services:
  streamlit-app:
    environment:
      - NODE_ENV=${NODE_ENV:-test}
      - CRACKSEG_ENV=${CRACKSEG_ENV:-test}
      - STREAMLIT_SERVER_HEADLESS=${STREAMLIT_SERVER_HEADLESS:-true}
      - DEBUG=${DEBUG:-false}
      - FEATURE_ADVANCED_METRICS=${FEATURE_ADVANCED_METRICS:-true}
```

### Volume Path Management

Environment variables control volume mounting:

```yaml
volumes:
  test-results:
    driver_opts:
      device: ${TEST_RESULTS_PATH:-./test-results}
```

### Usage Workflow

1. **Configure Environment**:

   ```bash
   ./scripts/setup-env.sh test --apply
   source .env-test.sh
   ```

2. **Start Services**:

   ```bash
   docker-compose -f docker-compose.test.yml up
   ```

3. **Verify Configuration**:

   ```bash
   docker-compose -f docker-compose.test.yml config
   ```

## Security Considerations

### Secret Management

1. **Never commit secrets** to version control
2. **Use separate secret files** for sensitive data
3. **Environment-specific secret handling**:
   - Local: Use dummy/development secrets
   - Staging: Use staging-specific secrets
   - Production: Use external secret management

### File Security

```bash
# Ensure .env files are excluded from git
echo ".env.*" >> .gitignore

# Set appropriate permissions
chmod 600 .env.production  # Restrict access to production secrets
```

### Validation

The system validates:

- Port number ranges (1024-65535)
- Log level values
- Browser names
- Timeout values (positive integers)
- Required fields

## Testing

### Unit Tests

```bash
# Run environment management tests
pytest tests/unit/docker/test_env_manager.py -v

# Test specific functionality
pytest tests/unit/docker/test_env_manager.py::TestEnvironmentConfig::test_validation -v
```

### Integration Tests

```bash
# Test full workflow
pytest tests/unit/docker/test_env_manager.py::TestEnvironmentIntegration -v
```

### Manual Testing

```bash
# Test script functionality
./scripts/setup-env.sh test --validate
./scripts/setup-env.sh local --export test-export.json

# Test Python module
python env_manager.py --env test --validate
python env_manager.py --env staging --export staging.json
```

## Troubleshooting

### Common Issues

1. **Template file not found**:

   ```bash
   # Ensure template files exist
   ls env.*.template
   ```

2. **Permission denied on script**:

   ```bash
   chmod +x scripts/setup-env.sh
   ```

3. **Python import errors**:

   ```bash
   # Ensure you're in the correct directory
   cd tests/docker
   python env_manager.py --help
   ```

4. **Environment variables not applied**:

   ```bash
   # Source the generated script
   source .env-local.sh
   echo $CRACKSEG_ENV
   ```

### Validation Errors

Common validation errors and fixes:

```bash
# Invalid port
STREAMLIT_SERVER_PORT=999  # Error: must be 1024-65535

# Invalid browser
TEST_BROWSER=invalid_browser  # Error: must be chrome,firefox,edge,safari

# Invalid log level
LOG_LEVEL=INVALID  # Error: must be DEBUG,INFO,WARNING,ERROR,CRITICAL
```

### Environment Detection Issues

```bash
# Check environment detection
python -c "
from env_manager import EnvironmentManager
manager = EnvironmentManager()
print(f'Detected: {manager.detect_environment()}')
"

# Set environment explicitly
export CRACKSEG_ENV=test
```

## Best Practices

1. **Always validate** configurations before deployment
2. **Use templates** as starting points for environment files
3. **Keep secrets separate** from configuration files
4. **Document environment-specific** requirements
5. **Test configurations** in isolated environments
6. **Version control templates** but not actual .env files
7. **Use CI/CD integration** for staging and production deployments

## Integration with CrackSeg Project

### ML Training Integration

```bash
# Configure for GPU training
CUDA_VISIBLE_DEVICES=0,1
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
MODEL_CACHE_DIR=/data/models

# Training environment
FEATURE_TENSORBOARD=true
FEATURE_EXPERIMENT_TRACKING=true
```

### Testing Integration

```bash
# E2E testing configuration
TEST_BROWSER=chrome
TEST_HEADLESS=true
SELENIUM_HUB_HOST=selenium-hub
COVERAGE_ENABLED=true
```

### Development Workflow

```bash
# Local development setup
./scripts/setup-env.sh local --apply
source .env-local.sh
docker-compose -f docker-compose.test.yml up streamlit-app
```

## Future Enhancements

- [ ] Kubernetes configuration support
- [ ] External secret management integration (HashiCorp Vault, AWS Secrets Manager)
- [ ] Environment promotion workflows
- [ ] Configuration drift detection
- [ ] Automated security scanning for secrets
- [ ] Integration with CI/CD pipelines

## References

- [Docker Compose Environment Variables](https://docs.docker.com/compose/environment-variables/)
- [Twelve-Factor App Config](https://12factor.net/config)
- CrackSeg Development Workflow: see `docs/guides/operational-guides/workflows/legacy/WORKFLOW_TRAINING.md`
- Docker Testing Infrastructure: see `infrastructure/testing/docs/docker-compose.README.md`
