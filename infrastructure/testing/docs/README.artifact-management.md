# CrackSeg Artifact Management System

Subtask 13.5 - Implement Artifact Management and Volume Configuration

## Overview

The CrackSeg Artifact Management System provides comprehensive management of test artifacts, reports,
screenshots, logs, and videos generated during Docker-based E2E testing. It implements a hybrid
approach using both named Docker volumes and bind mounts for optimal flexibility and persistence.

## Architecture

### Core Components

1. **artifact-manager.sh** - Main artifact management script
2. **Enhanced Docker volumes** - Hybrid named volumes + bind mounts
3. **Integration with run-test-runner.sh** - Automated artifact lifecycle
4. **Python test interfaces** - Programmatic access for testing

### Design Principles

- **Hybrid Storage Strategy**: Named volumes for persistence, bind mounts for development
- **Environment Variable Integration**: Seamless integration with existing env management (Subtask 13.6)
- **Automated Lifecycle**: Collection, cleanup, archiving with configurable policies
- **Type Safety**: Python interfaces with proper type annotations
- **Quality Gates**: All components pass basedpyright, black, ruff

## Volume Configuration

### Enhanced Volume Types

| Volume | Purpose | Retention | Management |
|--------|---------|-----------|------------|
| `test-results` | Test execution outputs | Configurable | Environment variables |
| `test-artifacts` | Collected artifacts | Configurable | artifact-manager.sh |
| `selenium-videos` | Video recordings | Short-term | Size-limited |
| `artifact-archive` | Long-term storage | 30 days | Automated archiving |
| `artifact-temp` | Temporary processing | 1 day | Auto-cleanup |

### Volume Labels

All volumes include comprehensive labels for management:

```yaml
labels:
  - "crackseg.volume=test-artifacts"
  - "crackseg.version=13.5"
  - "crackseg.purpose=artifact-collection"
  - "crackseg.managed-by=artifact-manager"
  - "crackseg.retention=configurable"
```

## Usage Guide

### Basic Commands

#### Artifact Collection

```bash
# Collect all artifacts from containers and volumes
./infrastructure/testing/scripts/artifact-manager.sh collect

# Collect with compression
./infrastructure/testing/scripts/artifact-manager.sh collect --compress

# Collect specific artifact types
./infrastructure/testing/scripts/artifact-manager.sh collect --type screenshots
```

#### Cleanup Operations

```bash
# Clean artifacts older than 7 days (dry run)
./infrastructure/testing/scripts/artifact-manager.sh cleanup --older-than 7 --dry-run

# Force cleanup with retention policy
./infrastructure/testing/scripts/artifact-manager.sh cleanup --older-than 3 --keep-latest 5 --force

# Clean Docker volumes
./infrastructure/testing/scripts/artifact-manager.sh cleanup --volumes --force
```

#### Archiving

```bash
# Create compressed archive
./infrastructure/testing/scripts/artifact-manager.sh archive --format tar.gz

# Archive with 30-day retention
./infrastructure/testing/scripts/artifact-manager.sh archive --retention-days 30

# Archive to custom location
./infrastructure/testing/scripts/artifact-manager.sh archive --storage-path /custom/archive
```

#### Verification

```bash
# Verify directory structure
./infrastructure/testing/scripts/artifact-manager.sh verify --structure

# Verify checksums
./infrastructure/testing/scripts/artifact-manager.sh verify --checksum

# Comprehensive verification
./infrastructure/testing/scripts/artifact-manager.sh verify --structure --checksum --completeness
```

#### Status and Reporting

```bash
# Show storage status
./tests/docker/scripts/artifact-manager.sh status

# List recent collections
find ./test-artifacts -name "collection_*" -type d -mtime -7
```

### Integration with Test Runner

#### Enhanced Commands

```bash
# Run tests with automatic artifact collection
./infrastructure/testing/scripts/run-test-runner.sh run --browser chrome

# Collect artifacts after execution
./infrastructure/testing/scripts/run-test-runner.sh collect-artifacts

# Clean up old artifacts
./infrastructure/testing/scripts/run-test-runner.sh cleanup-artifacts

# Archive current artifacts
./infrastructure/testing/scripts/run-test-runner.sh archive-artifacts
```

#### Environment Variables

Configure artifact management behavior:

```bash
# Core paths (managed by environment system from 13.6)
export TEST_RESULTS_PATH="./test-results"
export TEST_ARTIFACTS_PATH="./test-artifacts"
export SELENIUM_VIDEOS_PATH="./selenium-videos"
export ARCHIVE_PATH="./archived-artifacts"

# Artifact management flags (new in 13.5)
export ARTIFACT_COLLECTION_ENABLED="true"
export ARTIFACT_CLEANUP_ENABLED="false"
export ARTIFACT_ARCHIVE_ENABLED="false"
```

## Development Workflow

### Local Development

1. **Setup Environment**

   ```bash
   cd tests/docker
   cp env.local.template .env.local
   # Edit paths as needed
   ```

2. **Start Testing Environment**

   ```bash
   ./scripts/run-test-runner.sh run --profiles "recording,monitoring"
   ```

3. **Collect Artifacts**

   ```bash
   ./scripts/artifact-manager.sh collect --compress --debug
   ```

4. **Review Artifacts**

   ```bash
   ./scripts/artifact-manager.sh status
   ls -la test-artifacts/collection_*/
   ```

### CI/CD Integration

For automated environments:

```bash
# Environment-specific configuration
export ARTIFACT_COLLECTION_ENABLED="true"
export ARTIFACT_CLEANUP_ENABLED="true"
export ARTIFACT_ARCHIVE_ENABLED="true"

# Run tests with full artifact lifecycle
./scripts/run-test-runner.sh run
# Artifacts are automatically collected, cleaned, and archived
```

## Technical Details

### Hybrid Volume Strategy

The system implements a sophisticated hybrid approach:

#### Named Volumes

- **Purpose**: Persistent storage across container lifecycles
- **Benefits**: Docker-managed, consistent performance, easy backup
- **Use cases**: CI/CD environments, production testing

#### Bind Mounts

- **Purpose**: Direct host filesystem access
- **Benefits**: Real-time access, easy debugging, development flexibility
- **Use cases**: Local development, debugging, artifact inspection

#### Implementation

```yaml
# Docker Compose configuration
test-runner:
  volumes:
    # Named volumes for persistence
    - test-artifacts:/app/test-artifacts
    - artifact-archive:/app/archive

    # Bind mounts for development
    - ${TEST_ARTIFACTS_PATH:-./test-artifacts}:/app/host-artifacts
    - ${TEST_RESULTS_PATH:-./test-results}:/app/host-test-results
```

### Collection Mechanism

#### Container Artifact Collection

```bash
# Collect from running containers
collect_from_container() {
    local container="$1"
    local output_dir="$2"

    # Container logs
    docker logs "$container" > "$output_dir/container.log"

    # Application artifacts
    docker cp "$container:/app/test-results" "$output_dir/"
    docker cp "$container:/app/screenshots" "$output_dir/"
}
```

#### Volume Data Collection

```bash
# Collect from Docker volumes
collect_volume_data() {
    local volume_name="$1"
    local output_dir="$2"

    # Use temporary container to access volume
    docker run --rm \
        -v "$volume_name:/source:ro" \
        -v "$output_dir:/dest" \
        alpine:latest \
        cp -r /source/* /dest/
}
```

### Cleanup Policies

#### Time-based Cleanup

- **File age**: Remove files older than N days
- **Collection retention**: Keep latest N collections
- **Archive retention**: 30-day default with configurable override

#### Size-based Management

- **Video size limits**: Configurable per volume
- **Archive compression**: Automatic for space efficiency
- **Temporary cleanup**: Daily cleanup of processing volumes

### Testing

#### Unit Tests

Located in `tests/unit/docker/test_artifact_manager.py`:

- Script existence and permissions
- Command-line interface validation
- Environment variable integration
- Docker volume configuration
- Integration with test runner

#### Integration Tests

```python
# Python interface for testing
interface = ArtifactManagerInterface(project_root)
result = interface.collect_artifacts(compress=True)
assert result.returncode == 0
```

#### Quality Gates

```bash
# All artifact management code passes quality gates
basedpyright tests/unit/docker/test_artifact_manager.py
black tests/unit/docker/test_artifact_manager.py
ruff tests/unit/docker/test_artifact_manager.py --fix
```

## Troubleshooting

### Common Issues

#### Permission Problems

```bash
# Ensure scripts are executable
chmod +x infrastructure/testing/scripts/artifact-manager.sh
chmod +x infrastructure/testing/scripts/run-test-runner.sh
```

#### Docker Volume Issues

```bash
# Check volume status
docker volume ls --filter "name=crackseg-"

# Inspect volume configuration
docker volume inspect crackseg-test-artifacts

# Clean up volumes if needed
docker-compose -f infrastructure/testing/docker/docker-compose.test.yml down -v
```

#### Path Configuration

```bash
# Verify environment variables
./scripts/artifact-manager.sh status
# Check that paths are correctly configured
```

#### Collection Failures

```bash
# Debug collection process
./scripts/artifact-manager.sh collect --debug

# Check container status
docker ps -a --filter "name=crackseg-"

# Review container logs
docker logs crackseg-test-runner
```

### Debug Mode

Enable comprehensive debugging:

```bash
export DEBUG="true"
./scripts/artifact-manager.sh collect --debug
```

Output includes:

- Environment variable values
- Path validation
- Container status checks
- Volume accessibility tests
- Collection progress details

## Integration Points

### Environment Management (Subtask 13.6)

- Seamless integration with existing environment variable system
- Automatic path configuration
- Template-based environment setup

### Test Runner (Subtask 13.4)

- Post-execution artifact management
- Enhanced command interface
- Automatic lifecycle management

### Docker Infrastructure (Task 13)

- Integrated volume configuration
- Service dependency management
- Network and resource optimization

## Future Enhancements

### Planned Features

1. **Encryption Support**: Secure artifact archiving
2. **Remote Storage**: Cloud storage backends
3. **Metrics Collection**: Artifact size and performance metrics
4. **Auto-rotation**: Intelligent cleanup based on usage patterns
5. **Notification System**: Alerts for storage limits and failures

### Extension Points

- **Custom Collection Scripts**: Plugin architecture for specialized artifacts
- **Storage Backends**: Configurable storage providers
- **Retention Policies**: Advanced policy engine
- **Monitoring Integration**: Prometheus metrics export

## References

- **Docker Compose Configuration**: `infrastructure/testing/docker/docker-compose.test.yml`
- **Environment Management**: `infrastructure/testing/env_manager.py`
- **Test Runner Integration**: `infrastructure/testing/scripts/run-test-runner.sh`
- **Unit Tests**: `tests/unit/docker/test_artifact_manager.py`
- **Project Documentation**: `docs/guides/WORKFLOW_TRAINING.md`

---

**Artifact Management System Version**: 1.0
**Compatibility**: CrackSeg v1.2+, Task Master v0.17.0, Docker Compose v3.8+
**Last Updated**: Subtask 13.5 completion
**Maintainer**: CrackSeg Development Team
