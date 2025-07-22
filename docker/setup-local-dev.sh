#!/bin/bash

# CrackSeg Local Development Setup Script
# Cross-platform setup for Docker testing infrastructure
# Supports Windows (Git Bash/WSL), Linux, and macOS

set -euo pipefail

# Script metadata
SCRIPT_VERSION="1.0.0"
SCRIPT_NAME="setup-local-dev.sh"
CRACKSEG_DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${CRACKSEG_DOCKER_DIR}/../.." && pwd)"

# Color codes for output
declare -r RED='\033[0;31m'
declare -r GREEN='\033[0;32m'
declare -r YELLOW='\033[1;33m'
declare -r BLUE='\033[0;34m'
declare -r PURPLE='\033[0;35m'
declare -r CYAN='\033[0;36m'
declare -r WHITE='\033[1;37m'
declare -r NC='\033[0m' # No Color

# Configuration
declare -r MIN_DOCKER_VERSION="20.10.0"
declare -r MIN_COMPOSE_VERSION="2.0.0"
declare -r MIN_RAM_GB=8
declare -r RECOMMENDED_RAM_GB=16
declare -r MIN_DISK_GB=10

# Global variables
VERBOSE=false
VALIDATE_ONLY=false
FORCE_SETUP=false
SKIP_DOCKER_CHECK=false
DRY_RUN=false

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_debug() {
    if [[ "${VERBOSE}" == "true" ]]; then
        echo -e "${PURPLE}[DEBUG]${NC} $*"
    fi
}

log_step() {
    echo -e "${CYAN}[STEP]${NC} $*"
}

# Help function
show_help() {
    cat << EOF
${WHITE}CrackSeg Local Development Setup${NC}
${BLUE}Automated setup for Docker testing infrastructure${NC}

${WHITE}Usage:${NC}
    $0 [OPTIONS]

${WHITE}Options:${NC}
    -h, --help              Show this help message
    -v, --verbose           Enable verbose logging
    -V, --validate          Only validate system requirements (no setup)
    -f, --force             Force setup even if requirements not met
    -s, --skip-docker       Skip Docker validation (for CI environments)
    -d, --dry-run           Show what would be done without executing
    --version               Show script version

${WHITE}Examples:${NC}
    # Basic setup
    $0

    # Validate system only
    $0 --validate

    # Verbose setup with force
    $0 --verbose --force

    # Dry run to see what would happen
    $0 --dry-run

${WHITE}System Requirements:${NC}
    - Docker ${MIN_DOCKER_VERSION}+ with Compose V2
    - ${MIN_RAM_GB}GB RAM (${RECOMMENDED_RAM_GB}GB recommended)
    - ${MIN_DISK_GB}GB free disk space
    - Windows 10/11, Ubuntu 20.04+, or macOS 11+

${WHITE}Features:${NC}
    ✓ Cross-platform compatibility (Windows/Linux/macOS)
    ✓ Automatic system requirements validation
    ✓ Docker infrastructure setup and validation
    ✓ Environment configuration management
    ✓ Network setup and verification
    ✓ Health check system initialization
    ✓ Development environment optimization

${WHITE}Generated Files:${NC}
    - .env.local (local environment configuration)
    - docker-compose.override.yml (development overrides)
    - Local development configuration

EOF
}

# Version function
show_version() {
    echo "${SCRIPT_NAME} version ${SCRIPT_VERSION}"
    echo "CrackSeg Docker Testing Infrastructure"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -V|--validate)
                VALIDATE_ONLY=true
                shift
                ;;
            -f|--force)
                FORCE_SETUP=true
                shift
                ;;
            -s|--skip-docker)
                SKIP_DOCKER_CHECK=true
                shift
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            --version)
                show_version
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information."
                exit 1
                ;;
        esac
    done
}

# Detect operating system
detect_os() {
    case "$(uname -s)" in
        Linux*)     echo "linux" ;;
        Darwin*)    echo "macos" ;;
        CYGWIN*|MINGW*|MSYS*) echo "windows" ;;
        *)          echo "unknown" ;;
    esac
}

# Check if running in WSL
is_wsl() {
    if [[ -f /proc/version ]] && grep -q Microsoft /proc/version; then
        return 0
    fi
    return 1
}

# Version comparison function
version_compare() {
    local version1=$1
    local version2=$2

    # Remove 'v' prefix if present
    version1=${version1#v}
    version2=${version2#v}

    if [[ "$version1" == "$version2" ]]; then
        return 0
    fi

    # Split versions and compare
    local IFS='.'
    local i ver1=($version1) ver2=($version2)

    # Fill empty fields with zeros
    for ((i=${#ver1[@]}; i<${#ver2[@]}; i++)); do
        ver1[i]=0
    done
    for ((i=${#ver2[@]}; i<${#ver1[@]}; i++)); do
        ver2[i]=0
    done

    for ((i=0; i<${#ver1[@]}; i++)); do
        if [[ ${ver1[i]} -lt ${ver2[i]} ]]; then
            return 1
        elif [[ ${ver1[i]} -gt ${ver2[i]} ]]; then
            return 0
        fi
    done

    return 0
}

# System requirements validation
validate_system_requirements() {
    log_step "Validating system requirements..."

    local os_type
    os_type=$(detect_os)
    local requirements_met=true

    log_info "Operating System: $os_type"
    if is_wsl; then
        log_info "WSL environment detected"
    fi

    # Check Docker
    if [[ "${SKIP_DOCKER_CHECK}" != "true" ]]; then
        log_debug "Checking Docker installation..."
        if ! command -v docker &> /dev/null; then
            log_error "Docker is not installed or not in PATH"
            requirements_met=false
        else
            local docker_version
            docker_version=$(docker --version | sed 's/Docker version \([0-9.]*\).*/\1/')
            log_info "Docker version: $docker_version"

            if ! version_compare "$docker_version" "$MIN_DOCKER_VERSION"; then
                log_error "Docker version $docker_version is below minimum required $MIN_DOCKER_VERSION"
                requirements_met=false
            else
                log_success "Docker version requirement met"
            fi

            # Check Docker Compose
            log_debug "Checking Docker Compose..."
            if docker compose version &> /dev/null; then
                local compose_version
                compose_version=$(docker compose version --short)
                log_info "Docker Compose version: $compose_version"

                if ! version_compare "$compose_version" "$MIN_COMPOSE_VERSION"; then
                    log_error "Docker Compose version $compose_version is below minimum required $MIN_COMPOSE_VERSION"
                    requirements_met=false
                else
                    log_success "Docker Compose version requirement met"
                fi
            else
                log_error "Docker Compose V2 is not available"
                requirements_met=false
            fi

            # Check Docker daemon
            log_debug "Checking Docker daemon..."
            if ! docker info &> /dev/null; then
                log_error "Docker daemon is not running"
                requirements_met=false
            else
                log_success "Docker daemon is running"
            fi
        fi
    else
        log_warning "Docker validation skipped"
    fi

    # Check RAM
    log_debug "Checking available RAM..."
    local ram_gb=0
    case "$os_type" in
        "linux")
            if command -v free &> /dev/null; then
                ram_gb=$(($(free -g | awk '/^Mem:/{print $2}')))
            fi
            ;;
        "macos")
            if command -v sysctl &> /dev/null; then
                ram_gb=$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))
            fi
            ;;
        "windows")
            # Try multiple methods for Windows/WSL
            if command -v wmic &> /dev/null; then
                ram_gb=$(($(wmic computersystem get TotalPhysicalMemory /value | grep = | cut -d= -f2 | tr -d '\r') / 1024 / 1024 / 1024))
            elif [[ -f /proc/meminfo ]]; then
                ram_gb=$(($(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024 / 1024))
            fi
            ;;
    esac

    if [[ $ram_gb -gt 0 ]]; then
        log_info "Available RAM: ${ram_gb}GB"
        if [[ $ram_gb -lt $MIN_RAM_GB ]]; then
            log_error "Available RAM (${ram_gb}GB) is below minimum required (${MIN_RAM_GB}GB)"
            requirements_met=false
        elif [[ $ram_gb -lt $RECOMMENDED_RAM_GB ]]; then
            log_warning "Available RAM (${ram_gb}GB) is below recommended (${RECOMMENDED_RAM_GB}GB)"
        else
            log_success "RAM requirement met"
        fi
    else
        log_warning "Could not determine available RAM"
    fi

    # Check disk space
    log_debug "Checking available disk space..."
    local disk_gb
    if command -v df &> /dev/null; then
        disk_gb=$(df -BG "${PROJECT_ROOT}" | awk 'NR==2 {print $4}' | sed 's/G//')
        log_info "Available disk space: ${disk_gb}GB"

        if [[ $disk_gb -lt $MIN_DISK_GB ]]; then
            log_error "Available disk space (${disk_gb}GB) is below minimum required (${MIN_DISK_GB}GB)"
            requirements_met=false
        else
            log_success "Disk space requirement met"
        fi
    else
        log_warning "Could not determine available disk space"
    fi

    # Check required commands
    log_debug "Checking required commands..."
    local missing_commands=()
    for cmd in git curl; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_commands+=("$cmd")
        fi
    done

    if [[ ${#missing_commands[@]} -gt 0 ]]; then
        log_error "Missing required commands: ${missing_commands[*]}"
        requirements_met=false
    else
        log_success "All required commands available"
    fi

    if [[ "$requirements_met" == "true" ]]; then
        log_success "All system requirements met!"
        return 0
    else
        log_error "System requirements validation failed"
        if [[ "${FORCE_SETUP}" != "true" ]]; then
            log_error "Use --force to proceed anyway"
            return 1
        else
            log_warning "Proceeding with setup despite failed requirements (--force specified)"
            return 0
        fi
    fi
}

# Setup environment configuration
setup_environment() {
    log_step "Setting up local development environment..."

    local env_file="${CRACKSEG_DOCKER_DIR}/.env.local"

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would create environment file: $env_file"
        return 0
    fi

    log_info "Creating local environment configuration..."

    # Create .env.local based on template
    if [[ -f "${CRACKSEG_DOCKER_DIR}/env.local.template" ]]; then
        cp "${CRACKSEG_DOCKER_DIR}/env.local.template" "$env_file"
        log_success "Environment file created from template"
    else
        # Create basic environment file
        cat > "$env_file" << EOF
# CrackSeg Local Development Environment
# Generated by setup-local-dev.sh

# Development mode
NODE_ENV=development
ENVIRONMENT=local

# Docker configuration
COMPOSE_PROJECT_NAME=crackseg-local
COMPOSE_FILE=docker-compose.test.yml

# Test configuration
TEST_BROWSER=chrome
TEST_PARALLEL_WORKERS=auto
TEST_TIMEOUT=300
TEST_HEADLESS=true

# Feature flags
COVERAGE_ENABLED=true
HTML_REPORT_ENABLED=true
JSON_REPORT_ENABLED=true
SCREENSHOT_ON_FAILURE=true
VIDEO_RECORDING_ENABLED=false
MONITORING_ENABLED=false

# Performance tuning
SELENIUM_MEMORY=2g
STREAMLIT_MEMORY=1g
TEST_RUNNER_MEMORY=1g

# Network configuration
FRONTEND_NETWORK=crackseg-frontend-network
BACKEND_NETWORK=crackseg-backend-network
MANAGEMENT_NETWORK=crackseg-management-network

# Paths
PROJECT_ROOT=${PROJECT_ROOT}
CRACKSEG_DOCKER_DIR=${CRACKSEG_DOCKER_DIR}
ARTIFACTS_DIR=${PROJECT_ROOT}/test-artifacts
RESULTS_DIR=${PROJECT_ROOT}/test-results
VIDEOS_DIR=${PROJECT_ROOT}/selenium-videos

EOF
        log_success "Basic environment file created"
    fi

    # Make environment file read-only to prevent accidental modification
    chmod 644 "$env_file"
    log_info "Environment file permissions set to 644"
}

# Setup Docker override configuration
setup_docker_override() {
    log_step "Setting up Docker Compose overrides for local development..."

    local override_file="${CRACKSEG_DOCKER_DIR}/docker-compose.override.yml"

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would create Docker override file: $override_file"
        return 0
    fi

    log_info "Creating docker-compose.override.yml for development..."

    cat > "$override_file" << 'EOF'
# CrackSeg Local Development Docker Compose Override
# This file provides development-specific configurations

version: '3.8'

services:
  streamlit-app:
    volumes:
      # Mount source code for live reload
      - ../../src:/app/src:ro
      - ../../scripts:/app/scripts:ro
      - ../../configs:/app/configs:ro
    environment:
      - STREAMLIT_SERVER_RUNONTERMINATION=true
      - STREAMLIT_BROWSER_GATHERUSAGESTATS=false
      - STREAMLIT_SERVER_HEADLESS=false
    ports:
      - "8501:8501"  # Expose Streamlit for direct access

  test-runner:
    volumes:
      # Mount test directory for live updates
      - ../../tests:/app/tests:ro
      # Mount source for coverage
      - ../../src:/app/src:ro
    environment:
      - PYTEST_CURRENT_TEST=""
      - PYTHONPATH=/app/src:/app/tests
    # Override default command for development
    command: ["tail", "-f", "/dev/null"]  # Keep container running

  selenium-hub:
    environment:
      - SE_OPTS="--log-level INFO"
    ports:
      - "4444:4444"  # Expose grid console

  chrome-node:
    environment:
      - SE_OPTS="--log-level INFO"
      - SE_NODE_MAX_INSTANCES=2
      - SE_NODE_MAX_SESSIONS=2
    shm_size: 2gb

  firefox-node:
    environment:
      - SE_OPTS="--log-level INFO"
      - SE_NODE_MAX_INSTANCES=2
      - SE_NODE_MAX_SESSIONS=2
    shm_size: 2gb

  edge-node:
    environment:
      - SE_OPTS="--log-level INFO"
      - SE_NODE_MAX_INSTANCES=2
      - SE_NODE_MAX_SESSIONS=2
    shm_size: 2gb

# Development networks with external access
networks:
  crackseg-frontend-network:
    external: false
  crackseg-backend-network:
    external: false
  crackseg-management-network:
    external: false

# Development volumes
volumes:
  crackseg-test-artifacts:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${PWD}/../../test-artifacts
  crackseg-test-results:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${PWD}/../../test-results
EOF

    log_success "Docker Compose override file created"
}

# Setup directories
setup_directories() {
    log_step "Setting up project directories..."

    local directories=(
        "${PROJECT_ROOT}/test-artifacts"
        "${PROJECT_ROOT}/test-results"
        "${PROJECT_ROOT}/test-results/reports"
        "${PROJECT_ROOT}/test-results/coverage"
        "${PROJECT_ROOT}/test-results/screenshots"
        "${PROJECT_ROOT}/test-results/logs"
        "${PROJECT_ROOT}/selenium-videos"
        "${PROJECT_ROOT}/outputs"
    )

    for dir in "${directories[@]}"; do
        if [[ "${DRY_RUN}" == "true" ]]; then
            log_info "[DRY RUN] Would create directory: $dir"
        else
            if [[ ! -d "$dir" ]]; then
                mkdir -p "$dir"
                log_info "Created directory: $dir"
            else
                log_debug "Directory already exists: $dir"
            fi
        fi
    done

    # Create .gitkeep files for empty directories
    if [[ "${DRY_RUN}" != "true" ]]; then
        for dir in "${directories[@]}"; do
            if [[ ! -f "$dir/.gitkeep" ]] && [[ -z "$(ls -A "$dir" 2>/dev/null)" ]]; then
                touch "$dir/.gitkeep"
                log_debug "Created .gitkeep in: $dir"
            fi
        done
    fi

    log_success "Project directories setup complete"
}

# Validate Docker infrastructure
validate_docker_infrastructure() {
    log_step "Validating Docker infrastructure..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would validate Docker infrastructure"
        return 0
    fi

    # Check if docker-compose.test.yml exists
    local compose_file="${CRACKSEG_DOCKER_DIR}/docker-compose.test.yml"
    if [[ ! -f "$compose_file" ]]; then
        log_error "Docker Compose file not found: $compose_file"
        return 1
    fi

    log_info "Docker Compose file found"

    # Validate compose file
    log_debug "Validating Docker Compose configuration..."
    cd "${CRACKSEG_DOCKER_DIR}"

    if docker compose -f docker-compose.test.yml config > /dev/null 2>&1; then
        log_success "Docker Compose configuration is valid"
    else
        log_error "Docker Compose configuration validation failed"
        if [[ "${VERBOSE}" == "true" ]]; then
            docker compose -f docker-compose.test.yml config
        fi
        return 1
    fi

    # Pull required images (without starting services)
    log_info "Checking Docker images availability..."
    local images=(
        "selenium/hub:4.27.0"
        "selenium/node-chrome:4.27.0"
        "selenium/node-firefox:4.27.0"
        "selenium/node-edge:4.27.0"
    )

    for image in "${images[@]}"; do
        log_debug "Checking image: $image"
        if ! docker image inspect "$image" > /dev/null 2>&1; then
            log_info "Pulling image: $image"
            if ! docker pull "$image"; then
                log_warning "Failed to pull image: $image (will be pulled when needed)"
            fi
        else
            log_debug "Image available: $image"
        fi
    done

    log_success "Docker infrastructure validation complete"
}

# Setup development scripts
setup_dev_scripts() {
    log_step "Setting up development convenience scripts..."

    local scripts_dir="${PROJECT_ROOT}/scripts/dev"

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would create development scripts in: $scripts_dir"
        return 0
    fi

    mkdir -p "$scripts_dir"

    # Create quick start script
    cat > "${scripts_dir}/start-testing.sh" << 'EOF'
#!/bin/bash
# Quick start script for CrackSeg Docker testing

set -euo pipefail

cd "$(dirname "$0")/../../tests/docker"

echo "🚀 Starting CrackSeg Docker testing environment..."

# Start services
./scripts/docker-stack-manager.sh start

echo "✅ Environment ready!"
echo ""
echo "📊 Service URLs:"
echo "  - Streamlit App: http://localhost:8501"
echo "  - Selenium Grid: http://localhost:4444"
echo ""
echo "🧪 Run tests:"
echo "  ./scripts/run-test-runner.sh run"
echo ""
echo "🔍 Monitor:"
echo "  ./scripts/system-monitor.sh dashboard"
EOF

    # Create stop script
    cat > "${scripts_dir}/stop-testing.sh" << 'EOF'
#!/bin/bash
# Stop script for CrackSeg Docker testing

set -euo pipefail

cd "$(dirname "$0")/../../tests/docker"

echo "🛑 Stopping CrackSeg Docker testing environment..."

./scripts/docker-stack-manager.sh stop

echo "✅ Environment stopped!"
EOF

    # Create test script
    cat > "${scripts_dir}/run-tests.sh" << 'EOF'
#!/bin/bash
# Test execution script for CrackSeg

set -euo pipefail

cd "$(dirname "$0")/../../tests/docker"

echo "🧪 Running CrackSeg tests..."

# Run tests with coverage and reports
./scripts/run-test-runner.sh run --coverage --html-report

echo "✅ Tests completed!"
echo ""
echo "📊 View results:"
echo "  - HTML Report: test-results/reports/report.html"
echo "  - Coverage: test-results/coverage/html/index.html"
EOF

    # Make scripts executable
    chmod +x "${scripts_dir}"/*.sh

    log_success "Development convenience scripts created in: $scripts_dir"
}

# Main setup function
main_setup() {
    log_step "Starting CrackSeg local development setup..."

    # Create banner
    echo -e "${WHITE}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║               CrackSeg Local Development Setup                 ║"
    echo "║            Docker Testing Infrastructure v1.0.0               ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    # Validate system requirements
    if ! validate_system_requirements; then
        exit 1
    fi

    # If validate-only mode, exit after validation
    if [[ "${VALIDATE_ONLY}" == "true" ]]; then
        log_success "System validation completed successfully!"
        exit 0
    fi

    # Setup components
    setup_environment
    setup_docker_override
    setup_directories
    validate_docker_infrastructure
    setup_dev_scripts

    # Final success message
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                   Setup Completed Successfully!                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    echo -e "${WHITE}Next Steps:${NC}"
    echo "1. Start the testing environment:"
    echo "   ${CYAN}./scripts/dev/start-testing.sh${NC}"
    echo ""
    echo "2. Run tests:"
    echo "   ${CYAN}./scripts/dev/run-tests.sh${NC}"
    echo ""
    echo "3. Access services:"
    echo "   - Streamlit: ${BLUE}http://localhost:8501${NC}"
    echo "   - Selenium Grid: ${BLUE}http://localhost:4444${NC}"
    echo ""
    echo "4. Monitor system:"
    echo "   ${CYAN}cd tests/docker && ./scripts/system-monitor.sh dashboard${NC}"
    echo ""
    echo -e "${YELLOW}📚 Documentation:${NC}"
    echo "   - Master Guide: tests/docker/README-DOCKER-TESTING.md"
    echo "   - Usage Guide: tests/docker/README-USAGE.md"
    echo "   - Troubleshooting: tests/docker/README-TROUBLESHOOTING.md"
}

# Main execution
main() {
    # Change to script directory
    cd "${CRACKSEG_DOCKER_DIR}"

    # Parse arguments
    parse_args "$@"

    # Run main setup
    main_setup
}

# Execute main function with all arguments
main "$@"