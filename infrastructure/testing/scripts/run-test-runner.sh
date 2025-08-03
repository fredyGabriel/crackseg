#!/bin/bash
# =============================================================================
# CrackSeg Test Runner Management Script
# =============================================================================
# Purpose: Manage the specialized test runner container (Subtask 13.4)
# Features: Multi-browser support, artifact management, parallel execution
# Usage: ./run-test-runner.sh [options]
# =============================================================================

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DOCKER_DIR="$PROJECT_ROOT/infrastructure/testing"

# Default configuration
DEFAULT_BROWSER="chrome"
DEFAULT_PARALLEL_WORKERS="auto"
DEFAULT_TEST_TIMEOUT="300"
DEFAULT_HEADLESS="true"
DEFAULT_PROFILES=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

print_banner() {
    echo -e "${CYAN}"
    echo "============================================================================="
    echo "  CrackSeg - Specialized Test Runner Management (Subtask 13.4)"
    echo "============================================================================="
    echo -e "${NC}"
}

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
CrackSeg Test Runner Management Script

USAGE:
    $0 [OPTIONS] COMMAND

COMMANDS:
    build             Build the specialized test runner image
    run               Run the test runner container
    clean             Clean up test artifacts and containers
    logs              Show test runner logs
    status            Show container status
    artifacts         Manage test artifacts (basic)
    collect-artifacts Collect artifacts after test execution (enhanced)
    cleanup-artifacts Clean up old artifacts with retention policies
    archive-artifacts Archive artifacts for long-term storage
    help              Show this help message

OPTIONS:
    --browser BROWSER           Browser to use (chrome, firefox, edge) [default: $DEFAULT_BROWSER]
    --parallel-workers NUM      Number of parallel workers [default: $DEFAULT_PARALLEL_WORKERS]
    --timeout SECONDS          Test timeout in seconds [default: $DEFAULT_TEST_TIMEOUT]
    --headless BOOL            Run in headless mode [default: $DEFAULT_HEADLESS]
    --profiles PROFILES        Additional docker-compose profiles [default: none]
    --debug                    Enable debug mode
    --coverage                 Enable coverage reporting
    --html-report              Generate HTML test report
    --json-report              Generate JSON test report
    --screenshots              Enable screenshot capture on failure
    --videos                   Enable video recording (requires --profiles recording)
    --monitoring               Enable monitoring services (requires --profiles monitoring)
    --env-file FILE            Load environment variables from file

EXAMPLES:
    $0 build
    $0 run --browser chrome --parallel-workers 4
    $0 run --browser firefox,chrome --headless true --coverage
    $0 run --profiles "recording,monitoring" --videos --monitoring
    $0 artifacts --collect --archive
    $0 clean --all

ENVIRONMENT VARIABLES:
    TEST_BROWSER              Browser selection
    TEST_PARALLEL_WORKERS     Parallel execution configuration
    TEST_TIMEOUT              Test execution timeout
    TEST_HEADLESS             Headless mode setting
    TEST_DEBUG                Debug mode setting
    COVERAGE_ENABLED          Coverage reporting
    HTML_REPORT_ENABLED       HTML report generation
    JSON_REPORT_ENABLED       JSON report generation

EOF
}

# =============================================================================
# Environment Setup
# =============================================================================

setup_environment() {
    local env_file="${ENV_FILE:-}"

    # Load environment file if specified
    if [[ -n "$env_file" && -f "$env_file" ]]; then
        print_status "Loading environment from: $env_file"
        source "$env_file"
    fi

    # Set environment variables for docker-compose
    export TEST_BROWSER="${TEST_BROWSER:-$BROWSER}"
    export TEST_PARALLEL_WORKERS="${TEST_PARALLEL_WORKERS:-$PARALLEL_WORKERS}"
    export TEST_TIMEOUT="${TEST_TIMEOUT:-$TIMEOUT}"
    export TEST_HEADLESS="${TEST_HEADLESS:-$HEADLESS}"
    export TEST_DEBUG="${TEST_DEBUG:-$DEBUG}"

    # Test artifact paths (Enhanced - Subtask 13.5)
    export TEST_RESULTS_PATH="${TEST_RESULTS_PATH:-$PROJECT_ROOT/test-results}"
    export TEST_DATA_PATH="${TEST_DATA_PATH:-$PROJECT_ROOT/test-data}"
    export TEST_ARTIFACTS_PATH="${TEST_ARTIFACTS_PATH:-$PROJECT_ROOT/test-artifacts}"
    export SELENIUM_VIDEOS_PATH="${SELENIUM_VIDEOS_PATH:-$PROJECT_ROOT/selenium-videos}"

    # NEW: Archive and temporary paths (Subtask 13.5)
    export ARCHIVE_PATH="${ARCHIVE_PATH:-$PROJECT_ROOT/archived-artifacts}"
    export TEMP_ARTIFACTS_PATH="${TEMP_ARTIFACTS_PATH:-$PROJECT_ROOT/temp-artifacts}"

    # Feature flags
    export COVERAGE_ENABLED="${COVERAGE_ENABLED:-$COVERAGE}"
    export HTML_REPORT_ENABLED="${HTML_REPORT_ENABLED:-$HTML_REPORT}"
    export JSON_REPORT_ENABLED="${JSON_REPORT_ENABLED:-$JSON_REPORT}"
    export SCREENSHOT_ON_FAILURE="${SCREENSHOT_ON_FAILURE:-$SCREENSHOTS}"

    # NEW: Artifact management flags (Subtask 13.5)
    export ARTIFACT_COLLECTION_ENABLED="${ARTIFACT_COLLECTION_ENABLED:-true}"
    export ARTIFACT_CLEANUP_ENABLED="${ARTIFACT_CLEANUP_ENABLED:-false}"
    export ARTIFACT_ARCHIVE_ENABLED="${ARTIFACT_ARCHIVE_ENABLED:-false}"

    # Create required directories
    mkdir -p "$TEST_RESULTS_PATH" "$TEST_DATA_PATH" "$TEST_ARTIFACTS_PATH" "$SELENIUM_VIDEOS_PATH"
    mkdir -p "$ARCHIVE_PATH" "$TEMP_ARTIFACTS_PATH"

    print_status "Environment configured:"
    print_status "  Browser: $TEST_BROWSER"
    print_status "  Workers: $TEST_PARALLEL_WORKERS"
    print_status "  Timeout: $TEST_TIMEOUT"
    print_status "  Headless: $TEST_HEADLESS"
    print_status "  Debug: $TEST_DEBUG"
}

# =============================================================================
# Docker Management
# =============================================================================

build_test_runner() {
    print_status "Building specialized test runner image..."

    cd "$DOCKER_DIR"

    docker-compose -f docker-compose.test.yml build test-runner

    if [[ $? -eq 0 ]]; then
        print_success "Test runner image built successfully"
    else
        print_error "Failed to build test runner image"
        exit 1
    fi
}

run_test_runner() {
    print_status "Starting test runner environment..."

    cd "$DOCKER_DIR"

    # Build profiles argument
    local profiles_arg=""
    if [[ -n "$PROFILES" ]]; then
        profiles_arg="--profile $PROFILES"
    fi

    # Start dependencies first
    print_status "Starting Selenium Grid and Streamlit app..."
    docker-compose -f docker-compose.test.yml $profiles_arg up -d streamlit-app selenium-hub chrome-node firefox-node

    # Wait for services to be healthy
    print_status "Waiting for services to be ready..."
    docker-compose -f docker-compose.test.yml $profiles_arg up --wait streamlit-app selenium-hub chrome-node firefox-node

    # Run the test runner
    print_status "Executing test runner..."
    docker-compose -f docker-compose.test.yml $profiles_arg run --rm test-runner

    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        print_success "Test execution completed successfully"

        # Post-execution artifact management (Subtask 13.5)
        post_execution_artifact_management
    else
        print_warning "Test execution completed with exit code: $exit_code"
    fi

    return $exit_code
}

show_logs() {
    print_status "Showing test runner logs..."

    cd "$DOCKER_DIR"
    docker-compose -f docker-compose.test.yml logs test-runner
}

show_status() {
    print_status "Container status:"

    cd "$DOCKER_DIR"
    docker-compose -f docker-compose.test.yml ps
}

# =============================================================================
# Artifact Management
# =============================================================================

manage_artifacts() {
    local action="${1:-list}"

    case "$action" in
        list)
            print_status "Test artifacts:"
            find "$TEST_RESULTS_PATH" "$TEST_ARTIFACTS_PATH" -type f 2>/dev/null | head -20
            ;;
        collect)
            print_status "Collecting test artifacts..."
            # Copy artifacts from container volumes
            docker run --rm -v crackseg-test-results:/source -v "$TEST_RESULTS_PATH":/dest alpine cp -r /source/. /dest/
            docker run --rm -v crackseg-test-artifacts:/source -v "$TEST_ARTIFACTS_PATH":/dest alpine cp -r /source/. /dest/
            print_success "Artifacts collected"
            ;;
        archive)
            local timestamp=$(date +%Y%m%d_%H%M%S)
            local archive_name="test-artifacts-$timestamp.tar.gz"
            print_status "Creating archive: $archive_name"
            tar -czf "$PROJECT_ROOT/$archive_name" -C "$PROJECT_ROOT" test-results test-artifacts
            print_success "Archive created: $archive_name"
            ;;
        clean)
            print_status "Cleaning test artifacts..."
            rm -rf "$TEST_RESULTS_PATH"/* "$TEST_ARTIFACTS_PATH"/* "$SELENIUM_VIDEOS_PATH"/*
            print_success "Artifacts cleaned"
            ;;
        *)
            print_error "Unknown artifact action: $action"
            print_status "Available actions: list, collect, archive, clean"
            exit 1
            ;;
    esac
}

# =============================================================================
# Cleanup Functions
# =============================================================================

cleanup_containers() {
    local scope="${1:-test-runner}"

    cd "$DOCKER_DIR"

    case "$scope" in
        test-runner)
            print_status "Stopping test runner container..."
            docker-compose -f docker-compose.test.yml stop test-runner
            docker-compose -f docker-compose.test.yml rm -f test-runner
            ;;
        all)
            print_status "Stopping all test services..."
            docker-compose -f docker-compose.test.yml down
            ;;
        volumes)
            print_status "Removing test volumes..."
            docker-compose -f docker-compose.test.yml down -v
            ;;
        images)
            print_status "Removing test images..."
            docker rmi $(docker images | grep crackseg | awk '{print $3}') 2>/dev/null || true
            ;;
        *)
            print_error "Unknown cleanup scope: $scope"
            print_status "Available scopes: test-runner, all, volumes, images"
            exit 1
            ;;
    esac

    print_success "Cleanup completed: $scope"
}

# =============================================================================
# Enhanced Artifact Management Functions (Subtask 13.5)
# =============================================================================

post_execution_artifact_management() {
    print_status "Post-execution artifact management..."

    if [[ "$ARTIFACT_COLLECTION_ENABLED" == "true" ]]; then
        collect_artifacts_enhanced
    fi

    if [[ "$ARTIFACT_CLEANUP_ENABLED" == "true" ]]; then
        cleanup_artifacts_enhanced
    fi

    if [[ "$ARTIFACT_ARCHIVE_ENABLED" == "true" ]]; then
        archive_artifacts_enhanced
    fi
}

collect_artifacts_enhanced() {
    print_status "Enhanced artifact collection using artifact-manager.sh"

    local artifact_manager="$SCRIPT_DIR/artifact-manager.sh"
    if [[ -f "$artifact_manager" ]]; then
        "$artifact_manager" collect --compress --debug

        # Generate collection report
        "$artifact_manager" status > "$TEST_ARTIFACTS_PATH/last-collection-status.log"

        print_success "Enhanced artifact collection completed"
    else
        print_warning "artifact-manager.sh not found, falling back to basic collection"
        manage_artifacts collect
    fi
}

cleanup_artifacts_enhanced() {
    print_status "Enhanced artifact cleanup using artifact-manager.sh"

    local artifact_manager="$SCRIPT_DIR/artifact-manager.sh"
    if [[ -f "$artifact_manager" ]]; then
        # Clean artifacts older than 7 days, keep latest 5 collections
        "$artifact_manager" cleanup --older-than 7 --keep-latest 5 --force

        print_success "Enhanced artifact cleanup completed"
    else
        print_warning "artifact-manager.sh not found, falling back to basic cleanup"
        manage_artifacts clean
    fi
}

archive_artifacts_enhanced() {
    print_status "Enhanced artifact archiving using artifact-manager.sh"

    local artifact_manager="$SCRIPT_DIR/artifact-manager.sh"
    if [[ -f "$artifact_manager" ]]; then
        # Archive with compression and 30-day retention
        "$artifact_manager" archive --format tar.gz --retention-days 30

        print_success "Enhanced artifact archiving completed"
    else
        print_warning "artifact-manager.sh not found, falling back to basic archiving"
        manage_artifacts archive
    fi
}

# =============================================================================
# Main Script Logic
# =============================================================================

# Parse command line arguments
BROWSER="$DEFAULT_BROWSER"
PARALLEL_WORKERS="$DEFAULT_PARALLEL_WORKERS"
TIMEOUT="$DEFAULT_TEST_TIMEOUT"
HEADLESS="$DEFAULT_HEADLESS"
PROFILES="$DEFAULT_PROFILES"
DEBUG="false"
COVERAGE="true"
HTML_REPORT="true"
JSON_REPORT="true"
SCREENSHOTS="true"
VIDEOS="false"
MONITORING="false"
ENV_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --browser)
            BROWSER="$2"
            shift 2
            ;;
        --parallel-workers)
            PARALLEL_WORKERS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --headless)
            HEADLESS="$2"
            shift 2
            ;;
        --profiles)
            PROFILES="$2"
            shift 2
            ;;
        --debug)
            DEBUG="true"
            shift
            ;;
        --coverage)
            COVERAGE="true"
            shift
            ;;
        --html-report)
            HTML_REPORT="true"
            shift
            ;;
        --json-report)
            JSON_REPORT="true"
            shift
            ;;
        --screenshots)
            SCREENSHOTS="true"
            shift
            ;;
        --videos)
            VIDEOS="true"
            PROFILES="recording,$PROFILES"
            shift
            ;;
        --monitoring)
            MONITORING="true"
            PROFILES="monitoring,$PROFILES"
            shift
            ;;
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        -*)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            COMMAND="$1"
            shift
            break
            ;;
    esac
done

# Validate command
if [[ -z "${COMMAND:-}" ]]; then
    print_error "No command specified"
    show_help
    exit 1
fi

# Execute command
print_banner

case "$COMMAND" in
    build)
        setup_environment
        build_test_runner
        ;;
    run)
        setup_environment
        run_test_runner
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    artifacts)
        manage_artifacts "${1:-list}"
        ;;
    collect-artifacts)
        setup_environment
        collect_artifacts_enhanced
        ;;
    cleanup-artifacts)
        setup_environment
        cleanup_artifacts_enhanced
        ;;
    archive-artifacts)
        setup_environment
        archive_artifacts_enhanced
        ;;
    clean)
        cleanup_containers "${1:-test-runner}"
        ;;
    help)
        show_help
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

print_status "Operation completed: $COMMAND"