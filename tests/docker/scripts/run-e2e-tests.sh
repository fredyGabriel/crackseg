#!/bin/bash

# =============================================================================
# CrackSeg Docker Testing - E2E Test Execution Script
# =============================================================================
# Purpose: Execute E2E tests using Docker Compose infrastructure
# Usage: ./run-e2e-tests.sh [options]
# =============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$DOCKER_DIR")")"
COMPOSE_FILE="$DOCKER_DIR/docker-compose.test.yml"

# Default values
BROWSER="chrome"
HEADLESS=true
PARALLEL_WORKERS="auto"
TEST_PATTERN="tests/e2e/"
TIMEOUT=300
VERBOSE=false
RECORD_VIDEO=false
KEEP_CONTAINERS=false
COLLECT_ARTIFACTS=true
REPORT_FORMAT="html"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
CrackSeg E2E Test Execution Script

Usage: $0 [OPTIONS]

OPTIONS:
    -b, --browser BROWSER       Browser to use: chrome, firefox, both (default: chrome)
    -H, --headless              Run in headless mode (default: true)
    -g, --gui                   Run in GUI mode (sets headless to false)
    -p, --parallel WORKERS      Number of parallel workers: auto, 1, 2, 4 (default: auto)
    -t, --test-pattern PATTERN  Test pattern to run (default: tests/e2e/)
    -T, --timeout SECONDS       Timeout for test execution (default: 300)
    -v, --verbose               Enable verbose logging
    -r, --record                Enable video recording
    -k, --keep-containers       Keep containers running after tests
    -n, --no-artifacts          Skip artifact collection
    -f, --format FORMAT         Report format: html, json, junit (default: html)
    -h, --help                  Show this help message

EXAMPLES:
    $0                                          # Run all E2E tests with Chrome
    $0 -b firefox -g                           # Run with Firefox in GUI mode
    $0 -b both -p 2 -r                        # Run with both browsers, 2 workers, recording
    $0 -t "tests/e2e/test_login.py" -v        # Run specific test with verbose output
    $0 --browser chrome --parallel 1 --gui    # Single worker Chrome with GUI

BROWSERS:
    chrome      - Run tests only with Chrome
    firefox     - Run tests only with Firefox
    both        - Run tests with both Chrome and Firefox

PARALLEL EXECUTION:
    auto        - Automatically determine worker count
    1           - Single-threaded execution
    2, 4, etc.  - Specific number of workers

EOF
}

check_prerequisites() {
    log "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not available"
        exit 1
    fi

    # Check compose file exists
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    fi

    # Check if services are running
    cd "$PROJECT_ROOT"
    if ! docker compose -f "$COMPOSE_FILE" ps --filter "status=running" | grep -q "crackseg-selenium-hub"; then
        log_warning "Selenium hub is not running. Starting test environment..."
        "$SCRIPT_DIR/start-test-env.sh" -d
        sleep 10
    fi

    log_success "Prerequisites check passed"
}

prepare_test_environment() {
    log "Preparing test environment..."

    # Create artifacts directory
    local artifacts_dir="$PROJECT_ROOT/test-results"
    mkdir -p "$artifacts_dir"

    # Clean up previous test results if not keeping containers
    if [[ "$KEEP_CONTAINERS" == false ]]; then
        rm -rf "$artifacts_dir"/*
    fi

    log_success "Test environment prepared"
}

build_pytest_command() {
    local pytest_args=()

    # Basic pytest configuration
    pytest_args+=("python" "-m" "pytest")
    pytest_args+=("$TEST_PATTERN")
    pytest_args+=("--tb=short")

    # Verbose output
    if [[ "$VERBOSE" == true ]]; then
        pytest_args+=("--verbose" "-s")
    else
        pytest_args+=("--quiet")
    fi

    # Browser configuration
    pytest_args+=("--browser=$BROWSER")

    # Headless mode
    if [[ "$HEADLESS" == true ]]; then
        pytest_args+=("--headless")
    fi

    # Parallel execution
    if [[ "$PARALLEL_WORKERS" != "1" ]]; then
        pytest_args+=("-n" "$PARALLEL_WORKERS")
    fi

    # Timeout
    pytest_args+=("--timeout=$TIMEOUT")

    # Report generation
    case "$REPORT_FORMAT" in
        html)
            pytest_args+=("--html=/app/test-results/report.html")
            pytest_args+=("--self-contained-html")
            ;;
        json)
            pytest_args+=("--json-report")
            pytest_args+=("--json-report-file=/app/test-results/report.json")
            ;;
        junit)
            pytest_args+=("--junitxml=/app/test-results/junit.xml")
            ;;
    esac

    # Coverage
    pytest_args+=("--cov=src")
    pytest_args+=("--cov-report=html:/app/test-results/coverage/")
    pytest_args+=("--cov-report=term-missing")

    # Additional pytest options
    pytest_args+=("--strict-markers")
    pytest_args+=("--disable-warnings")

    echo "${pytest_args[*]}"
}

run_tests_single_browser() {
    local browser=$1
    log "Running E2E tests with $browser browser..."

    cd "$PROJECT_ROOT"

    # Build pytest command
    local pytest_command
    pytest_command=$(BROWSER="$browser" build_pytest_command)

    # Build docker compose run command
    local compose_args=()
    compose_args+=("-f" "$COMPOSE_FILE")

    local run_args=()
    run_args+=("run" "--rm")

    # Add video recording profile if enabled
    if [[ "$RECORD_VIDEO" == true ]]; then
        run_args+=("--profile" "recording")
    fi

    # Environment variables
    run_args+=("-e" "BROWSER=$browser")
    run_args+=("-e" "HEADLESS=$HEADLESS")
    run_args+=("-e" "PARALLEL_WORKERS=$PARALLEL_WORKERS")
    run_args+=("-e" "TEST_TIMEOUT=$TIMEOUT")
    run_args+=("-e" "VERBOSE=$VERBOSE")

    # Run the tests
    run_args+=("test-runner" "bash" "-c" "$pytest_command")

    log "Executing: docker compose ${compose_args[*]} ${run_args[*]}"

    # Execute the test command
    local exit_code=0
    docker compose "${compose_args[@]}" "${run_args[@]}" || exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        log_success "Tests passed with $browser browser"
    else
        log_error "Tests failed with $browser browser (exit code: $exit_code)"
    fi

    return $exit_code
}

run_tests() {
    local overall_exit_code=0

    case "$BROWSER" in
        chrome)
            run_tests_single_browser "chrome" || overall_exit_code=$?
            ;;
        firefox)
            run_tests_single_browser "firefox" || overall_exit_code=$?
            ;;
        both)
            log "Running tests with both Chrome and Firefox browsers..."
            run_tests_single_browser "chrome" || overall_exit_code=$?
            run_tests_single_browser "firefox" || overall_exit_code=$?
            ;;
        *)
            log_error "Invalid browser: $BROWSER"
            exit 1
            ;;
    esac

    return $overall_exit_code
}

collect_artifacts() {
    if [[ "$COLLECT_ARTIFACTS" == true ]]; then
        log "Collecting test artifacts..."

        local artifacts_dir="$PROJECT_ROOT/test-results"

        # Copy artifacts from Docker volumes
        cd "$PROJECT_ROOT"

        # Get container ID for artifacts volume
        local volume_name="crackseg-test-results"

        # Create temporary container to access volume
        docker run --rm -v "$volume_name:/source" -v "$artifacts_dir:/dest" \
            alpine:latest sh -c "cp -r /source/* /dest/ 2>/dev/null || true"

        # Collect videos if recording was enabled
        if [[ "$RECORD_VIDEO" == true ]]; then
            local video_volume="crackseg-selenium-videos"
            local video_dir="$artifacts_dir/videos"
            mkdir -p "$video_dir"

            docker run --rm -v "$video_volume:/source" -v "$video_dir:/dest" \
                alpine:latest sh -c "cp -r /source/* /dest/ 2>/dev/null || true"
        fi

        # Generate artifact summary
        local summary_file="$artifacts_dir/test-summary.txt"
        {
            echo "CrackSeg E2E Test Results Summary"
            echo "================================="
            echo "Execution Time: $(date)"
            echo "Browser(s): $BROWSER"
            echo "Headless Mode: $HEADLESS"
            echo "Parallel Workers: $PARALLEL_WORKERS"
            echo "Test Pattern: $TEST_PATTERN"
            echo "Video Recording: $RECORD_VIDEO"
            echo ""
            echo "Artifacts:"
            find "$artifacts_dir" -type f -name "*.html" -o -name "*.json" -o -name "*.xml" -o -name "*.mp4" | sort
        } > "$summary_file"

        log_success "Artifacts collected in: $artifacts_dir"

        # Show quick summary
        if [[ -f "$artifacts_dir/report.html" ]]; then
            log "📊 HTML Report: file://$artifacts_dir/report.html"
        fi

        if [[ -d "$artifacts_dir/coverage" ]]; then
            log "📈 Coverage Report: file://$artifacts_dir/coverage/index.html"
        fi
    fi
}

cleanup() {
    if [[ "$KEEP_CONTAINERS" == false ]]; then
        log "Cleaning up test containers..."
        cd "$PROJECT_ROOT"
        docker compose -f "$COMPOSE_FILE" down --remove-orphans &> /dev/null || true
        log_success "Cleanup completed"
    else
        log "Keeping containers running as requested"
    fi
}

cleanup_on_exit() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Test execution failed with exit code $exit_code"
    fi

    collect_artifacts

    if [[ "$KEEP_CONTAINERS" == false ]]; then
        cleanup
    fi

    exit $exit_code
}

# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -b|--browser)
                BROWSER="$2"
                shift 2
                ;;
            -H|--headless)
                HEADLESS=true
                shift
                ;;
            -g|--gui)
                HEADLESS=false
                shift
                ;;
            -p|--parallel)
                PARALLEL_WORKERS="$2"
                shift 2
                ;;
            -t|--test-pattern)
                TEST_PATTERN="$2"
                shift 2
                ;;
            -T|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -r|--record)
                RECORD_VIDEO=true
                shift
                ;;
            -k|--keep-containers)
                KEEP_CONTAINERS=true
                shift
                ;;
            -n|--no-artifacts)
                COLLECT_ARTIFACTS=false
                shift
                ;;
            -f|--format)
                REPORT_FORMAT="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Validate browser
    case "$BROWSER" in
        chrome|firefox|both)
            ;;
        *)
            log_error "Invalid browser: $BROWSER"
            show_help
            exit 1
            ;;
    esac

    # Validate report format
    case "$REPORT_FORMAT" in
        html|json|junit)
            ;;
        *)
            log_error "Invalid report format: $REPORT_FORMAT"
            show_help
            exit 1
            ;;
    esac

    # Set up exit trap
    trap cleanup_on_exit EXIT

    # Execute test sequence
    log "Starting CrackSeg E2E test execution..."
    log "Browser: $BROWSER"
    log "Headless: $HEADLESS"
    log "Parallel Workers: $PARALLEL_WORKERS"
    log "Test Pattern: $TEST_PATTERN"
    log "Video Recording: $RECORD_VIDEO"

    check_prerequisites
    prepare_test_environment

    local exit_code=0
    run_tests || exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        log_success "All E2E tests completed successfully!"
    else
        log_error "E2E tests failed"
    fi

    exit $exit_code
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi