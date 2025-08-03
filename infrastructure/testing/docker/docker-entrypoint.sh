#!/bin/bash
# =============================================================================
# CrackSeg Docker Entrypoint Script
# =============================================================================
# Purpose: Initialize test environment, run health checks, and manage
#          container lifecycle for E2E testing
#
# Usage: Called automatically by Docker container
# Environment: Streamlit GUI testing with Selenium automation
# =============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# =============================================================================
# Configuration & Environment Variables
# =============================================================================
export STREAMLIT_SERVER_PORT=${STREAMLIT_PORT:-8501}
export STREAMLIT_SERVER_ADDRESS=${STREAMLIT_HOST:-"0.0.0.0"}
export TEST_MODE=${TEST_MODE:-"e2e"}
export BROWSER=${BROWSER:-"chrome"}
export HEADLESS=${HEADLESS:-"true"}
export PARALLEL_WORKERS=${PARALLEL_WORKERS:-"auto"}
export TEST_TIMEOUT=${TEST_TIMEOUT:-"300"}

# Directories
export TEST_RESULTS_DIR="/app/test-results"
export TEST_DATA_DIR="/app/test-data"
export LOG_DIR="$TEST_RESULTS_DIR/logs"
export SCREENSHOT_DIR="$TEST_RESULTS_DIR/screenshots"

# =============================================================================
# Logging Functions
# =============================================================================
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ENTRYPOINT] $1" | tee -a "$LOG_DIR/container.log"
}

error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] $1" >&2 | tee -a "$LOG_DIR/container.log"
}

# =============================================================================
# Pre-flight Checks
# =============================================================================
preflight_checks() {
    log "Starting pre-flight checks..."

    # Check required directories exist
    for dir in "$TEST_RESULTS_DIR" "$TEST_DATA_DIR" "$LOG_DIR" "$SCREENSHOT_DIR"; do
        if [[ ! -d "$dir" ]]; then
            log "Creating directory: $dir"
            mkdir -p "$dir"
        fi
    done

    # Check Python environment
    if ! command -v python &> /dev/null; then
        error "Python not found in container"
        exit 1
    fi

    # Check essential Python packages
    local required_packages=("streamlit" "pytest" "selenium")
    for package in "${required_packages[@]}"; do
        if ! python -c "import $package" &> /dev/null; then
            error "Required package '$package' not found"
            exit 1
        fi
    done

    # Check if running as correct user
    if [[ "$(id -u)" == "0" ]] && [[ "$ALLOW_ROOT" != "true" ]]; then
        error "Container should not run as root in production"
        exit 1
    fi

    log "Pre-flight checks completed successfully"
}

# =============================================================================
# Health Check Functions
# =============================================================================
wait_for_streamlit() {
    local max_attempts=30
    local attempt=1

    log "Waiting for Streamlit to be ready..."

    while [[ $attempt -le $max_attempts ]]; do
        if curl -f "http://localhost:$STREAMLIT_SERVER_PORT/_stcore/health" &> /dev/null; then
            log "Streamlit is ready (attempt $attempt/$max_attempts)"
            return 0
        fi

        log "Streamlit not ready yet (attempt $attempt/$max_attempts), waiting 2 seconds..."
        sleep 2
        ((attempt++))
    done

    error "Streamlit failed to start within $((max_attempts * 2)) seconds"
    return 1
}

check_selenium_grid() {
    if [[ -n "${SELENIUM_HUB_HOST:-}" ]]; then
        log "Checking Selenium Grid connectivity..."

        local selenium_url="http://${SELENIUM_HUB_HOST}:${SELENIUM_HUB_PORT:-4444}/wd/hub/status"
        if curl -f "$selenium_url" &> /dev/null; then
            log "Selenium Grid is accessible"
        else
            error "Cannot connect to Selenium Grid at $selenium_url"
            return 1
        fi
    else
        log "No external Selenium Grid configured, using local WebDriver"
    fi
}

# =============================================================================
# Application Startup
# =============================================================================
start_streamlit() {
    log "Starting Streamlit application..."

    # Set Streamlit configuration
    export STREAMLIT_SERVER_HEADLESS=true
    export STREAMLIT_SERVER_ENABLE_CORS=false
    export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
    export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

    # Start Streamlit in background
    streamlit run scripts/gui/app.py \
        --server.port="$STREAMLIT_SERVER_PORT" \
        --server.address="$STREAMLIT_SERVER_ADDRESS" \
        --server.headless=true \
        --browser.gatherUsageStats=false \
        > "$LOG_DIR/streamlit.log" 2>&1 &

    local streamlit_pid=$!
    echo $streamlit_pid > /tmp/streamlit.pid

    log "Streamlit started with PID $streamlit_pid"

    # Wait for Streamlit to be ready
    if ! wait_for_streamlit; then
        error "Failed to start Streamlit application"
        exit 1
    fi
}

# =============================================================================
# Test Execution
# =============================================================================
run_tests() {
    local test_type="$1"

    log "Starting $test_type test execution..."

    # Construct pytest command
    local pytest_cmd=(
        "python" "-m" "pytest"
        "tests/e2e/"
        "--verbose"
        "--tb=short"
        "--html=$TEST_RESULTS_DIR/report.html"
        "--self-contained-html"
        "--cov=src"
        "--cov-report=html:$TEST_RESULTS_DIR/coverage"
        "--cov-report=json:$TEST_RESULTS_DIR/coverage.json"
        "--json-report"
        "--json-report-file=$TEST_RESULTS_DIR/test-report.json"
    )

    # Add parallel execution if requested
    if [[ "$PARALLEL_WORKERS" != "1" ]]; then
        pytest_cmd+=("-n" "$PARALLEL_WORKERS")
    fi

    # Add browser configuration
    pytest_cmd+=("--browser=$BROWSER")
    if [[ "$HEADLESS" == "true" ]]; then
        pytest_cmd+=("--headless")
    fi

    # Add timeout
    pytest_cmd+=("--timeout=$TEST_TIMEOUT")

    # Execute tests
    log "Running command: ${pytest_cmd[*]}"
    "${pytest_cmd[@]}" || {
        error "Test execution failed"
        return 1
    }

    log "Test execution completed successfully"
}

# =============================================================================
# Cleanup Functions
# =============================================================================
cleanup() {
    log "Starting cleanup process..."

    # Kill Streamlit process if running
    if [[ -f /tmp/streamlit.pid ]]; then
        local streamlit_pid=$(cat /tmp/streamlit.pid)
        if kill -0 "$streamlit_pid" 2>/dev/null; then
            log "Stopping Streamlit (PID $streamlit_pid)"
            kill "$streamlit_pid"
            wait "$streamlit_pid" 2>/dev/null || true
        fi
        rm -f /tmp/streamlit.pid
    fi

    # Archive test results with timestamp
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local archive_name="test-results-$timestamp.tar.gz"

    if [[ -d "$TEST_RESULTS_DIR" ]]; then
        log "Archiving test results to $archive_name"
        tar -czf "/app/$archive_name" -C "$TEST_RESULTS_DIR" .
    fi

    log "Cleanup completed"
}

# Set up signal handlers for graceful shutdown
trap cleanup EXIT
trap 'error "Received SIGTERM, shutting down..."; cleanup; exit 0' TERM
trap 'error "Received SIGINT, shutting down..."; cleanup; exit 0' INT

# =============================================================================
# Main Execution Flow
# =============================================================================
main() {
    log "CrackSeg Docker container starting..."
    log "Configuration: TEST_MODE=$TEST_MODE, BROWSER=$BROWSER, HEADLESS=$HEADLESS"

    # Run pre-flight checks
    preflight_checks

    # Start Streamlit application
    start_streamlit

    # Check external dependencies
    check_selenium_grid

    # Execute tests based on mode
    case "$TEST_MODE" in
        "e2e")
            run_tests "end-to-end"
            ;;
        "integration")
            run_tests "integration"
            ;;
        "serve")
            log "Serving mode: keeping container alive for manual testing"
            # Keep container running for manual testing
            while true; do
                sleep 30
                if ! curl -f "http://localhost:$STREAMLIT_SERVER_PORT/_stcore/health" &> /dev/null; then
                    error "Streamlit health check failed, restarting..."
                    start_streamlit
                fi
            done
            ;;
        *)
            error "Unknown test mode: $TEST_MODE"
            exit 1
            ;;
    esac

    log "Container execution completed successfully"
}

# Execute main function
main "$@"