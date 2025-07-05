#!/usr/bin/env bash
# =============================================================================
# CrackSeg E2E Test Orchestrator
# =============================================================================
# Purpose: Advanced test execution orchestration with parallel execution,
#          resource management, and comprehensive reporting
# Version: 1.0 (Subtask 13.8)
# Author: CrackSeg Project
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration and Setup
# =============================================================================

# Script directory and project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$DOCKER_DIR")")"
COMPOSE_FILE="$DOCKER_DIR/docker-compose.test.yml"

# Component script paths
STACK_MANAGER="$SCRIPT_DIR/docker-stack-manager.sh"
HEALTH_MANAGER="$SCRIPT_DIR/health-check-manager.sh"
ARTIFACT_MANAGER="$SCRIPT_DIR/artifact-manager.sh"
NETWORK_MANAGER="$SCRIPT_DIR/network-manager.sh"

# Add to the beginning after existing configuration
BROWSER_MANAGER="$SCRIPT_DIR/browser-manager.sh"
CAPABILITIES_FILE="$DOCKER_DIR/browser-capabilities.json"

# Default configuration
DEFAULT_BROWSERS="chrome"
DEFAULT_PARALLEL_WORKERS="auto"
DEFAULT_TEST_TIMEOUT=300
DEFAULT_RETRY_COUNT=2
DEFAULT_REPORT_FORMAT="html,json,junit"

# Test execution directories
TEST_RESULTS_DIR="$PROJECT_ROOT/test-results"
TEST_ARTIFACTS_DIR="$PROJECT_ROOT/test-artifacts"
TEST_REPORTS_DIR="$TEST_RESULTS_DIR/reports"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly MAGENTA='\033[0;35m'
readonly NC='\033[0m' # No Color

# =============================================================================
# Logging Functions
# =============================================================================

log() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] [E2E-ORCHESTRATOR]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_step() {
    echo -e "${MAGENTA}[STEP]${NC} $*"
}

print_banner() {
    echo -e "${CYAN}"
    echo "============================================================================="
    echo "  CrackSeg E2E Test Orchestrator - Advanced Test Execution System"
    echo "  Parallel Testing, Resource Management & Comprehensive Reporting"
    echo "============================================================================="
    echo -e "${NC}"
}

# =============================================================================
# Environment Setup and Validation
# =============================================================================

setup_test_environment() {
    log_step "Setting up test execution environment..."

    # Create test directories
    mkdir -p "$TEST_RESULTS_DIR" "$TEST_ARTIFACTS_DIR" "$TEST_REPORTS_DIR"
    mkdir -p "$TEST_RESULTS_DIR/screenshots" "$TEST_RESULTS_DIR/videos"
    mkdir -p "$TEST_RESULTS_DIR/logs" "$TEST_RESULTS_DIR/coverage"

    # Set environment variables for test execution
    export TEST_EXECUTION_ID="e2e_$(date +%Y%m%d_%H%M%S)"
    export TEST_START_TIME=$(date -Iseconds)
    export PYTEST_CURRENT_TEST_DIR="$TEST_RESULTS_DIR"
    export SELENIUM_SCREENSHOTS_DIR="$TEST_RESULTS_DIR/screenshots"
    export SELENIUM_VIDEOS_DIR="$TEST_RESULTS_DIR/videos"

    log_info "Test execution ID: $TEST_EXECUTION_ID"
    log_info "Test results directory: $TEST_RESULTS_DIR"
    log_success "Test environment setup completed"
}

validate_test_prerequisites() {
    log_step "Validating test execution prerequisites..."

    # Check component scripts
    local scripts=("$STACK_MANAGER" "$HEALTH_MANAGER" "$ARTIFACT_MANAGER")
    for script in "${scripts[@]}"; do
        if [[ ! -f "$script" ]]; then
            log_error "Required component script not found: $script"
            exit 1
        fi
        if [[ ! -x "$script" ]]; then
            chmod +x "$script"
        fi
    done

    # Check test files
    local test_dir="$PROJECT_ROOT/tests"
    if [[ ! -d "$test_dir" ]]; then
        log_error "Tests directory not found: $test_dir"
        exit 1
    fi

    # Check for E2E test files
    if ! find "$test_dir" -name "test_*.py" -type f | grep -q .; then
        log_warning "No test files found in $test_dir"
    fi

    log_success "Prerequisites validation completed"
}

# =============================================================================
# Test Execution Planning
# =============================================================================

calculate_optimal_workers() {
    local browser_count="${1:-1}"
    local requested_workers="${2:-auto}"

    if [[ "$requested_workers" == "auto" ]]; then
        # Calculate based on available CPU cores and browser count
        local cpu_cores
        cpu_cores=$(nproc 2>/dev/null || echo "2")
        local optimal_workers=$((cpu_cores / browser_count))

        # Ensure minimum of 1 worker and maximum of 8
        if [[ $optimal_workers -lt 1 ]]; then
            optimal_workers=1
        elif [[ $optimal_workers -gt 8 ]]; then
            optimal_workers=8
        fi

        echo "$optimal_workers"
    else
        echo "$requested_workers"
    fi
}

# =============================================================================
# Cross-Browser Support Integration (Subtask 13.10)
# =============================================================================

setup_dynamic_browsers() {
    local browser_matrix="${1:-smoke_test}"
    local force_recreate="${2:-false}"

    log_step "Setting up dynamic browser matrix: $browser_matrix"

    # Check if browser manager exists
    if [[ ! -f "$BROWSER_MANAGER" ]]; then
        log_error "Browser manager script not found: $BROWSER_MANAGER"
        return 1
    fi

    # Make executable if needed
    if [[ ! -x "$BROWSER_MANAGER" ]]; then
        chmod +x "$BROWSER_MANAGER"
    fi

    # Cleanup existing dynamic browsers if force recreate
    if [[ "$force_recreate" == "true" ]]; then
        log_info "Force recreate enabled, cleaning up existing browsers..."
        "$BROWSER_MANAGER" cleanup || log_warning "Failed to cleanup existing browsers"
    fi

    # Create browser matrix
    log_info "Creating browser matrix: $browser_matrix"
    if "$BROWSER_MANAGER" matrix "$browser_matrix" true; then
        log_success "Browser matrix created successfully"

        # Give browsers time to register with Selenium Hub
        log_info "Waiting for browser registration..."
        sleep 10

        # Verify browser registration
        verify_browser_matrix_registration

        return 0
    else
        log_error "Failed to create browser matrix: $browser_matrix"
        return 1
    fi
}

verify_browser_matrix_registration() {
    log_step "Verifying browser matrix registration with Selenium Hub..."

    local max_attempts=15
    local attempt=1
    local min_nodes=2  # Minimum expected nodes

    while [[ $attempt -le $max_attempts ]]; do
        # Check Selenium Hub status
        if curl -s "http://localhost:4444/grid/api/hub/status" >/dev/null 2>&1; then
            # Get registered nodes count
            local nodes_count
            nodes_count=$(curl -s "http://localhost:4444/grid/api/hub" | jq -r '.value.nodes | length' 2>/dev/null || echo "0")

            if [[ "$nodes_count" -ge "$min_nodes" ]]; then
                log_success "Browser matrix verified: $nodes_count nodes registered"

                # List registered browsers
                local browsers
                browsers=$(curl -s "http://localhost:4444/grid/api/hub" | jq -r '.value.nodes[].slots[].stereotype.browserName' 2>/dev/null | sort | uniq | tr '\n' ' ' || echo "unknown")
                log_info "Registered browsers: $browsers"

                return 0
            fi
        fi

        log_info "Attempt $attempt/$max_attempts - waiting for browser registration ($nodes_count/$min_nodes nodes)"
        sleep 3
        ((attempt++))
    done

    log_warning "Browser matrix verification incomplete after timeout"
    return 1
}

cleanup_dynamic_browsers() {
    log_step "Cleaning up dynamic browser containers..."

    if [[ -f "$BROWSER_MANAGER" ]]; then
        "$BROWSER_MANAGER" cleanup
        log_success "Dynamic browser cleanup completed"
    else
        log_warning "Browser manager not found, skipping cleanup"
    fi
}

list_browser_capabilities() {
    log_step "Available browser configurations:"

    if [[ -f "$BROWSER_MANAGER" ]]; then
        "$BROWSER_MANAGER" browsers
    else
        log_error "Browser manager not found"
        return 1
    fi
}

# Enhanced browser test execution with matrix support
execute_browser_tests() {
    local browser="$1"
    local workers="$2"
    local test_patterns="$3"
    local results_dir="$4"
    local browser_matrix="${5:-}"

    log_step "Executing tests for browser: $browser"

    # Setup browser matrix if specified
    if [[ -n "$browser_matrix" ]]; then
        if ! setup_dynamic_browsers "$browser_matrix" false; then
            log_error "Failed to setup browser matrix: $browser_matrix"
            return 1
        fi
    fi

    # Create browser-specific configuration
    local pytest_args=(
        "-v"
        "--tb=short"
        "--capture=no"
        "--browser=$browser"
        "--headless=true"
        "--selenium-grid-url=http://localhost:4444/wd/hub"
        "--workers=$workers"
        "--results-dir=$results_dir"
        "--artifacts-dir=$results_dir/artifacts"
    )

    # Add test patterns
    pytest_args+=($test_patterns)

    # Setup environment for this browser
    export BROWSER_NAME="$browser"
    export SELENIUM_GRID_URL="http://localhost:4444/wd/hub"
    export TEST_RESULTS_DIR="$results_dir"
    export PYTEST_CURRENT_TEST="$browser"

    # Execute tests
    log_info "Running pytest with browser: $browser, workers: $workers"
    log_info "Command: pytest ${pytest_args[*]}"

    cd "$PROJECT_ROOT"

    if timeout 1800 python -m pytest "${pytest_args[@]}"; then
        log_success "Tests completed successfully for browser: $browser"
        return 0
    else
        local exit_code=$?
        log_error "Tests failed for browser: $browser (exit code: $exit_code)"
        return 1
    fi
}

# Enhanced execution plan creation with browser matrix support
create_test_execution_plan() {
    local browsers="${1:-$DEFAULT_BROWSERS}"
    local workers="${2:-$DEFAULT_PARALLEL_WORKERS}"
    local test_patterns="${3:-tests/e2e/}"
    local browser_matrix="${4:-}"

    log_step "Creating enhanced test execution plan..."

    # Parse browsers or use matrix
    local browser_list=()
    if [[ -n "$browser_matrix" ]]; then
        # Get browsers from matrix configuration
        if [[ -f "$CAPABILITIES_FILE" ]]; then
            local matrix_browsers
            matrix_browsers=$(jq -r ".test_matrices.${browser_matrix}.browsers[]" "$CAPABILITIES_FILE" 2>/dev/null || echo "")
            if [[ -n "$matrix_browsers" ]]; then
                while IFS= read -r browser_spec; do
                    if [[ -n "$browser_spec" ]]; then
                        browser_list+=("$browser_spec")
                    fi
                done <<< "$matrix_browsers"
                log_info "Using browser matrix: $browser_matrix"
            else
                log_warning "Matrix $browser_matrix not found, using default browsers"
                IFS=',' read -ra browser_list <<< "$browsers"
            fi
        else
            log_warning "Capabilities file not found, using provided browsers"
            IFS=',' read -ra browser_list <<< "$browsers"
        fi
    else
        IFS=',' read -ra browser_list <<< "$browsers"
    fi

    local browser_count=${#browser_list[@]}

    # Calculate optimal workers
    local calculated_workers
    calculated_workers=$(calculate_optimal_workers "$browser_count" "$workers")

    # Create execution plan file
    local plan_file="$TEST_RESULTS_DIR/execution-plan.json"
    cat > "$plan_file" << EOF
{
    "execution_id": "$TEST_EXECUTION_ID",
    "start_time": "$TEST_START_TIME",
    "browsers": $(printf '%s\n' "${browser_list[@]}" | jq -R . | jq -s .),
    "browser_count": $browser_count,
    "browser_matrix": ${browser_matrix:+\"$browser_matrix\"},
    "parallel_workers": $calculated_workers,
    "test_patterns": "$test_patterns",
    "timeout": $DEFAULT_TEST_TIMEOUT,
    "retry_count": $DEFAULT_RETRY_COUNT,
    "report_formats": "$DEFAULT_REPORT_FORMAT"
}
EOF

    log_info "Execution plan created: $plan_file"
    if [[ -n "$browser_matrix" ]]; then
        log_info "Browser matrix: $browser_matrix"
    fi
    log_info "Browsers: ${browser_list[*]} (${browser_count} total)"
    log_info "Parallel workers: ${calculated_workers}"
    log_success "Enhanced test execution plan completed"

    # Return the plan file path
    echo "$plan_file"
}

# Enhanced test suite execution with browser matrix support
execute_test_suite() {
    local plan_file="$1"
    local browsers="${2:-chrome}"
    local workers="${3:-auto}"
    local test_patterns="${4:-tests/e2e/}"
    local browser_matrix="${5:-}"

    log_step "Executing enhanced test suite..."

    # Parse execution plan
    local browser_count
    browser_count=$(jq -r '.browser_count' "$plan_file")
    local calculated_workers
    calculated_workers=$(jq -r '.parallel_workers' "$plan_file")
    local matrix_name
    matrix_name=$(jq -r '.browser_matrix // empty' "$plan_file")

    # Setup browser matrix if specified
    if [[ -n "$matrix_name" && "$matrix_name" != "null" ]]; then
        log_info "Setting up browser matrix: $matrix_name"
        if ! setup_dynamic_browsers "$matrix_name" false; then
            log_error "Failed to setup browser matrix: $matrix_name"
            return 1
        fi
    fi

    # Parse browsers from plan
    local browser_array=()
    while IFS= read -r browser; do
        browser_array+=("$browser")
    done < <(jq -r '.browsers[]' "$plan_file")

    local test_results=()
    local test_pids=()

    # Execute tests for each browser in parallel
    for browser in "${browser_array[@]}"; do
        log_info "Starting test execution for browser: $browser"

        # Create browser-specific result directory
        local browser_results_dir="$TEST_RESULTS_DIR/$browser"
        mkdir -p "$browser_results_dir"

        # Execute tests for this browser
        execute_browser_tests "$browser" "$calculated_workers" "$test_patterns" "$browser_results_dir" "" &
        local test_pid=$!
        test_pids+=("$test_pid")

        log_info "Test execution started for $browser (PID: $test_pid)"

        # Small delay to prevent resource conflicts
        sleep 2
    done

    # Wait for all test executions to complete
    log_step "Waiting for test executions to complete..."
    local failed_browsers=()

    for i in "${!test_pids[@]}"; do
        local pid="${test_pids[$i]}"
        local browser="${browser_array[$i]}"

        if wait "$pid"; then
            log_success "Test execution completed for $browser"
            test_results+=("$browser:success")
        else
            log_error "Test execution failed for $browser"
            test_results+=("$browser:failed")
            failed_browsers+=("$browser")
        fi
    done

    # Generate summary
    log_step "Test execution summary:"
    for result in "${test_results[@]}"; do
        local browser_name="${result%:*}"
        local result_status="${result#*:}"
        if [[ "$result_status" == "success" ]]; then
            log_success "$browser_name: PASSED"
        else
            log_error "$browser_name: FAILED"
        fi
    done

    # Cleanup dynamic browsers if matrix was used
    if [[ -n "$matrix_name" && "$matrix_name" != "null" ]]; then
        cleanup_dynamic_browsers
    fi

    # Return overall status
    if [[ ${#failed_browsers[@]} -eq 0 ]]; then
        log_success "All browser tests completed successfully"
        return 0
    else
        log_error "Some browser tests failed: ${failed_browsers[*]}"
        return 1
    fi
}

# =============================================================================
# Stack Management Integration
# =============================================================================

ensure_stack_ready() {
    local mode="${1:-standard}"
    local profiles="${2:-}"

    log_step "Ensuring Docker stack is ready for testing..."

    # Check if stack is already running
    if "$STACK_MANAGER" status >/dev/null 2>&1; then
        log_info "Stack appears to be running, validating health..."

        if "$HEALTH_MANAGER" check >/dev/null 2>&1; then
            log_success "Stack is healthy and ready"
            return 0
        else
            log_warning "Stack is unhealthy, restarting..."
            "$STACK_MANAGER" restart "$mode" 300 "$profiles"
        fi
    else
        log_info "Stack is not running, starting..."
        "$STACK_MANAGER" start "$mode" 300 "$profiles"
    fi

    # Final health check
    if ! "$HEALTH_MANAGER" wait 120; then
        log_error "Stack failed to become ready within timeout"
        return 1
    fi

    log_success "Stack is ready for test execution"
}

# =============================================================================
# Test Retry and Recovery
# =============================================================================

retry_failed_tests() {
    local plan_file="$1"
    local max_retries="${2:-$DEFAULT_RETRY_COUNT}"

    log_step "Checking for failed tests to retry..."

    # Extract failed browsers from plan
    local failed_browsers
    failed_browsers=$(jq -r '.results[] | select(endswith(":failed")) | split(":")[0]' "$plan_file" 2>/dev/null || echo "")

    if [[ -z "$failed_browsers" ]]; then
        log_info "No failed tests to retry"
        return 0
    fi

    log_info "Found failed browsers: $failed_browsers"

    # Retry each failed browser
    local retry_count=1
    while [[ $retry_count -le $max_retries ]] && [[ -n "$failed_browsers" ]]; do
        log_step "Retry attempt $retry_count/$max_retries"

        local retry_results=()
        for browser in $failed_browsers; do
            log_info "Retrying tests for browser: $browser"

            # Create retry-specific result directory
            local retry_dir="$TEST_RESULTS_DIR/${browser}_retry_${retry_count}"
            mkdir -p "$retry_dir"

            # Execute retry
            if execute_browser_tests "$browser" "1" "tests/e2e/" "$retry_dir" ""; then
                log_success "Retry successful for $browser"
                # Remove from failed list
                failed_browsers=$(echo "$failed_browsers" | sed "s/$browser//g" | tr -s ' ')
            else
                log_warning "Retry failed for $browser"
            fi
        done

        ((retry_count++))
    done

    # Update plan with retry results
    if [[ -z "$failed_browsers" ]]; then
        log_success "All retries completed successfully"
        return 0
    else
        log_error "Some tests still failing after retries: $failed_browsers"
        return 1
    fi
}

# =============================================================================
# Report Generation and Aggregation
# =============================================================================

generate_consolidated_report() {
    local plan_file="$1"
    local output_dir="${2:-$TEST_REPORTS_DIR}"

    log_step "Generating consolidated test report..."

    mkdir -p "$output_dir"

    # Extract execution metadata
    local execution_id
    execution_id=$(jq -r '.execution_id' "$plan_file")
    local start_time
    start_time=$(jq -r '.start_time' "$plan_file")
    local end_time
    end_time=$(jq -r '.end_time // empty' "$plan_file")

    # Create consolidated HTML report
    local html_report="$output_dir/consolidated-report-${execution_id}.html"
    create_consolidated_html_report "$plan_file" "$html_report"

    # Create consolidated JSON report
    local json_report="$output_dir/consolidated-report-${execution_id}.json"
    create_consolidated_json_report "$plan_file" "$json_report"

    # Create JUnit XML report
    local junit_report="$output_dir/consolidated-junit-${execution_id}.xml"
    create_consolidated_junit_report "$plan_file" "$junit_report"

    log_success "Consolidated reports generated:"
    log_info "  HTML: $html_report"
    log_info "  JSON: $json_report"
    log_info "  JUnit: $junit_report"
}

create_consolidated_html_report() {
    local plan_file="$1"
    local output_file="$2"

    local execution_id
    execution_id=$(jq -r '.execution_id' "$plan_file")
    local browsers
    browsers=$(jq -r '.browsers[]' "$plan_file" | tr '\n' ' ')

    cat > "$output_file" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CrackSeg E2E Test Report - $execution_id</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f5f5f5; padding: 20px; border-radius: 5px; }
        .browser-section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .success { color: #28a745; }
        .failure { color: #dc3545; }
        .warning { color: #ffc107; }
        .timestamp { font-size: 0.9em; color: #666; }
    </style>
</head>
<body>
    <div class="header">
        <h1>CrackSeg E2E Test Report</h1>
        <p><strong>Execution ID:</strong> $execution_id</p>
        <p><strong>Browsers:</strong> $browsers</p>
        <p class="timestamp">Generated: $(date)</p>
    </div>

    <div id="summary">
        <h2>Test Summary</h2>
        <!-- Summary will be populated by JavaScript or additional processing -->
    </div>

    <div id="browser-results">
        <h2>Browser-Specific Results</h2>
        <!-- Browser results will be populated -->
    </div>

    <div id="artifacts">
        <h2>Test Artifacts</h2>
        <p>Screenshots and videos are available in the test results directory.</p>
    </div>
</body>
</html>
EOF

    log_info "HTML report template created: $output_file"
}

create_consolidated_json_report() {
    local plan_file="$1"
    local output_file="$2"

    # Merge plan with additional metadata
    jq --arg generated_at "$(date -Iseconds)" \
       '. + {generated_at: $generated_at, report_type: "consolidated"}' \
       "$plan_file" > "$output_file"

    log_info "JSON report created: $output_file"
}

create_consolidated_junit_report() {
    local plan_file="$1"
    local output_file="$2"

    # Basic JUnit XML structure
    cat > "$output_file" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="CrackSeg E2E Tests" tests="0" failures="0" errors="0" time="0">
    <!-- Test suites will be populated from individual browser results -->
</testsuites>
EOF

    log_info "JUnit report template created: $output_file"
}

# =============================================================================
# Artifact Collection and Cleanup
# =============================================================================

collect_test_artifacts() {
    local plan_file="$1"

    log_step "Collecting test artifacts..."

    # Use artifact manager for collection
    "$ARTIFACT_MANAGER" collect --test-session --execution-id "$(jq -r '.execution_id' "$plan_file")"

    # Create artifact index
    local artifact_index="$TEST_ARTIFACTS_DIR/artifact-index.json"
    create_artifact_index "$artifact_index"

    log_success "Test artifacts collected and indexed"
}

create_artifact_index() {
    local index_file="$1"

    # Scan artifacts directory and create index
    cat > "$index_file" << EOF
{
    "execution_id": "$TEST_EXECUTION_ID",
    "collection_time": "$(date -Iseconds)",
    "artifacts": {
        "screenshots": $(find "$TEST_RESULTS_DIR" -name "*.png" -type f | jq -R . | jq -s .),
        "videos": $(find "$TEST_RESULTS_DIR" -name "*.mp4" -type f | jq -R . | jq -s .),
        "logs": $(find "$TEST_RESULTS_DIR" -name "*.log" -type f | jq -R . | jq -s .),
        "reports": $(find "$TEST_REPORTS_DIR" -name "*.html" -o -name "*.json" -o -name "*.xml" | jq -R . | jq -s .)
    }
}
EOF

    log_info "Artifact index created: $index_file"
}

# =============================================================================
# Help and Usage
# =============================================================================

show_help() {
    cat << EOF
${CYAN}CrackSeg E2E Test Orchestrator - Advanced Test Execution System${NC}

${YELLOW}USAGE:${NC}
    $0 [COMMAND] [OPTIONS]

${YELLOW}COMMANDS:${NC}
    ${GREEN}run [browsers] [workers] [patterns]${NC}      Run complete E2E test suite
    ${GREEN}retry [plan-file] [max-retries]${NC}         Retry failed tests
    ${GREEN}report [plan-file] [output-dir]${NC}         Generate consolidated report
    ${GREEN}status${NC}                                  Show current test status
    ${GREEN}cleanup${NC}                                 Clean up test artifacts
    ${GREEN}help${NC}                                    Show this help

${YELLOW}OPTIONS:${NC}
    ${GREEN}browsers${NC}     Comma-separated browser list (chrome,firefox,edge)
    ${GREEN}workers${NC}      Number of parallel workers (auto, 1-8)
    ${GREEN}patterns${NC}     Test file patterns (tests/e2e/, tests/integration/)

${YELLOW}BROWSER OPTIONS:${NC}
    ${GREEN}chrome${NC}       Google Chrome (default)
    ${GREEN}firefox${NC}      Mozilla Firefox
    ${GREEN}edge${NC}         Microsoft Edge

${YELLOW}EXAMPLES:${NC}
    ${GREEN}$0 run${NC}                                  # Run with defaults (chrome, auto workers)
    ${GREEN}$0 run chrome,firefox 4${NC}                # Multi-browser with 4 workers
    ${GREEN}$0 run chrome 2 tests/e2e/test_login*${NC}   # Specific test pattern
    ${GREEN}$0 retry execution-plan.json 3${NC}         # Retry failed tests up to 3 times
    ${GREEN}$0 report execution-plan.json${NC}          # Generate consolidated report

${YELLOW}ENVIRONMENT VARIABLES:${NC}
    ${GREEN}ENABLE_COVERAGE${NC}        Enable test coverage (true/false)
    ${GREEN}ENABLE_SCREENSHOTS${NC}     Enable screenshot capture (true/false)
    ${GREEN}ENABLE_VIDEOS${NC}          Enable video recording (true/false)
    ${GREEN}TEST_TIMEOUT${NC}           Override default test timeout

${YELLOW}QUICK START:${NC}
    ${GREEN}$0 run${NC}                                  # Start basic test execution

EOF
}

# =============================================================================
# Main Script Logic
# =============================================================================

main() {
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi

    case "$1" in
        "run")
            print_banner
            setup_test_environment
            validate_test_prerequisites

            local browsers="${2:-$DEFAULT_BROWSERS}"
            local workers="${3:-$DEFAULT_PARALLEL_WORKERS}"
            local patterns="${4:-tests/e2e/}"

            # Ensure stack is ready
            ensure_stack_ready "standard" ""

            # Create execution plan
            local plan_file
            plan_file=$(create_test_execution_plan "$browsers" "$workers" "$patterns")

            # Execute tests
            if execute_test_suite "$plan_file" "$browsers" "$workers" "$patterns"; then
                log_success "Test execution completed successfully"
                exit_code=0
            else
                log_warning "Test execution completed with failures, attempting retries..."
                if retry_failed_tests "$plan_file"; then
                    log_success "Retries completed successfully"
                    exit_code=0
                else
                    log_error "Test execution failed even after retries"
                    exit_code=1
                fi
            fi

            # Generate reports and collect artifacts
            generate_consolidated_report "$plan_file"
            collect_test_artifacts "$plan_file"

            exit $exit_code
            ;;
        "retry")
            local plan_file="${2:-}"
            local max_retries="${3:-$DEFAULT_RETRY_COUNT}"

            if [[ ! -f "$plan_file" ]]; then
                log_error "Plan file not found: $plan_file"
                exit 1
            fi

            retry_failed_tests "$plan_file" "$max_retries"
            ;;
        "report")
            local plan_file="${2:-}"
            local output_dir="${3:-$TEST_REPORTS_DIR}"

            if [[ ! -f "$plan_file" ]]; then
                log_error "Plan file not found: $plan_file"
                exit 1
            fi

            generate_consolidated_report "$plan_file" "$output_dir"
            ;;
        "status")
            # Show current test execution status
            if [[ -f "$TEST_RESULTS_DIR/execution-plan.json" ]]; then
                local plan_file="$TEST_RESULTS_DIR/execution-plan.json"
                log_info "Current execution plan:"
                jq . "$plan_file"
            else
                log_info "No active test execution found"
            fi
            ;;
        "cleanup")
            log_step "Cleaning up test artifacts..."
            "$ARTIFACT_MANAGER" cleanup
            rm -rf "$TEST_RESULTS_DIR" "$TEST_ARTIFACTS_DIR" 2>/dev/null || true
            log_success "Cleanup completed"
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            echo
            show_help
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"
