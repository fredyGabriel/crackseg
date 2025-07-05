#!/usr/bin/env bash
# =============================================================================
# CrackSeg Docker Stack Manager - Master Orchestration Script
# =============================================================================
# Purpose: Unified orchestration for complete Docker testing infrastructure
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
NETWORK_MANAGER="$SCRIPT_DIR/network-manager.sh"
HEALTH_MANAGER="$SCRIPT_DIR/health-check-manager.sh"
TEST_ENV_SCRIPT="$SCRIPT_DIR/start-test-env.sh"
GRID_MANAGER="$SCRIPT_DIR/manage-grid.sh"
ARTIFACT_MANAGER="$SCRIPT_DIR/artifact-manager.sh"
BROWSER_MANAGER="$SCRIPT_DIR/browser-manager.sh"
E2E_ORCHESTRATOR="$SCRIPT_DIR/e2e-test-orchestrator.sh"
CAPABILITIES_FILE="$DOCKER_DIR/browser-capabilities.json"

# Default configuration
DEFAULT_STARTUP_TIMEOUT=300
DEFAULT_SHUTDOWN_TIMEOUT=60
DEFAULT_HEALTH_CHECK_INTERVAL=30
DEFAULT_PARALLEL_WORKERS="auto"

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
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] [ORCHESTRATOR]${NC} $*"
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
    echo "  CrackSeg Docker Stack Manager - Master Orchestration System"
    echo "  Unified Infrastructure Management for E2E Testing (Subtask 13.8)"
    echo "============================================================================="
    echo -e "${NC}"
}

# =============================================================================
# Validation Functions
# =============================================================================

validate_environment() {
    log_step "Validating environment prerequisites..."

    # Check required commands
    local required_commands=("docker" "docker-compose" "jq" "curl")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "Required command not found: $cmd"
            exit 1
        fi
    done

    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi

    # Check compose file
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    fi

    # Check component scripts
    local scripts=("$NETWORK_MANAGER" "$HEALTH_MANAGER" "$GRID_MANAGER" "$ARTIFACT_MANAGER" "$BROWSER_MANAGER" "$E2E_ORCHESTRATOR")
    for script in "${scripts[@]}"; do
        if [[ ! -f "$script" ]]; then
            log_error "Required component script not found: $script"
            exit 1
        fi
        if [[ ! -x "$script" ]]; then
            chmod +x "$script"
            log_info "Made script executable: $script"
        fi
    done

    log_success "Environment validation completed"
}

validate_docker_compose() {
    log_step "Validating Docker Compose configuration..."

    cd "$DOCKER_DIR"

    # Validate compose file syntax
    if ! docker-compose -f docker-compose.test.yml config --quiet; then
        log_error "Docker Compose configuration is invalid"
        return 1
    fi

    # Check for required services
    local required_services=("streamlit-app" "selenium-hub" "chrome-node" "firefox-node")
    local available_services
    available_services=$(docker-compose -f docker-compose.test.yml config --services)

    for service in "${required_services[@]}"; do
        if ! echo "$available_services" | grep -q "^$service$"; then
            log_error "Required service not found in compose file: $service"
            return 1
        fi
    done

    log_success "Docker Compose configuration is valid"
}

# =============================================================================
# Stack Lifecycle Management
# =============================================================================

start_stack() {
    local mode="${1:-standard}"
    local timeout="${2:-$DEFAULT_STARTUP_TIMEOUT}"
    local profiles="${3:-}"

    print_banner
    log "Starting CrackSeg Docker testing stack..."
    log_info "Mode: $mode | Timeout: ${timeout}s | Profiles: ${profiles:-none}"

    # Step 1: Create networks
    log_step "Step 1/6: Setting up Docker networks..."
    if ! "$NETWORK_MANAGER" start; then
        log_error "Failed to create Docker networks"
        return 1
    fi

    # Step 2: Start core infrastructure
    log_step "Step 2/6: Starting core infrastructure services..."

    local start_args=("--mode" "$mode" "--detach" "--timeout" "$timeout")
    if [[ -n "$profiles" ]]; then
        # Parse profiles and add them one by one
        IFS=',' read -ra PROFILE_ARRAY <<< "$profiles"
        for profile in "${PROFILE_ARRAY[@]}"; do
            start_args+=("--profile" "$profile")
        done
    fi

    if ! "$TEST_ENV_SCRIPT" "${start_args[@]}"; then
        log_error "Failed to start core infrastructure"
        cleanup_on_failure
        return 1
    fi

    # Step 3: Validate Selenium Grid
    log_step "Step 3/6: Validating Selenium Grid infrastructure..."
    if ! "$GRID_MANAGER" status; then
        log_warning "Selenium Grid validation failed, attempting restart..."
        "$GRID_MANAGER" restart
        sleep 10
        if ! "$GRID_MANAGER" status; then
            log_error "Failed to establish Selenium Grid"
            cleanup_on_failure
            return 1
        fi
    fi

    # Step 4: Wait for all services to be healthy
    log_step "Step 4/6: Waiting for services to be healthy..."
    if ! "$HEALTH_MANAGER" wait "$timeout"; then
        log_error "Services failed to become healthy within timeout"
        show_service_status
        cleanup_on_failure
        return 1
    fi

    # Step 5: Test network connectivity
    log_step "Step 5/6: Testing network connectivity..."
    if ! "$NETWORK_MANAGER" test; then
        log_warning "Network connectivity test failed, but proceeding..."
    fi

    # Step 6: Start monitoring
    log_step "Step 6/6: Starting health monitoring..."
    "$HEALTH_MANAGER" monitor &
    local monitor_pid=$!
    echo "$monitor_pid" > "/tmp/docker-stack-monitor.pid"

    log_success "Docker stack startup completed successfully!"
    show_stack_info
}

stop_stack() {
    local timeout="${1:-$DEFAULT_SHUTDOWN_TIMEOUT}"
    local cleanup_volumes="${2:-false}"

    log "Stopping CrackSeg Docker testing stack..."
    log_info "Timeout: ${timeout}s | Cleanup volumes: $cleanup_volumes"

    # Step 1: Stop monitoring
    log_step "Step 1/5: Stopping health monitoring..."
    stop_monitoring

    # Step 2: Collect artifacts before shutdown
    log_step "Step 2/5: Collecting final artifacts..."
    "$ARTIFACT_MANAGER" collect --final || log_warning "Artifact collection failed"

    # Step 3: Stop services gracefully
    log_step "Step 3/5: Stopping Docker services..."
    cd "$DOCKER_DIR"

    if [[ "$cleanup_volumes" == "true" ]]; then
        docker-compose -f docker-compose.test.yml down -v --remove-orphans --timeout "$timeout"
    else
        docker-compose -f docker-compose.test.yml down --remove-orphans --timeout "$timeout"
    fi

    # Step 4: Remove networks
    log_step "Step 4/5: Cleaning up Docker networks..."
    "$NETWORK_MANAGER" stop

    # Step 5: Final cleanup
    log_step "Step 5/5: Final cleanup..."
    docker system prune -f >/dev/null 2>&1 || true

    log_success "Docker stack stopped successfully!"
}

restart_stack() {
    local mode="${1:-standard}"
    local timeout="${2:-$DEFAULT_STARTUP_TIMEOUT}"
    local profiles="${3:-}"

    log "Restarting CrackSeg Docker testing stack..."

    stop_stack "$DEFAULT_SHUTDOWN_TIMEOUT" "false"
    sleep 5
    start_stack "$mode" "$timeout" "$profiles"
}

# =============================================================================
# Stack Status and Monitoring
# =============================================================================

show_stack_status() {
    print_banner
    echo -e "${BLUE}Current Stack Status:${NC}"
    echo "===================="

    # Check networks
    echo -e "\n${CYAN}📡 Networks:${NC}"
    "$NETWORK_MANAGER" status | grep -E "(Status|Containers|Connectivity)" || echo "Network status unavailable"

    # Check services
    echo -e "\n${CYAN}🐳 Services:${NC}"
    cd "$DOCKER_DIR"
    if docker-compose -f docker-compose.test.yml ps --format table | grep -q "Up"; then
        docker-compose -f docker-compose.test.yml ps --format table
    else
        echo "No services running"
    fi

    # Check health
    echo -e "\n${CYAN}❤️  Health Status:${NC}"
    "$HEALTH_MANAGER" summary 2>/dev/null || echo "Health status unavailable"

    # Check monitoring
    echo -e "\n${CYAN}📊 Monitoring:${NC}"
    if [[ -f "/tmp/docker-stack-monitor.pid" ]] && kill -0 "$(cat "/tmp/docker-stack-monitor.pid")" 2>/dev/null; then
        echo "✅ Health monitoring active (PID: $(cat "/tmp/docker-stack-monitor.pid"))"
    else
        echo "❌ Health monitoring not running"
    fi
}

show_stack_info() {
    echo
    echo -e "${GREEN}🎉 Stack is ready! Access points:${NC}"
    echo "=================================="
    echo "🌐 Streamlit Application: http://localhost:8501"
    echo "🕷️  Selenium Grid Console: http://localhost:4444/grid/console"
    echo "📊 Grid Status API: http://localhost:4444/wd/hub/status"
    echo
    echo -e "${CYAN}📋 Management commands:${NC}"
    echo "Status:   $0 status"
    echo "Logs:     $0 logs [service]"
    echo "Monitor:  $0 monitor"
    echo "Stop:     $0 stop"
    echo
}

monitor_stack() {
    local interval="${1:-$DEFAULT_HEALTH_CHECK_INTERVAL}"

    log "Starting stack monitoring (interval: ${interval}s)..."

    while true; do
        clear
        show_stack_status
        echo -e "\n${YELLOW}Last updated: $(date) | Monitoring every ${interval}s${NC}"
        echo "Press Ctrl+C to stop monitoring"

        sleep "$interval"
    done
}

# =============================================================================
# Test Execution Orchestration
# =============================================================================

run_e2e_tests() {
    local browser="${1:-chrome}"
    local parallel_workers="${2:-$DEFAULT_PARALLEL_WORKERS}"
    local test_profiles="${3:-}"

    log "Orchestrating E2E test execution..."
    log_info "Browser: $browser | Workers: $parallel_workers | Profiles: ${test_profiles:-none}"

    # Ensure stack is running
    if ! is_stack_running; then
        log_warning "Stack is not running, starting it first..."
        start_stack "standard" "$DEFAULT_STARTUP_TIMEOUT" "$test_profiles"
    fi

    # Wait for services to be ready
    log_step "Ensuring services are ready for testing..."
    if ! "$HEALTH_MANAGER" wait 60; then
        log_error "Services are not ready for testing"
        return 1
    fi

    # Execute tests via test runner
    log_step "Executing E2E tests..."
    cd "$SCRIPT_DIR"

    if ! ./run-test-runner.sh run \
        --browser "$browser" \
        --parallel-workers "$parallel_workers" \
        --headless true \
        --coverage \
        --html-report \
        --screenshots; then
        log_error "E2E test execution failed"
        return 1
    fi

    # Collect artifacts
    log_step "Collecting test artifacts..."
    "$ARTIFACT_MANAGER" collect --test-session

    log_success "E2E test execution completed!"
}

# =============================================================================
# Log Collection and Management
# =============================================================================

collect_logs() {
    local service="${1:-all}"
    local follow="${2:-false}"
    local lines="${3:-100}"

    log "Collecting logs for service: $service"

    cd "$DOCKER_DIR"

    case "$service" in
        "all")
            if [[ "$follow" == "true" ]]; then
                docker-compose -f docker-compose.test.yml logs -f
            else
                docker-compose -f docker-compose.test.yml logs --tail="$lines"
            fi
            ;;
        *)
            if [[ "$follow" == "true" ]]; then
                docker-compose -f docker-compose.test.yml logs -f "$service"
            else
                docker-compose -f docker-compose.test.yml logs --tail="$lines" "$service"
            fi
            ;;
    esac
}

aggregate_logs() {
    local output_dir="${1:-$PROJECT_ROOT/test-results/logs}"
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)

    log "Aggregating logs to: $output_dir"
    mkdir -p "$output_dir"

    cd "$DOCKER_DIR"

    # Get all services
    local services
    services=$(docker-compose -f docker-compose.test.yml config --services)

    # Collect logs for each service
    for service in $services; do
        local log_file="$output_dir/${service}-${timestamp}.log"
        docker-compose -f docker-compose.test.yml logs "$service" > "$log_file" 2>&1
        log_info "Collected logs for $service: $log_file"
    done

    # Create combined log
    local combined_log="$output_dir/combined-${timestamp}.log"
    docker-compose -f docker-compose.test.yml logs > "$combined_log" 2>&1

    log_success "Log aggregation completed: $output_dir"
}

# =============================================================================
# Cleanup and Maintenance
# =============================================================================

cleanup_stack() {
    local scope="${1:-containers}"

    log "Performing stack cleanup: $scope"

    case "$scope" in
        "containers")
            stop_stack "$DEFAULT_SHUTDOWN_TIMEOUT" "false"
            ;;
        "volumes")
            stop_stack "$DEFAULT_SHUTDOWN_TIMEOUT" "true"
            ;;
        "all")
            stop_stack "$DEFAULT_SHUTDOWN_TIMEOUT" "true"
            docker system prune -af
            "$ARTIFACT_MANAGER" cleanup --all
            ;;
        "artifacts")
            "$ARTIFACT_MANAGER" cleanup
            ;;
        *)
            log_error "Unknown cleanup scope: $scope"
            log_info "Available scopes: containers, volumes, all, artifacts"
            return 1
            ;;
    esac

    log_success "Cleanup completed: $scope"
}

cleanup_on_failure() {
    log_warning "Cleaning up due to failure..."
    stop_monitoring
    cd "$DOCKER_DIR"
    docker-compose -f docker-compose.test.yml down --remove-orphans >/dev/null 2>&1 || true
}

# =============================================================================
# Utility Functions
# =============================================================================

is_stack_running() {
    cd "$DOCKER_DIR"
    local running_services
    running_services=$(docker-compose -f docker-compose.test.yml ps -q | wc -l)
    [[ "$running_services" -gt 0 ]]
}

stop_monitoring() {
    if [[ -f "/tmp/docker-stack-monitor.pid" ]]; then
        local monitor_pid
        monitor_pid=$(cat "/tmp/docker-stack-monitor.pid")
        if kill -0 "$monitor_pid" 2>/dev/null; then
            kill "$monitor_pid" 2>/dev/null || true
            rm -f "/tmp/docker-stack-monitor.pid"
            log_info "Stopped health monitoring"
        fi
    fi
}

show_service_status() {
    log_info "Current service status:"
    cd "$DOCKER_DIR"
    docker-compose -f docker-compose.test.yml ps
}

# =============================================================================
# Cross-Browser Management Integration (Subtask 13.10)
# =============================================================================

browser_management() {
    local action="${1:-help}"
    local param="${2:-}"

    # Ensure browser manager is available
    if [[ ! -f "$BROWSER_MANAGER" ]]; then
        log_error "Browser manager not found: $BROWSER_MANAGER"
        return 1
    fi

    # Make executable if needed
    if [[ ! -x "$BROWSER_MANAGER" ]]; then
        chmod +x "$BROWSER_MANAGER"
    fi

    case "$action" in
        "list"|"browsers")
            log_step "Listing available browser configurations..."
            "$BROWSER_MANAGER" browsers
            ;;
        "matrix")
            if [[ -z "$param" ]]; then
                log_error "Matrix name required (smoke_test, compatibility_test, mobile_test, full_matrix)"
                return 1
            fi
            log_step "Creating browser matrix: $param"
            "$BROWSER_MANAGER" matrix "$param" true
            ;;
        "create")
            if [[ -z "$param" ]]; then
                log_error "Browser specification required (e.g., chrome:latest, firefox:beta)"
                return 1
            fi
            log_step "Creating browser container: $param"
            "$BROWSER_MANAGER" create "$param"
            ;;
        "cleanup")
            log_step "Cleaning up dynamic browser containers..."
            "$BROWSER_MANAGER" cleanup
            ;;
        "status")
            log_step "Dynamic browser container status..."
            "$BROWSER_MANAGER" list
            ;;
        "help")
            cat << EOF

Browser Management Commands:
  browsers                   List available browser configurations
  browser-matrix <name>      Create browser matrix for testing
  browser-create <spec>      Create single browser container
  browser-cleanup           Remove all dynamic browser containers
  browser-status            Show dynamic browser container status

Available Browser Matrices:
  smoke_test                Basic Chrome + Firefox testing
  compatibility_test        Multi-version browser testing
  mobile_test               Mobile emulation testing
  full_matrix               Complete cross-browser + mobile testing

Browser Specifications:
  chrome:latest, chrome:stable, chrome:beta
  firefox:latest, firefox:stable, firefox:beta
  edge:latest, edge:stable
  chrome:pixel_5, chrome:iphone_12, chrome:galaxy_s21
  firefox:responsive, edge:surface_duo

EOF
            ;;
        *)
            log_error "Unknown browser management action: $action"
            return 1
            ;;
    esac
}

run_cross_browser_tests() {
    local matrix_name="${1:-smoke_test}"
    local test_patterns="${2:-tests/e2e/}"
    local force_recreate="${3:-false}"

    log_step "Running cross-browser tests with matrix: $matrix_name"

    # Ensure E2E orchestrator is available
    if [[ ! -f "$E2E_ORCHESTRATOR" ]]; then
        log_error "E2E orchestrator not found: $E2E_ORCHESTRATOR"
        return 1
    fi

    if [[ ! -x "$E2E_ORCHESTRATOR" ]]; then
        chmod +x "$E2E_ORCHESTRATOR"
    fi

    # Start the stack if not running
    if ! is_stack_running; then
        log_info "Stack not running, starting..."
        start_stack "standard" "$DEFAULT_STARTUP_TIMEOUT" ""
    fi

    # Run tests with browser matrix
    log_info "Executing cross-browser test suite..."
    if "$E2E_ORCHESTRATOR" run \
        --matrix "$matrix_name" \
        --patterns "$test_patterns" \
        --force-recreate "$force_recreate"; then
        log_success "Cross-browser tests completed successfully"
        return 0
    else
        log_error "Cross-browser tests failed"
        return 1
    fi
}

# =============================================================================
# Help and Usage
# =============================================================================

show_help() {
    cat << EOF
${CYAN}CrackSeg Docker Stack Manager - Master Orchestration System${NC}

${YELLOW}USAGE:${NC}
    $0 [COMMAND] [OPTIONS]

${YELLOW}COMMANDS:${NC}
    ${GREEN}Lifecycle Management:${NC}
        ${GREEN}start [mode] [timeout] [profiles]${NC}     Start complete Docker stack
        ${GREEN}stop [timeout] [cleanup-volumes]${NC}      Stop Docker stack
        ${GREEN}restart [mode] [timeout] [profiles]${NC}   Restart Docker stack
        ${GREEN}status${NC}                               Show stack status

    ${GREEN}Test Orchestration:${NC}
        ${GREEN}test [browser] [workers] [profiles]${NC}   Run E2E tests
        ${GREEN}monitor [interval]${NC}                   Monitor stack health

    ${GREEN}Log Management:${NC}
        ${GREEN}logs [service] [follow] [lines]${NC}       Show service logs
        ${GREEN}collect-logs [output-dir]${NC}            Aggregate all logs

    ${GREEN}Maintenance:${NC}
        ${GREEN}cleanup [scope]${NC}                      Clean up resources
        ${GREEN}validate${NC}                            Validate environment
        ${GREEN}help${NC}                                Show this help

    ${GREEN}Browser Management:${NC}
        ${GREEN}browsers${NC}                            List available browser configurations
        ${GREEN}browser-matrix <name>${NC}                Create browser matrix for testing
        ${GREEN}browser-cleanup${NC}                        Cleanup dynamic browser containers
        ${GREEN}browser-create <spec>${NC}                  Create single browser container
        ${GREEN}browser-status${NC}                           Show dynamic browser container status
        ${GREEN}test-cross-browser <matrix_name> <test_patterns> <force_recreate>${NC} Run cross-browser tests

${YELLOW}MODE OPTIONS:${NC}
    ${GREEN}standard${NC}     Default testing mode
    ${GREEN}debug${NC}        Debug mode with additional logging
    ${GREEN}minimal${NC}      Minimal resource usage

${YELLOW}PROFILE OPTIONS:${NC}
    ${GREEN}recording${NC}    Enable video recording
    ${GREEN}monitoring${NC}   Extended monitoring
    ${GREEN}debug${NC}        Debug containers with VNC access

${YELLOW}CLEANUP SCOPES:${NC}
    ${GREEN}containers${NC}   Stop and remove containers
    ${GREEN}volumes${NC}      Also remove persistent volumes
    ${GREEN}all${NC}          Complete cleanup including images
    ${GREEN}artifacts${NC}    Clean up test artifacts only

${YELLOW}EXAMPLES:${NC}
    ${GREEN}$0 start standard 300${NC}                   # Start stack with 5min timeout
    ${GREEN}$0 start debug 600 "recording,monitoring"${NC} # Debug mode with extras
    ${GREEN}$0 test chrome 4${NC}                        # Run tests with Chrome, 4 workers
    ${GREEN}$0 logs streamlit-app true${NC}              # Follow Streamlit logs
    ${GREEN}$0 cleanup volumes${NC}                      # Stop and remove volumes
    ${GREEN}$0 monitor 30${NC}                           # Monitor every 30 seconds
    ${GREEN}$0 browsers${NC}                              # List available browser configurations
    ${GREEN}$0 browser-matrix smoke_test${NC}              # Create browser matrix for testing
    ${GREEN}$0 browser-cleanup${NC}                        # Cleanup dynamic browser containers
    ${GREEN}$0 browser-create chrome:latest${NC}            # Create single browser container
    ${GREEN}$0 browser-status${NC}                           # Show dynamic browser container status
    ${GREEN}$0 test-cross-browser smoke_test tests/e2e/${NC} Run cross-browser tests

${YELLOW}QUICK START:${NC}
    ${GREEN}$0 start${NC}                                # Start with defaults
    ${GREEN}$0 test${NC}                                 # Run E2E tests
    ${GREEN}$0 stop${NC}                                 # Stop everything

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

    # Set up error handling
    trap cleanup_on_failure ERR

    case "$1" in
        "start")
            validate_environment
            validate_docker_compose
            start_stack "${2:-standard}" "${3:-$DEFAULT_STARTUP_TIMEOUT}" "${4:-}"
            ;;
        "stop")
            stop_stack "${2:-$DEFAULT_SHUTDOWN_TIMEOUT}" "${3:-false}"
            ;;
        "restart")
            validate_environment
            validate_docker_compose
            restart_stack "${2:-standard}" "${3:-$DEFAULT_STARTUP_TIMEOUT}" "${4:-}"
            ;;
        "status")
            show_stack_status
            ;;
        "test")
            validate_environment
            run_e2e_tests "${2:-chrome}" "${3:-$DEFAULT_PARALLEL_WORKERS}" "${4:-}"
            ;;
        "monitor")
            monitor_stack "${2:-$DEFAULT_HEALTH_CHECK_INTERVAL}"
            ;;
        "logs")
            collect_logs "${2:-all}" "${3:-false}" "${4:-100}"
            ;;
        "collect-logs")
            aggregate_logs "${2:-}"
            ;;
        "cleanup")
            cleanup_stack "${2:-containers}"
            ;;
        "validate")
            validate_environment
            validate_docker_compose
            log_success "Environment validation passed"
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        "browsers"|"browser-list")
            browser_management "list"
            ;;
        "browser-matrix")
            if [[ $# -lt 2 ]]; then
                log_error "Matrix name required"
                show_help
                exit 1
            fi
            browser_management "matrix" "$2"
            ;;
        "browser-create")
            if [[ $# -lt 2 ]]; then
                log_error "Browser specification required"
                show_help
                exit 1
            fi
            browser_management "create" "$2"
            ;;
        "browser-cleanup")
            browser_management "cleanup"
            ;;
        "browser-status")
            browser_management "status"
            ;;
        "browser-help")
            browser_management "help"
            ;;
        "test-cross-browser")
            run_cross_browser_tests "${2:-smoke_test}" "${3:-tests/e2e/}" "${4:-false}"
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
