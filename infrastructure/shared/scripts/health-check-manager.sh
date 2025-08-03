#!/bin/bash
# =============================================================================
# CrackSeg Health Check Manager
# =============================================================================
# Purpose: Manage health checks for all Docker services
# Integration: Combines Docker native health checks with custom monitoring
# Usage: ./health-check-manager.sh [command] [options]
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

# Script directory and project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$DOCKER_DIR")")"

# Docker configuration
COMPOSE_FILE="$DOCKER_DIR/docker-compose.test.yml"
HEALTH_CHECK_SYSTEM="$DOCKER_DIR/health_check_system.py"

# Output directories
OUTPUT_DIR="$PROJECT_ROOT/test-results/health-checks"
DASHBOARD_DIR="$OUTPUT_DIR/dashboard"
LOGS_DIR="$OUTPUT_DIR/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_INTERVAL=30
DEFAULT_TIMEOUT=60
MONITOR_PID_FILE="/tmp/health-monitor.pid"

# =============================================================================
# Logging Functions
# =============================================================================

log() {
    echo -e "${CYAN}[$(date '+%Y-%m-%d %H:%M:%S')] [INFO]${NC} $1" | tee -a "$LOGS_DIR/health-manager.log"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS]${NC} $1" | tee -a "$LOGS_DIR/health-manager.log"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] [WARNING]${NC} $1" | tee -a "$LOGS_DIR/health-manager.log"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR]${NC} $1" >&2 | tee -a "$LOGS_DIR/health-manager.log"
}

# =============================================================================
# Setup and Validation Functions
# =============================================================================

setup_directories() {
    log "Setting up health check directories..."

    mkdir -p "$OUTPUT_DIR" "$DASHBOARD_DIR" "$LOGS_DIR"

    # Create log file with proper permissions
    touch "$LOGS_DIR/health-manager.log"

    log_success "Directories setup completed"
}

validate_environment() {
    log "Validating environment prerequisites..."

    # Check required commands
    local required_commands=("docker" "python3" "curl" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "Required command not found: $cmd"
            exit 1
        fi
    done

    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not available"
        exit 1
    fi

    # Check compose file
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    fi

    # Check health check system
    if [[ ! -f "$HEALTH_CHECK_SYSTEM" ]]; then
        log_error "Health check system not found: $HEALTH_CHECK_SYSTEM"
        exit 1
    fi

    # Check Python dependencies
    if ! python3 -c "import requests, click, asyncio" &> /dev/null; then
        log_warning "Some Python dependencies may be missing. Install requirements-testing.txt"
    fi

    log_success "Environment validation completed"
}

# =============================================================================
# Docker Service Management
# =============================================================================

get_service_status() {
    local service_name="$1"

    # Get container name from compose file
    local container_name
    container_name=$(docker compose -f "$COMPOSE_FILE" ps --format json | jq -r "select(.Service == \"$service_name\") | .Name" 2>/dev/null || echo "")

    if [[ -z "$container_name" ]]; then
        echo "unknown"
        return 1
    fi

    # Check container status
    local container_status
    container_status=$(docker inspect --format='{{.State.Status}}' "$container_name" 2>/dev/null || echo "not_found")

    if [[ "$container_status" == "running" ]]; then
        # Check health status if available
        local health_status
        health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "none")

        if [[ "$health_status" == "none" ]]; then
            echo "running"
        else
            echo "$health_status"
        fi
    else
        echo "$container_status"
    fi
}

wait_for_service_health() {
    local service_name="$1"
    local timeout="${2:-$DEFAULT_TIMEOUT}"
    local interval="${3:-5}"

    log "Waiting for $service_name to be healthy (timeout: ${timeout}s)..."

    local elapsed=0
    while [[ $elapsed -lt $timeout ]]; do
        local status
        status=$(get_service_status "$service_name")

        case "$status" in
            "healthy"|"running")
                log_success "$service_name is healthy"
                return 0
                ;;
            "starting")
                echo -n "."
                ;;
            "unhealthy")
                log_error "$service_name is unhealthy"
                return 1
                ;;
            *)
                log_warning "$service_name status: $status"
                ;;
        esac

        sleep "$interval"
        elapsed=$((elapsed + interval))
    done

    log_error "$service_name failed to become healthy within ${timeout}s"
    return 1
}

wait_for_all_services() {
    local timeout="${1:-$DEFAULT_TIMEOUT}"

    log "Waiting for all services to be healthy..."

    # Define services in dependency order
    local services=("selenium-hub" "chrome-node" "firefox-node" "streamlit-app")

    for service in "${services[@]}"; do
        if ! wait_for_service_health "$service" "$timeout"; then
            log_error "Failed waiting for service: $service"
            return 1
        fi
    done

    log_success "All services are healthy"
}

# =============================================================================
# Health Check Operations
# =============================================================================

run_health_check() {
    local output_format="${1:-json}"
    local save_file="${2:-}"

    log "Running comprehensive health check..."

    # Prepare output file path
    local output_file=""
    if [[ -n "$save_file" ]]; then
        output_file="$OUTPUT_DIR/$save_file"
    else
        output_file="$OUTPUT_DIR/health-report-$(date '+%Y%m%d-%H%M%S').json"
    fi

    # Run health check system
    cd "$DOCKER_DIR"

    if python3 "$HEALTH_CHECK_SYSTEM" check --format "$output_format" --output "$output_file"; then
        log_success "Health check completed successfully"
        echo "Report saved to: $output_file"

        # Display summary
        if [[ "$output_format" == "json" ]] && command -v jq &> /dev/null; then
            echo -e "\n${CYAN}=== Health Check Summary ===${NC}"
            local overall_status
            overall_status=$(jq -r '.overall_status' "$output_file" 2>/dev/null || echo "unknown")
            echo "Overall Status: $overall_status"

            local healthy_count
            healthy_count=$(jq -r '.metrics.healthy_services' "$output_file" 2>/dev/null || echo "0")
            local total_count
            total_count=$(jq -r '.metrics.total_services' "$output_file" 2>/dev/null || echo "0")
            echo "Services: $healthy_count/$total_count healthy"

            # Show recommendations if any
            local recommendations
            recommendations=$(jq -r '.recommendations[]?' "$output_file" 2>/dev/null || echo "")
            if [[ -n "$recommendations" ]]; then
                echo -e "\n${YELLOW}Recommendations:${NC}"
                echo "$recommendations" | while read -r line; do
                    echo "  - $line"
                done
            fi
        fi

        return 0
    else
        log_error "Health check failed"
        return 1
    fi
}

generate_dashboard() {
    log "Generating health dashboard data..."

    cd "$DOCKER_DIR"

    local dashboard_file="$DASHBOARD_DIR/dashboard-$(date '+%Y%m%d-%H%M%S').json"

    if python3 "$HEALTH_CHECK_SYSTEM" dashboard > "$dashboard_file"; then
        log_success "Dashboard data generated: $dashboard_file"

        # Create symlink to latest
        ln -sf "$(basename "$dashboard_file")" "$DASHBOARD_DIR/latest.json"

        return 0
    else
        log_error "Failed to generate dashboard data"
        return 1
    fi
}

start_monitoring() {
    local interval="${1:-$DEFAULT_INTERVAL}"
    local background="${2:-false}"

    log "Starting health monitoring (interval: ${interval}s)..."

    # Check if already running
    if [[ -f "$MONITOR_PID_FILE" ]] && kill -0 "$(cat "$MONITOR_PID_FILE")" 2>/dev/null; then
        log_warning "Health monitoring is already running (PID: $(cat "$MONITOR_PID_FILE"))"
        return 1
    fi

    cd "$DOCKER_DIR"

    if [[ "$background" == "true" ]]; then
        # Start in background
        nohup python3 "$HEALTH_CHECK_SYSTEM" monitor \
            --interval "$interval" \
            --output-dir "$OUTPUT_DIR" \
            > "$LOGS_DIR/monitor.log" 2>&1 &

        local monitor_pid=$!
        echo "$monitor_pid" > "$MONITOR_PID_FILE"

        log_success "Health monitoring started in background (PID: $monitor_pid)"
        echo "Logs: $LOGS_DIR/monitor.log"
        echo "Stop with: $0 stop-monitoring"
    else
        # Start in foreground
        python3 "$HEALTH_CHECK_SYSTEM" monitor \
            --interval "$interval" \
            --output-dir "$OUTPUT_DIR"
    fi
}

stop_monitoring() {
    log "Stopping health monitoring..."

    if [[ -f "$MONITOR_PID_FILE" ]]; then
        local monitor_pid
        monitor_pid=$(cat "$MONITOR_PID_FILE")

        if kill -0 "$monitor_pid" 2>/dev/null; then
            kill "$monitor_pid"
            rm -f "$MONITOR_PID_FILE"
            log_success "Health monitoring stopped (PID: $monitor_pid)"
        else
            log_warning "Health monitoring process not found (PID: $monitor_pid)"
            rm -f "$MONITOR_PID_FILE"
        fi
    else
        log_warning "Health monitoring is not running"
    fi
}

monitoring_status() {
    if [[ -f "$MONITOR_PID_FILE" ]] && kill -0 "$(cat "$MONITOR_PID_FILE")" 2>/dev/null; then
        local monitor_pid
        monitor_pid=$(cat "$MONITOR_PID_FILE")
        echo -e "${GREEN}Health monitoring is running${NC} (PID: $monitor_pid)"

        # Show recent log entries
        if [[ -f "$LOGS_DIR/monitor.log" ]]; then
            echo -e "\nRecent log entries:"
            tail -5 "$LOGS_DIR/monitor.log"
        fi
    else
        echo -e "${YELLOW}Health monitoring is not running${NC}"
    fi
}

# =============================================================================
# Integration with Docker Compose
# =============================================================================

restart_unhealthy_services() {
    log "Checking for unhealthy services..."

    # Get list of services
    local services
    services=$(docker compose -f "$COMPOSE_FILE" config --services)

    local unhealthy_services=()

    # Check each service
    for service in $services; do
        local status
        status=$(get_service_status "$service")

        if [[ "$status" == "unhealthy" ]]; then
            unhealthy_services+=("$service")
            log_warning "Service $service is unhealthy"
        fi
    done

    if [[ ${#unhealthy_services[@]} -eq 0 ]]; then
        log_success "No unhealthy services found"
        return 0
    fi

    # Restart unhealthy services
    log "Restarting ${#unhealthy_services[@]} unhealthy service(s)..."

    for service in "${unhealthy_services[@]}"; do
        log "Restarting service: $service"
        docker compose -f "$COMPOSE_FILE" restart "$service"
    done

    # Wait for services to become healthy
    for service in "${unhealthy_services[@]}"; do
        wait_for_service_health "$service"
    done

    log_success "Service restart completed"
}

show_service_logs() {
    local service_name="${1:-}"
    local lines="${2:-50}"

    if [[ -z "$service_name" ]]; then
        log "Showing logs for all services..."
        docker compose -f "$COMPOSE_FILE" logs --tail="$lines"
    else
        log "Showing logs for service: $service_name"
        docker compose -f "$COMPOSE_FILE" logs --tail="$lines" "$service_name"
    fi
}

# =============================================================================
# Report and Analytics
# =============================================================================

show_health_summary() {
    log "Generating health summary..."

    echo -e "\n${CYAN}=== CrackSeg Docker Services Health Summary ===${NC}"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"

    # Services status table
    echo -e "\n${CYAN}Service Status:${NC}"
    printf "%-20s %-15s %-15s\n" "Service" "Container Status" "Health Status"
    echo "------------------------------------------------------------"

    local services
    services=$(docker compose -f "$COMPOSE_FILE" config --services)

    local healthy_count=0
    local total_count=0

    for service in $services; do
        local container_status
        container_status=$(get_service_status "$service")

        # Get container name for additional info
        local container_name
        container_name=$(docker compose -f "$COMPOSE_FILE" ps --format json | jq -r "select(.Service == \"$service\") | .Name" 2>/dev/null || echo "N/A")

        # Color code status
        local status_color=""
        case "$container_status" in
            "healthy"|"running")
                status_color="$GREEN"
                healthy_count=$((healthy_count + 1))
                ;;
            "unhealthy")
                status_color="$RED"
                ;;
            "starting")
                status_color="$YELLOW"
                ;;
            *)
                status_color="$NC"
                ;;
        esac

        printf "%-20s ${status_color}%-15s${NC} %-15s\n" "$service" "$container_status" "$container_name"
        total_count=$((total_count + 1))
    done

    echo -e "\n${CYAN}Overall Health:${NC} $healthy_count/$total_count services healthy"

    # Check latest health report if available
    local latest_report
    latest_report=$(find "$OUTPUT_DIR" -name "health-report-*.json" -type f | sort | tail -1)

    if [[ -n "$latest_report" ]] && [[ -f "$latest_report" ]]; then
        echo -e "\n${CYAN}Latest Health Check Report:${NC}"
        echo "File: $latest_report"

        if command -v jq &> /dev/null; then
            local overall_status
            overall_status=$(jq -r '.overall_status' "$latest_report" 2>/dev/null || echo "unknown")
            echo "Overall Status: $overall_status"

            local dependencies_satisfied
            dependencies_satisfied=$(jq -r '.dependencies_satisfied' "$latest_report" 2>/dev/null || echo "unknown")
            echo "Dependencies Satisfied: $dependencies_satisfied"

            # Show recommendations
            local recommendations
            recommendations=$(jq -r '.recommendations[]?' "$latest_report" 2>/dev/null)
            if [[ -n "$recommendations" ]]; then
                echo -e "\n${YELLOW}Recommendations:${NC}"
                echo "$recommendations" | while read -r line; do
                    echo "  - $line"
                done
            fi
        fi
    fi
}

cleanup_old_reports() {
    local days="${1:-7}"

    log "Cleaning up health reports older than $days days..."

    # Clean up old reports
    find "$OUTPUT_DIR" -name "health-report-*.json" -type f -mtime +$days -delete
    find "$DASHBOARD_DIR" -name "dashboard-*.json" -type f -mtime +$days -delete

    # Clean up old logs
    find "$LOGS_DIR" -name "*.log" -type f -mtime +$days -delete

    log_success "Cleanup completed"
}

# =============================================================================
# Main Command Interface
# =============================================================================

show_usage() {
    cat << EOF
CrackSeg Health Check Manager

USAGE:
    $0 [COMMAND] [OPTIONS]

COMMANDS:
    check [format] [file]       Run one-time health check
    monitor [interval]          Start continuous monitoring
    stop-monitoring            Stop continuous monitoring
    status                     Show monitoring status
    dashboard                  Generate dashboard data
    summary                    Show health summary
    restart-unhealthy          Restart unhealthy services
    wait [timeout]             Wait for all services to be healthy
    logs [service] [lines]     Show service logs
    cleanup [days]             Clean up old reports

EXAMPLES:
    $0 check                   # Run health check with JSON output
    $0 check dashboard         # Run health check with dashboard format
    $0 monitor 60              # Start monitoring with 60s interval
    $0 wait 120                # Wait up to 120s for services to be healthy
    $0 logs streamlit-app 100  # Show last 100 lines of streamlit logs
    $0 restart-unhealthy       # Restart any unhealthy services
    $0 cleanup 3               # Remove reports older than 3 days

OUTPUT:
    Reports: $OUTPUT_DIR
    Logs: $LOGS_DIR
    Dashboard: $DASHBOARD_DIR

EOF
}

main() {
    # Setup
    setup_directories
    validate_environment

    # Parse command
    local command="${1:-check}"

    case "$command" in
        "check")
            local format="${2:-json}"
            local file="${3:-}"
            run_health_check "$format" "$file"
            ;;
        "monitor")
            local interval="${2:-$DEFAULT_INTERVAL}"
            start_monitoring "$interval" "true"
            ;;
        "monitor-fg")
            local interval="${2:-$DEFAULT_INTERVAL}"
            start_monitoring "$interval" "false"
            ;;
        "stop-monitoring")
            stop_monitoring
            ;;
        "status")
            monitoring_status
            ;;
        "dashboard")
            generate_dashboard
            ;;
        "summary")
            show_health_summary
            ;;
        "restart-unhealthy")
            restart_unhealthy_services
            ;;
        "wait")
            local timeout="${2:-$DEFAULT_TIMEOUT}"
            wait_for_all_services "$timeout"
            ;;
        "logs")
            local service="${2:-}"
            local lines="${3:-50}"
            show_service_logs "$service" "$lines"
            ;;
        "cleanup")
            local days="${2:-7}"
            cleanup_old_reports "$days"
            ;;
        "help"|"--help"|"-h")
            show_usage
            ;;
        *)
            log_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"
