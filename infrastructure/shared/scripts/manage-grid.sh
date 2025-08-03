#!/bin/bash

# =============================================================================
# CrackSeg Selenium Grid Management Script
# =============================================================================
# Purpose: Advanced management and monitoring of Selenium Grid infrastructure
# Usage: ./manage-grid.sh [command] [options]
# Author: CrackSeg Development Team
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration and Constants
# =============================================================================
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
readonly DOCKER_DIR="${PROJECT_ROOT}/infrastructure/testing"
readonly COMPOSE_FILE="${DOCKER_DIR}/docker-compose.test.yml"
readonly GRID_CONFIG="${DOCKER_DIR}/grid-config.json"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Grid configuration
readonly HUB_URL="http://localhost:4444"
readonly GRID_CONSOLE_URL="http://localhost:4446"
readonly GRID_STATUS_ENDPOINT="${HUB_URL}/wd/hub/status"

# =============================================================================
# Utility Functions
# =============================================================================

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ${1}${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] ${1}${NC}" >&2
}

error() {
    echo -e "${RED}[ERROR] ${1}${NC}" >&2
}

info() {
    echo -e "${BLUE}[INFO] ${1}${NC}"
}

debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${PURPLE}[DEBUG] ${1}${NC}" >&2
    fi
}

# =============================================================================
# Docker Compose Helper Functions
# =============================================================================

compose_cmd() {
    docker-compose -f "${COMPOSE_FILE}" "$@"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
}

# =============================================================================
# Grid Status and Monitoring Functions
# =============================================================================

get_grid_status() {
    local max_retries=5
    local retry_delay=2

    for i in $(seq 1 $max_retries); do
        if curl -s -f "${GRID_STATUS_ENDPOINT}" > /dev/null 2>&1; then
            return 0
        fi
        debug "Grid status check attempt ${i}/${max_retries} failed, retrying in ${retry_delay}s..."
        sleep $retry_delay
    done

    return 1
}

show_grid_status() {
    log "Checking Selenium Grid Status..."

    if ! get_grid_status; then
        error "Selenium Grid is not accessible at ${HUB_URL}"
        return 1
    fi

    local status_json
    status_json=$(curl -s "${GRID_STATUS_ENDPOINT}")

    echo -e "\n${CYAN}=== Selenium Grid Status ===${NC}"
    echo "Hub URL: ${HUB_URL}"
    echo "Grid Console: ${GRID_CONSOLE_URL}"
    echo "Status: $(echo "$status_json" | jq -r '.value.ready // "unknown"')"

    local nodes_count
    nodes_count=$(echo "$status_json" | jq -r '.value.nodes | length')
    echo "Connected Nodes: ${nodes_count}"

    if [[ "$nodes_count" -gt 0 ]]; then
        echo -e "\n${CYAN}=== Node Details ===${NC}"
        echo "$status_json" | jq -r '.value.nodes[] | "Browser: \(.stereotypes[0].browserName // "unknown") | Max Sessions: \(.maxSessions) | Status: \(.availability)"'
    fi

    # Show session information
    local sessions_json
    if sessions_json=$(curl -s "${HUB_URL}/wd/hub/sessions" 2>/dev/null); then
        local active_sessions
        active_sessions=$(echo "$sessions_json" | jq -r '.value | length')
        echo -e "\nActive Sessions: ${active_sessions}"
    fi
}

monitor_grid() {
    local interval="${1:-10}"

    log "Starting Grid monitoring (refresh every ${interval}s). Press Ctrl+C to stop."

    while true; do
        clear
        show_grid_status
        echo -e "\n${YELLOW}Last updated: $(date)${NC}"
        echo "Monitoring interval: ${interval}s"
        sleep "$interval"
    done
}

# =============================================================================
# Grid Lifecycle Management
# =============================================================================

start_grid() {
    local profile="${1:-""}"
    local services="${2:-""}"

    log "Starting Selenium Grid infrastructure..."
    check_docker

    # Build images if needed
    if [[ ! -f "${DOCKER_DIR}/.build_complete" ]]; then
        log "Building Docker images (first-time setup)..."
        compose_cmd build
        touch "${DOCKER_DIR}/.build_complete"
    fi

    # Start core services
    local cmd_args=("up" "-d")

    if [[ -n "$profile" ]]; then
        cmd_args+=("--profile" "$profile")
    fi

    if [[ -n "$services" ]]; then
        cmd_args+=($services)
    else
        cmd_args+=("selenium-hub" "chrome-node" "firefox-node" "streamlit-app")
    fi

    compose_cmd "${cmd_args[@]}"

    # Wait for grid to be ready
    log "Waiting for Selenium Grid to become ready..."
    local timeout=120
    local elapsed=0

    while ! get_grid_status && [[ $elapsed -lt $timeout ]]; do
        sleep 5
        elapsed=$((elapsed + 5))
        echo -n "."
    done
    echo

    if get_grid_status; then
        log "Selenium Grid is ready!"
        show_grid_status
    else
        error "Selenium Grid failed to start within ${timeout} seconds"
        return 1
    fi
}

stop_grid() {
    log "Stopping Selenium Grid infrastructure..."
    compose_cmd down
    log "Grid stopped successfully"
}

restart_grid() {
    log "Restarting Selenium Grid infrastructure..."
    stop_grid
    start_grid "$@"
}

scale_nodes() {
    local browser="$1"
    local count="$2"

    if [[ ! "$browser" =~ ^(chrome|firefox|edge)$ ]]; then
        error "Invalid browser: $browser. Must be chrome, firefox, or edge"
        return 1
    fi

    if [[ ! "$count" =~ ^[0-9]+$ ]] || [[ "$count" -lt 0 ]] || [[ "$count" -gt 10 ]]; then
        error "Invalid count: $count. Must be between 0 and 10"
        return 1
    fi

    log "Scaling ${browser} nodes to ${count} instances..."
    compose_cmd up -d --scale "${browser}-node=${count}"

    sleep 10
    show_grid_status
}

# =============================================================================
# Testing and Validation Functions
# =============================================================================

validate_grid() {
    log "Validating Selenium Grid configuration..."

    # Check if grid is accessible
    if ! get_grid_status; then
        error "Grid validation failed: Hub not accessible"
        return 1
    fi

    # Test browser connectivity
    local browsers=("chrome" "firefox")
    local test_script="${DOCKER_DIR}/scripts/test-browser.py"

    for browser in "${browsers[@]}"; do
        info "Testing ${browser} connectivity..."

        if python3 -c "
import sys
sys.path.append('${PROJECT_ROOT}')
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

options = webdriver.${browser^}Options()
options.add_argument('--headless')

caps = DesiredCapabilities.${browser^^}.copy()
caps['se:screenResolution'] = '1920x1080x24'

try:
    driver = webdriver.Remote(
        command_executor='${HUB_URL}/wd/hub',
        desired_capabilities=caps,
        options=options
    )
    driver.get('http://httpbin.org/html')
    assert 'Herman Melville' in driver.page_source
    driver.quit()
    print('✓ ${browser} test passed')
except Exception as e:
    print(f'✗ ${browser} test failed: {e}')
    sys.exit(1)
        "; then
            log "${browser} connectivity test passed"
        else
            error "${browser} connectivity test failed"
            return 1
        fi
    done

    log "Grid validation completed successfully"
}

# =============================================================================
# Cleanup and Maintenance Functions
# =============================================================================

cleanup_grid() {
    log "Cleaning up Selenium Grid resources..."

    # Stop all services
    compose_cmd down -v --remove-orphans

    # Remove unused networks
    docker network prune -f 2>/dev/null || true

    # Clean up build cache
    rm -f "${DOCKER_DIR}/.build_complete"

    log "Grid cleanup completed"
}

show_logs() {
    local service="${1:-""}"
    local follow="${2:-false}"

    if [[ -n "$service" ]]; then
        if [[ "$follow" == "true" ]]; then
            compose_cmd logs -f "$service"
        else
            compose_cmd logs "$service"
        fi
    else
        if [[ "$follow" == "true" ]]; then
            compose_cmd logs -f
        else
            compose_cmd logs
        fi
    fi
}

# =============================================================================
# Help and Usage Functions
# =============================================================================

show_help() {
    cat << EOF
${CYAN}CrackSeg Selenium Grid Management Script${NC}

${YELLOW}USAGE:${NC}
    ./manage-grid.sh [command] [options]

${YELLOW}COMMANDS:${NC}
    ${GREEN}start [profile] [services]${NC}    Start the grid infrastructure
    ${GREEN}stop${NC}                         Stop the grid infrastructure
    ${GREEN}restart [profile] [services]${NC} Restart the grid infrastructure
    ${GREEN}status${NC}                      Show current grid status
    ${GREEN}monitor [interval]${NC}          Monitor grid status (default: 10s)
    ${GREEN}scale <browser> <count>${NC}     Scale browser nodes (0-10)
    ${GREEN}validate${NC}                   Validate grid configuration
    ${GREEN}logs [service] [--follow]${NC}   Show logs for service(s)
    ${GREEN}cleanup${NC}                    Clean up all grid resources
    ${GREEN}help${NC}                       Show this help message

${YELLOW}PROFILES:${NC}
    ${BLUE}debug${NC}        Start with noVNC debugging enabled
    ${BLUE}recording${NC}    Start with video recording enabled
    ${BLUE}monitoring${NC}   Start with grid console monitoring
    ${BLUE}edge${NC}         Include Microsoft Edge nodes

${YELLOW}BROWSERS:${NC}
    chrome, firefox, edge

${YELLOW}EXAMPLES:${NC}
    ./manage-grid.sh start                    # Start basic grid
    ./manage-grid.sh start debug              # Start with debugging
    ./manage-grid.sh scale chrome 3           # Scale to 3 Chrome nodes
    ./manage-grid.sh monitor 5                # Monitor every 5 seconds
    ./manage-grid.sh logs selenium-hub        # Show hub logs
    ./manage-grid.sh validate                 # Test grid connectivity

${YELLOW}MONITORING URLS:${NC}
    Grid Hub:     ${HUB_URL}
    Grid Console: ${GRID_CONSOLE_URL}
    noVNC (Chrome): http://localhost:7900
    noVNC (Firefox): http://localhost:7901

EOF
}

# =============================================================================
# Main Function and Command Routing
# =============================================================================

main() {
    local command="${1:-help}"

    # Set debug mode if requested
    if [[ "${DEBUG:-false}" == "true" ]]; then
        set -x
    fi

    case "$command" in
        "start")
            start_grid "${2:-}" "${3:-}"
            ;;
        "stop")
            stop_grid
            ;;
        "restart")
            restart_grid "${2:-}" "${3:-}"
            ;;
        "status")
            show_grid_status
            ;;
        "monitor")
            monitor_grid "${2:-10}"
            ;;
        "scale")
            if [[ $# -lt 3 ]]; then
                error "Scale command requires browser and count arguments"
                echo "Usage: $0 scale <browser> <count>"
                exit 1
            fi
            scale_nodes "$2" "$3"
            ;;
        "validate")
            validate_grid
            ;;
        "logs")
            local follow=false
            if [[ "${3:-}" == "--follow" ]]; then
                follow=true
            fi
            show_logs "${2:-}" "$follow"
            ;;
        "cleanup")
            cleanup_grid
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            error "Unknown command: $command"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# =============================================================================
# Script Entry Point
# =============================================================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi