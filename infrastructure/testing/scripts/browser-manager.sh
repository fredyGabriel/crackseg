#!/usr/bin/env bash
# =============================================================================
# CrackSeg Browser Manager - Dynamic Browser Container Management
# =============================================================================
# Purpose: Dynamic browser container management for cross-browser testing
# Version: 1.0 (Subtask 13.10)
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

# Configuration files
CAPABILITIES_FILE="$DOCKER_DIR/browser-capabilities.json"
COMPOSE_FILE="$DOCKER_DIR/docker-compose.test.yml"
COMPOSE_BROWSERS_FILE="$DOCKER_DIR/docker-compose.browsers.yml"

# Default configuration
DEFAULT_BROWSER="chrome:latest"
DEFAULT_NETWORK="crackseg-backend-network"
DEFAULT_SELENIUM_HUB="selenium-hub:4444"
DEFAULT_BASE_PORT=5555

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
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] [BROWSER-MANAGER]${NC} $*"
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
    echo "  CrackSeg Browser Manager - Dynamic Cross-Browser Testing (Subtask 13.10)"
    echo "  Multi-Version Browser Container Management with Mobile Emulation"
    echo "============================================================================="
    echo -e "${NC}"
}

# =============================================================================
# Validation Functions
# =============================================================================

validate_environment() {
    log_step "Validating browser management environment..."

    # Check required commands
    local required_commands=("docker" "docker-compose" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "Required command not found: $cmd"
            exit 1
        fi
    done

    # Check capabilities file
    if [[ ! -f "$CAPABILITIES_FILE" ]]; then
        log_error "Browser capabilities file not found: $CAPABILITIES_FILE"
        exit 1
    fi

    # Validate JSON syntax
    if ! jq empty "$CAPABILITIES_FILE" 2>/dev/null; then
        log_error "Invalid JSON syntax in capabilities file"
        exit 1
    fi

    log_success "Environment validation completed"
}

# =============================================================================
# Browser Configuration Management
# =============================================================================

parse_browser_spec() {
    local browser_spec="$1"

    if [[ "$browser_spec" == *":"* ]]; then
        local browser_name="${browser_spec%:*}"
        local browser_variant="${browser_spec#*:}"
    else
        local browser_name="$browser_spec"
        local browser_variant="latest"
    fi

    echo "$browser_name" "$browser_variant"
}

get_browser_config() {
    local browser_name="$1"
    local variant="$2"

    # Check if it's a mobile device configuration
    local mobile_config
    mobile_config=$(jq -r ".browser_capabilities.${browser_name}.mobile_devices.${variant} // empty" "$CAPABILITIES_FILE")

    if [[ -n "$mobile_config" && "$mobile_config" != "null" ]]; then
        echo "$mobile_config"
        return 0
    fi

    # Check if it's a mobile emulation configuration
    local mobile_emulation_config
    mobile_emulation_config=$(jq -r ".browser_capabilities.${browser_name}.mobile_emulation.${variant} // empty" "$CAPABILITIES_FILE")

    if [[ -n "$mobile_emulation_config" && "$mobile_emulation_config" != "null" ]]; then
        echo "$mobile_emulation_config"
        return 0
    fi

    # Check regular version configuration
    local version_config
    version_config=$(jq -r ".browser_capabilities.${browser_name}.versions.${variant} // empty" "$CAPABILITIES_FILE")

    if [[ -n "$version_config" && "$version_config" != "null" ]]; then
        echo "$version_config"
        return 0
    fi

    # Browser configuration not found
    return 1
}

generate_container_name() {
    local browser_name="$1"
    local variant="$2"
    local instance_id="${3:-1}"

    echo "crackseg-${browser_name}-${variant}-${instance_id}"
}

generate_hostname() {
    local browser_name="$1"
    local variant="$2"
    local instance_id="${3:-1}"

    echo "${browser_name}-${variant}-${instance_id}"
}

calculate_port() {
    local base_port="$1"
    local instance_id="$2"

    echo $((base_port + instance_id))
}

# =============================================================================
# Dynamic Container Creation
# =============================================================================

create_browser_container() {
    local browser_spec="$1"
    local instance_id="${2:-1}"

    log_step "Creating dynamic browser container: $browser_spec (instance $instance_id)"

    # Parse browser specification
    read -r browser_name variant <<< "$(parse_browser_spec "$browser_spec")"

    # Get browser configuration
    local config
    if ! config=$(get_browser_config "$browser_name" "$variant"); then
        log_error "Browser configuration not found: $browser_name:$variant"
        return 1
    fi

    # Extract configuration values
    local image
    image=$(echo "$config" | jq -r '.image')
    local capabilities
    capabilities=$(echo "$config" | jq -c '.capabilities')
    local resources
    resources=$(echo "$config" | jq -c '.resources')
    local memory_limit
    memory_limit=$(echo "$resources" | jq -r '.memory')
    local cpu_limit
    cpu_limit=$(echo "$resources" | jq -r '.cpus')
    local is_mobile
    is_mobile=$(echo "$config" | jq -r '.mobile_emulation // false')

    # Generate container details
    local container_name
    container_name=$(generate_container_name "$browser_name" "$variant" "$instance_id")
    local hostname
    hostname=$(generate_hostname "$browser_name" "$variant" "$instance_id")
    local node_port
    node_port=$(calculate_port $DEFAULT_BASE_PORT "$instance_id")

    log_info "Container: $container_name"
    log_info "Image: $image"
    log_info "Port: $node_port"
    log_info "Mobile: $is_mobile"

    # Create and run container
    local docker_args=(
        "run" "-d"
        "--name" "$container_name"
        "--hostname" "$hostname"
        "--network" "$DEFAULT_NETWORK"
        "-p" "${node_port}:5555"
        "-v" "/dev/shm:/dev/shm"
        "-v" "selenium-videos:/videos"
        "--memory" "$memory_limit"
        "--cpus" "$cpu_limit"
        "--restart" "unless-stopped"
    )

    # Add environment variables
    docker_args+=(
        "-e" "SE_EVENT_BUS_HOST=selenium-hub"
        "-e" "SE_EVENT_BUS_PUBLISH_PORT=4442"
        "-e" "SE_EVENT_BUS_SUBSCRIBE_PORT=4443"
        "-e" "SE_HUB_HOST=selenium-hub"
        "-e" "SE_HUB_PORT=4444"
        "-e" "SE_NODE_HOST=$hostname"
        "-e" "SE_NODE_PORT=5555"
        "-e" "SE_NODE_MAX_INSTANCES=2"
        "-e" "SE_NODE_MAX_SESSIONS=2"
        "-e" "SE_NODE_OVERRIDE_MAX_SESSIONS=true"
        "-e" "SE_NODE_SESSION_TIMEOUT=300"
        "-e" "SE_VNC_NO_PASSWORD=1"
        "-e" "SE_VNC_PORT=5900"
        "-e" "SE_SCREEN_WIDTH=1920"
        "-e" "SE_SCREEN_HEIGHT=1080"
        "-e" "SE_SCREEN_DEPTH=24"
        "-e" "SE_START_XVFB=true"
        "-e" "SE_NODE_GRID_URL=http://selenium-hub:4444/"
        "-e" "SE_LOG_LEVEL=INFO"
    )

    # Add labels
    docker_args+=(
        "--label" "crackseg.service=browser-node"
        "--label" "crackseg.environment=test"
        "--label" "crackseg.browser=$browser_name"
        "--label" "crackseg.variant=$variant"
        "--label" "crackseg.instance=$instance_id"
        "--label" "crackseg.mobile=$is_mobile"
        "--label" "crackseg.dynamic=true"
    )

    # Add image
    docker_args+=("$image")

    # Create container
    if docker "${docker_args[@]}"; then
        log_success "Browser container created: $container_name"

        # Wait for container to be ready
        log_info "Waiting for browser node to register with Selenium Hub..."
        local max_attempts=30
        local attempt=1

        while [[ $attempt -le $max_attempts ]]; do
            if check_node_registration "$hostname"; then
                log_success "Browser node registered successfully"
                return 0
            fi

            log_info "Attempt $attempt/$max_attempts - waiting for registration..."
            sleep 2
            ((attempt++))
        done

        log_warning "Browser node may not have registered properly within timeout"
        return 0
    else
        log_error "Failed to create browser container: $container_name"
        return 1
    fi
}

check_node_registration() {
    local hostname="$1"

    # Check if node is registered with Selenium Hub
    local hub_status
    if hub_status=$(curl -s "http://localhost:4444/grid/api/hub/status" 2>/dev/null); then
        if echo "$hub_status" | jq -e ".value.ready // false" >/dev/null; then
            if curl -s "http://localhost:4444/grid/api/hub" | jq -e ".value.nodes[] | select(.uri | contains(\"$hostname\"))" >/dev/null 2>&1; then
                return 0
            fi
        fi
    fi

    return 1
}

# =============================================================================
# Browser Matrix Management
# =============================================================================

create_browser_matrix() {
    local matrix_name="${1:-smoke_test}"
    local parallel="${2:-true}"

    log_step "Creating browser matrix: $matrix_name"

    # Get matrix configuration
    local matrix_config
    if ! matrix_config=$(jq -r ".test_matrices.${matrix_name} // empty" "$CAPABILITIES_FILE"); then
        log_error "Matrix configuration not found: $matrix_name"
        return 1
    fi

    if [[ -z "$matrix_config" || "$matrix_config" == "null" ]]; then
        log_error "Matrix configuration not found: $matrix_name"
        return 1
    fi

    # Extract browser list
    local browsers
    browsers=$(echo "$matrix_config" | jq -r '.browsers[]')

    log_info "Matrix browsers: $(echo "$browsers" | tr '\n' ' ')"

    # Create containers for each browser
    local instance_id=1
    local container_names=()

    while IFS= read -r browser_spec; do
        if [[ -n "$browser_spec" ]]; then
            log_info "Creating browser: $browser_spec (instance $instance_id)"

            if create_browser_container "$browser_spec" "$instance_id"; then
                read -r browser_name variant <<< "$(parse_browser_spec "$browser_spec")"
                local container_name
                container_name=$(generate_container_name "$browser_name" "$variant" "$instance_id")
                container_names+=("$container_name")
                log_success "Browser created: $browser_spec"
            else
                log_error "Failed to create browser: $browser_spec"
            fi

            ((instance_id++))

            # Add delay between container creation to prevent resource conflicts
            if [[ "$parallel" == "true" ]]; then
                sleep 1
            else
                sleep 3
            fi
        fi
    done <<< "$browsers"

    log_success "Browser matrix created: $matrix_name"
    log_info "Created containers: ${container_names[*]}"

    return 0
}

# =============================================================================
# Container Lifecycle Management
# =============================================================================

list_dynamic_browsers() {
    log_step "Listing dynamic browser containers..."

    local containers
    containers=$(docker ps -a --filter "label=crackseg.dynamic=true" --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Labels}}")

    if [[ -n "$containers" && "$containers" != *"NAMES"* ]]; then
        echo "$containers"
    else
        log_info "No dynamic browser containers found"
    fi
}

stop_dynamic_browsers() {
    log_step "Stopping dynamic browser containers..."

    local containers
    containers=$(docker ps --filter "label=crackseg.dynamic=true" --format "{{.Names}}")

    if [[ -n "$containers" ]]; then
        log_info "Stopping containers: $(echo "$containers" | tr '\n' ' ')"
        echo "$containers" | xargs -r docker stop
        log_success "Dynamic browser containers stopped"
    else
        log_info "No running dynamic browser containers found"
    fi
}

remove_dynamic_browsers() {
    log_step "Removing dynamic browser containers..."

    local containers
    containers=$(docker ps -a --filter "label=crackseg.dynamic=true" --format "{{.Names}}")

    if [[ -n "$containers" ]]; then
        log_info "Removing containers: $(echo "$containers" | tr '\n' ' ')"
        echo "$containers" | xargs -r docker rm -f
        log_success "Dynamic browser containers removed"
    else
        log_info "No dynamic browser containers found"
    fi
}

cleanup_dynamic_browsers() {
    log_step "Cleaning up all dynamic browser containers..."

    # First stop running containers
    stop_dynamic_browsers

    # Then remove all containers
    remove_dynamic_browsers

    log_success "Dynamic browser cleanup completed"
}

# =============================================================================
# Capability Management
# =============================================================================

list_available_browsers() {
    log_step "Available browser configurations:"

    echo ""
    echo "Desktop Browsers:"
    echo "================"

    # List desktop browser versions
    local browsers=("chrome" "firefox" "edge" "safari")
    for browser in "${browsers[@]}"; do
        local versions
        versions=$(jq -r ".browser_capabilities.${browser}.versions // {} | keys[]" "$CAPABILITIES_FILE" 2>/dev/null || echo "")
        if [[ -n "$versions" ]]; then
            echo "  $browser: $(echo "$versions" | tr '\n' ' ')"
        fi
    done

    echo ""
    echo "Mobile Emulation:"
    echo "================"

    # List mobile configurations
    for browser in "${browsers[@]}"; do
        local mobile_devices
        mobile_devices=$(jq -r ".browser_capabilities.${browser}.mobile_devices // {} | keys[]" "$CAPABILITIES_FILE" 2>/dev/null || echo "")
        if [[ -n "$mobile_devices" ]]; then
            echo "  $browser mobile devices: $(echo "$mobile_devices" | tr '\n' ' ')"
        fi

        local mobile_emulation
        mobile_emulation=$(jq -r ".browser_capabilities.${browser}.mobile_emulation // {} | keys[]" "$CAPABILITIES_FILE" 2>/dev/null || echo "")
        if [[ -n "$mobile_emulation" ]]; then
            echo "  $browser mobile emulation: $(echo "$mobile_emulation" | tr '\n' ' ')"
        fi
    done

    echo ""
    echo "Test Matrices:"
    echo "============="

    local matrices
    matrices=$(jq -r '.test_matrices | keys[]' "$CAPABILITIES_FILE")
    for matrix in $matrices; do
        local matrix_browsers
        matrix_browsers=$(jq -r ".test_matrices.${matrix}.browsers[]" "$CAPABILITIES_FILE" | tr '\n' ' ')
        echo "  $matrix: $matrix_browsers"
    done
}

show_browser_config() {
    local browser_spec="$1"

    read -r browser_name variant <<< "$(parse_browser_spec "$browser_spec")"

    log_step "Configuration for $browser_name:$variant"

    local config
    if config=$(get_browser_config "$browser_name" "$variant"); then
        echo "$config" | jq .
    else
        log_error "Browser configuration not found: $browser_name:$variant"
        return 1
    fi
}

# =============================================================================
# Main Command Interface
# =============================================================================

show_usage() {
    cat << EOF
Usage: $0 <command> [options]

Commands:
  create <browser_spec> [instance_id]    Create a dynamic browser container
  matrix <matrix_name> [parallel]        Create browser matrix from configuration
  list                                   List all dynamic browser containers
  stop                                   Stop all dynamic browser containers
  remove                                 Remove all dynamic browser containers
  cleanup                                Stop and remove all dynamic containers
  browsers                               List available browser configurations
  config <browser_spec>                  Show configuration for specific browser
  validate                               Validate environment and capabilities

Browser Specifications:
  chrome:latest, firefox:stable, edge:beta
  chrome:pixel_5, firefox:responsive, edge:surface_duo

Matrix Names:
  smoke_test, compatibility_test, mobile_test, full_matrix

Examples:
  $0 create chrome:latest
  $0 create firefox:pixel_5 2
  $0 matrix smoke_test
  $0 matrix mobile_test true
  $0 config chrome:beta
  $0 cleanup

Environment Variables:
  CAPABILITIES_FILE    Path to browser capabilities JSON file
  DEFAULT_NETWORK      Docker network for browser containers
  DEFAULT_BASE_PORT    Base port for browser node services

EOF
}

main() {
    local command="${1:-help}"

    case "$command" in
        "create")
            if [[ $# -lt 2 ]]; then
                log_error "Browser specification required"
                show_usage
                exit 1
            fi
            validate_environment
            create_browser_container "$2" "${3:-1}"
            ;;
        "matrix")
            if [[ $# -lt 2 ]]; then
                log_error "Matrix name required"
                show_usage
                exit 1
            fi
            validate_environment
            create_browser_matrix "$2" "${3:-true}"
            ;;
        "list")
            list_dynamic_browsers
            ;;
        "stop")
            stop_dynamic_browsers
            ;;
        "remove")
            remove_dynamic_browsers
            ;;
        "cleanup")
            cleanup_dynamic_browsers
            ;;
        "browsers")
            list_available_browsers
            ;;
        "config")
            if [[ $# -lt 2 ]]; then
                log_error "Browser specification required"
                show_usage
                exit 1
            fi
            validate_environment
            show_browser_config "$2"
            ;;
        "validate")
            validate_environment
            log_success "Environment validation passed"
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

# =============================================================================
# Script Entry Point
# =============================================================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi