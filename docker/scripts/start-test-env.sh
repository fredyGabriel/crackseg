#!/bin/bash

# =============================================================================
# CrackSeg Docker Testing - Environment Startup Script
# =============================================================================
# Purpose: Start Docker Compose testing environment with proper sequence
# Usage: ./start-test-env.sh [options]
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
MODE="standard"
PROFILE=""
DETACH=false
BUILD=false
PULL=false
VERBOSE=false
TIMEOUT=300

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
CrackSeg Docker Testing Environment Startup Script

Usage: $0 [OPTIONS]

OPTIONS:
    -m, --mode MODE         Test mode: standard, debug, recording (default: standard)
    -d, --detach           Run in detached mode
    -b, --build            Force rebuild of images
    -p, --pull             Pull latest images before starting
    -v, --verbose          Enable verbose logging
    -t, --timeout SECONDS  Timeout for service startup (default: 300)
    -h, --help             Show this help message

MODES:
    standard    - Basic testing environment (default)
    debug       - Includes noVNC for browser debugging
    recording   - Includes video recording of browser sessions
    minimal     - Only hub and single Chrome node

EXAMPLES:
    $0                                  # Start standard environment
    $0 -m debug -d                     # Start debug mode in background
    $0 -m recording -b -v              # Start with video recording, rebuild images
    $0 --mode minimal --timeout 120    # Quick minimal setup

SERVICES:
    Standard Mode: selenium-hub, chrome-node, firefox-node, streamlit-app
    Debug Mode:    + noVNC (http://localhost:7900)
    Recording Mode: + video-recorder
    Minimal Mode:  selenium-hub, chrome-node, streamlit-app

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

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi

    # Check compose file exists
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

build_images() {
    if [[ "$BUILD" == true ]]; then
        log "Building Docker images..."
        cd "$PROJECT_ROOT"
        docker compose -f "$COMPOSE_FILE" build --no-cache
        log_success "Images built successfully"
    fi
}

pull_images() {
    if [[ "$PULL" == true ]]; then
        log "Pulling latest images..."
        cd "$PROJECT_ROOT"
        docker compose -f "$COMPOSE_FILE" pull
        log_success "Images pulled successfully"
    fi
}

start_services() {
    log "Starting Docker Compose services..."
    cd "$PROJECT_ROOT"

    local compose_args=()
    compose_args+=("-f" "$COMPOSE_FILE")

    # Add profile if specified
    if [[ -n "$PROFILE" ]]; then
        compose_args+=("--profile" "$PROFILE")
    fi

    # Build compose up command
    local up_args=("up")

    if [[ "$DETACH" == true ]]; then
        up_args+=("-d")
    fi

    if [[ "$VERBOSE" == true ]]; then
        up_args+=("--verbose")
    fi

    # Start services
    docker compose "${compose_args[@]}" "${up_args[@]}"

    if [[ "$DETACH" == true ]]; then
        log_success "Services started in detached mode"
    else
        log_success "Services started"
    fi
}

wait_for_services() {
    if [[ "$DETACH" == true ]]; then
        log "Waiting for services to be healthy..."

        local start_time=$(date +%s)
        local max_wait=$TIMEOUT

        # Services to check
        local services=("crackseg-selenium-hub" "crackseg-streamlit-app")

        for service in "${services[@]}"; do
            log "Waiting for $service to be healthy..."

            while true; do
                local current_time=$(date +%s)
                local elapsed=$((current_time - start_time))

                if [[ $elapsed -gt $max_wait ]]; then
                    log_error "Timeout waiting for $service (${max_wait}s)"
                    return 1
                fi

                # Check if container is healthy
                local health_status
                health_status=$(docker inspect --format='{{.State.Health.Status}}' "$service" 2>/dev/null || echo "unknown")

                case "$health_status" in
                    "healthy")
                        log_success "$service is healthy"
                        break
                        ;;
                    "unhealthy")
                        log_error "$service is unhealthy"
                        return 1
                        ;;
                    "starting"|"unknown")
                        echo -n "."
                        sleep 2
                        ;;
                    *)
                        log_warning "$service health status: $health_status"
                        sleep 2
                        ;;
                esac
            done
        done

        log_success "All services are healthy and ready"
    fi
}

show_service_info() {
    if [[ "$DETACH" == true ]]; then
        log "Service information:"
        echo
        echo "🌐 Streamlit Application: http://localhost:8501"
        echo "🕷️  Selenium Grid Console: http://localhost:4444/grid/console"

        if [[ "$MODE" == "debug" ]]; then
            echo "🖥️  noVNC Browser Access: http://localhost:7900"
        fi

        echo
        echo "📊 View running containers:"
        echo "   docker compose -f $COMPOSE_FILE ps"
        echo
        echo "📋 View logs:"
        echo "   docker compose -f $COMPOSE_FILE logs -f [service_name]"
        echo
        echo "🛑 Stop environment:"
        echo "   docker compose -f $COMPOSE_FILE down"
        echo
    fi
}

cleanup_on_exit() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Script failed with exit code $exit_code"
        log "Cleaning up any started containers..."
        cd "$PROJECT_ROOT"
        docker compose -f "$COMPOSE_FILE" down --remove-orphans &> /dev/null || true
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
            -m|--mode)
                MODE="$2"
                shift 2
                ;;
            -d|--detach)
                DETACH=true
                shift
                ;;
            -b|--build)
                BUILD=true
                shift
                ;;
            -p|--pull)
                PULL=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -t|--timeout)
                TIMEOUT="$2"
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

    # Validate mode and set profile
    case "$MODE" in
        standard|minimal)
            PROFILE=""
            ;;
        debug)
            PROFILE="debug"
            ;;
        recording)
            PROFILE="recording"
            ;;
        *)
            log_error "Invalid mode: $MODE"
            show_help
            exit 1
            ;;
    esac

    # Set up exit trap
    trap cleanup_on_exit EXIT

    # Execute startup sequence
    log "Starting CrackSeg Docker testing environment..."
    log "Mode: $MODE"
    if [[ -n "$PROFILE" ]]; then
        log "Profile: $PROFILE"
    fi

    check_prerequisites
    build_images
    pull_images
    start_services
    wait_for_services
    show_service_info

    log_success "Environment startup completed successfully!"
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi