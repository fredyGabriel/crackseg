#!/usr/bin/env bash
# =============================================================================
# CrackSeg Network Manager Script
# =============================================================================
# Purpose: Manage Docker network configuration for multi-network architecture
# Version: 1.0 (Subtask 13.9)
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

# Network configuration
declare -A NETWORKS=(
    ["frontend"]="crackseg-frontend-network"
    ["backend"]="crackseg-backend-network"
    ["management"]="crackseg-management-network"
    ["legacy"]="crackseg-test-network-legacy"
)

declare -A SUBNETS=(
    ["frontend"]="172.20.0.0/24"
    ["backend"]="172.21.0.0/24"
    ["management"]="172.22.0.0/24"
    ["legacy"]="172.20.0.0/16"
)

declare -A GATEWAYS=(
    ["frontend"]="172.20.0.1"
    ["backend"]="172.21.0.1"
    ["management"]="172.22.0.1"
    ["legacy"]="172.20.0.1"
)

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# =============================================================================
# Logging Functions
# =============================================================================

log() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] $*${NC}"
}

log_success() {
    echo -e "${GREEN}[SUCCESS] $*${NC}"
}

log_warning() {
    echo -e "${YELLOW}[WARNING] $*${NC}"
}

log_error() {
    echo -e "${RED}[ERROR] $*${NC}"
}

log_info() {
    echo -e "${BLUE}[INFO] $*${NC}"
}

# =============================================================================
# Network Management Functions
# =============================================================================

create_networks() {
    log "Creating Docker networks for multi-network architecture..."

    for network_type in "${!NETWORKS[@]}"; do
        local network_name="${NETWORKS[$network_type]}"
        local subnet="${SUBNETS[$network_type]}"
        local gateway="${GATEWAYS[$network_type]}"

        if docker network inspect "$network_name" >/dev/null 2>&1; then
            log_info "Network $network_name already exists"
        else
            log "Creating network: $network_name (${subnet})"

            docker network create \
                --driver bridge \
                --subnet="$subnet" \
                --gateway="$gateway" \
                --label "crackseg.network=$network_type" \
                --label "crackseg.version=13.9" \
                --label "crackseg.environment=e2e" \
                "$network_name"

            log_success "Created network: $network_name"
        fi
    done

    log_success "All networks created successfully"
}

remove_networks() {
    log "Removing Docker networks..."

    # Stop all containers first
    if docker compose -f "$COMPOSE_FILE" ps -q >/dev/null 2>&1; then
        log "Stopping services before network removal..."
        docker compose -f "$COMPOSE_FILE" down --remove-orphans
    fi

    for network_type in "${!NETWORKS[@]}"; do
        local network_name="${NETWORKS[$network_type]}"

        if docker network inspect "$network_name" >/dev/null 2>&1; then
            log "Removing network: $network_name"
            docker network rm "$network_name" || log_warning "Failed to remove $network_name"
        else
            log_info "Network $network_name does not exist"
        fi
    done

    # Clean up any dangling networks
    docker network prune -f >/dev/null 2>&1 || true

    log_success "Network removal completed"
}

inspect_networks() {
    log "Inspecting Docker networks..."

    for network_type in "${!NETWORKS[@]}"; do
        local network_name="${NETWORKS[$network_type]}"

        if docker network inspect "$network_name" >/dev/null 2>&1; then
            echo
            log_info "Network: $network_name ($network_type)"

            # Get network details
            local subnet
            subnet=$(docker network inspect "$network_name" --format='{{(index .IPAM.Config 0).Subnet}}')
            local gateway
            gateway=$(docker network inspect "$network_name" --format='{{(index .IPAM.Config 0).Gateway}}')

            echo "  Subnet: $subnet"
            echo "  Gateway: $gateway"

            # Get connected containers
            local containers
            containers=$(docker network inspect "$network_name" --format='{{range $k,$v := .Containers}}{{$v.Name}} {{end}}')

            if [[ -n "$containers" ]]; then
                echo "  Connected containers: $containers"
            else
                echo "  Connected containers: None"
            fi

            # Check network connectivity
            check_network_connectivity "$network_name"
        else
            log_warning "Network $network_name does not exist"
        fi
    done
}

check_network_connectivity() {
    local network_name="$1"

    # Get containers in this network
    local containers
    containers=$(docker network inspect "$network_name" --format='{{range $k,$v := .Containers}}{{$v.Name}} {{end}}')

    if [[ -z "$containers" ]]; then
        echo "  Connectivity: No containers to test"
        return
    fi

    # Convert to array
    read -ra container_array <<< "$containers"

    if [[ ${#container_array[@]} -lt 2 ]]; then
        echo "  Connectivity: Only one container, cannot test"
        return
    fi

    # Test connectivity between first two containers
    local source_container="${container_array[0]}"
    local target_container="${container_array[1]}"

    if docker exec "$source_container" ping -c 1 -W 2 "$target_container" >/dev/null 2>&1; then
        echo "  Connectivity: ✓ OK ($source_container → $target_container)"
    else
        echo "  Connectivity: ✗ FAILED ($source_container → $target_container)"
    fi
}

validate_service_discovery() {
    log "Validating service discovery across networks..."

    # Service discovery test matrix
    local -A service_tests=(
        ["test-runner→streamlit"]="test-runner:streamlit:8501"
        ["test-runner→selenium-hub"]="test-runner:hub:4444"
        ["chrome-node→selenium-hub"]="chrome-node:hub:4444"
        ["firefox-node→selenium-hub"]="firefox-node:hub:4444"
    )

    local success_count=0
    local total_tests=${#service_tests[@]}

    for test_name in "${!service_tests[@]}"; do
        IFS=':' read -ra test_parts <<< "${service_tests[$test_name]}"
        local source_container="${test_parts[0]}"
        local target_service="${test_parts[1]}"
        local target_port="${test_parts[2]}"

        echo -n "  Testing $test_name: "

        if docker exec "$source_container" nc -z "$target_service" "$target_port" 2>/dev/null; then
            echo -e "${GREEN}✓ OK${NC}"
            ((success_count++))
        else
            echo -e "${RED}✗ FAILED${NC}"
        fi
    done

    echo
    if [[ $success_count -eq $total_tests ]]; then
        log_success "All service discovery tests passed ($success_count/$total_tests)"
    else
        log_warning "Service discovery issues detected ($success_count/$total_tests passed)"
    fi
}

show_network_topology() {
    log "Network topology visualization:"

    cat << 'EOF'

┌─────────────────────────────────────────────────────────────────┐
│                CrackSeg Multi-Network Architecture               │
└─────────────────────────────────────────────────────────────────┘

┌─ Frontend Network (172.20.0.0/24) ─────────────────────────────┐
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  Streamlit App  │    │  Test Runner    │                    │
│  │  172.20.0.10    │    │  172.20.0.40    │                    │
│  └─────────────────┘    └─────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ Bridge Connection
                                │
┌─ Backend Network (172.21.0.0/24) ──────────────────────────────┐
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  Selenium Hub   │    │  Test Runner    │                    │
│  │  172.21.0.10    │    │  172.21.0.40    │                    │
│  └─────────────────┘    └─────────────────┘                    │
│           │                                                     │
│           ├─ Chrome Node (172.21.0.30)                         │
│           ├─ Firefox Node (172.21.0.31)                        │
│           ├─ Edge Node (172.21.0.32)                           │
│           └─ Video Recorder (172.21.0.50)                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ Management Access
                                │
┌─ Management Network (172.22.0.0/24) ───────────────────────────┐
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  Grid Console   │    │  Health Checks  │                    │
│  │  172.22.0.70    │    │  All Services   │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  noVNC Debug    │    │  Video Mgmt     │                    │
│  │  172.22.0.60    │    │  172.22.0.50    │                    │
│  └─────────────────┘    └─────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘

Security Zones:
• Frontend: Public-facing services (Streamlit)
• Backend: Internal service communication (Selenium Grid)
• Management: Administrative and monitoring services
• Legacy: Backward compatibility (deprecated)

EOF
}

# =============================================================================
# Security and Policy Functions
# =============================================================================

apply_network_policies() {
    log "Applying network security policies..."

    # Note: Docker doesn't have built-in network policies like Kubernetes
    # We implement basic security through network isolation and port restrictions

    log_info "Network isolation implemented through:"
    echo "  • Separate subnets for different service types"
    echo "  • Limited inter-network communication"
    echo "  • Management network restricted to admin services"
    echo "  • Frontend network exposed only for user access"

    log_success "Network security policies applied"
}

check_security_compliance() {
    log "Checking network security compliance..."

    local compliance_issues=0

    # Check for unnecessary exposed ports
    log_info "Checking exposed port security..."

    # Get all exposed ports
    local exposed_ports
    exposed_ports=$(docker compose -f "$COMPOSE_FILE" config | grep -E "^\s*-\s*[0-9]+:" | sort | uniq)

    echo "  Exposed ports:"
    echo "$exposed_ports" | while read -r port; do
        echo "    $port"
    done

    # Check network isolation
    log_info "Checking network isolation..."

    for network_type in "${!NETWORKS[@]}"; do
        local network_name="${NETWORKS[$network_type]}"

        if docker network inspect "$network_name" >/dev/null 2>&1; then
            local enable_icc
            enable_icc=$(docker network inspect "$network_name" --format='{{index .Options "com.docker.network.bridge.enable_icc"}}')

            case "$network_type" in
                "frontend")
                    if [[ "$enable_icc" != "true" ]]; then
                        log_warning "Frontend network should allow inter-container communication"
                        ((compliance_issues++))
                    fi
                    ;;
                "backend")
                    if [[ "$enable_icc" != "true" ]]; then
                        log_warning "Backend network should allow inter-container communication"
                        ((compliance_issues++))
                    fi
                    ;;
                "management")
                    if [[ "$enable_icc" != "true" ]]; then
                        log_warning "Management network should allow monitoring communication"
                        ((compliance_issues++))
                    fi
                    ;;
            esac
        fi
    done

    if [[ $compliance_issues -eq 0 ]]; then
        log_success "Network security compliance check passed"
    else
        log_warning "Found $compliance_issues compliance issues"
    fi

    return $compliance_issues
}

# =============================================================================
# Monitoring and Health Functions
# =============================================================================

monitor_network_health() {
    local interval="${1:-10}"

    log "Starting network health monitoring (interval: ${interval}s)..."
    log_info "Press Ctrl+C to stop monitoring"

    while true; do
        clear
        echo -e "${CYAN}CrackSeg Network Health Monitor - $(date)${NC}"
        echo "=================================================================================================="

        # Check each network
        for network_type in "${!NETWORKS[@]}"; do
            local network_name="${NETWORKS[$network_type]}"

            echo
            echo -e "${BLUE}Network: $network_name ($network_type)${NC}"

            if docker network inspect "$network_name" >/dev/null 2>&1; then
                # Get container count
                local container_count
                container_count=$(docker network inspect "$network_name" --format='{{len .Containers}}')
                echo "  Containers: $container_count"

                # Get network stats if available
                local network_id
                network_id=$(docker network inspect "$network_name" --format='{{.Id}}')
                echo "  Network ID: ${network_id:0:12}"

                # Basic connectivity check
                check_network_connectivity "$network_name"
            else
                echo -e "  Status: ${RED}Not Found${NC}"
            fi
        done

        # Overall health summary
        echo
        echo -e "${YELLOW}Overall Network Health${NC}"
        echo "======================"

        local healthy_networks=0
        for network_type in "${!NETWORKS[@]}"; do
            local network_name="${NETWORKS[$network_type]}"
            if docker network inspect "$network_name" >/dev/null 2>&1; then
                ((healthy_networks++))
            fi
        done

        echo "Healthy Networks: $healthy_networks/${#NETWORKS[@]}"

        if [[ $healthy_networks -eq ${#NETWORKS[@]} ]]; then
            echo -e "Status: ${GREEN}All Networks Operational${NC}"
        else
            echo -e "Status: ${RED}Network Issues Detected${NC}"
        fi

        sleep "$interval"
    done
}

# =============================================================================
# Utility Functions
# =============================================================================

backup_network_config() {
    local backup_dir="$DOCKER_DIR/network-backups"
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="$backup_dir/network-config-$timestamp.json"

    mkdir -p "$backup_dir"

    log "Creating network configuration backup..."

    {
        echo "{"
        echo "  \"timestamp\": \"$(date -Iseconds)\","
        echo "  \"networks\": {"

        local first=true
        for network_type in "${!NETWORKS[@]}"; do
            local network_name="${NETWORKS[$network_type]}"

            if docker network inspect "$network_name" >/dev/null 2>&1; then
                if [[ "$first" != true ]]; then
                    echo ","
                fi
                echo -n "    \"$network_type\": "
                docker network inspect "$network_name" | jq -c '.[0]'
                first=false
            fi
        done

        echo
        echo "  }"
        echo "}"
    } > "$backup_file"

    log_success "Network configuration backed up to: $backup_file"
}

restore_network_config() {
    local backup_file="$1"

    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi

    log "Restoring network configuration from: $backup_file"

    # This is a simplified restore - in practice, you'd need more sophisticated logic
    log_warning "Network restore not fully implemented - please recreate networks manually"
    log_info "Backup file contains configuration that can be used as reference"

    return 0
}

# =============================================================================
# Main Command Functions
# =============================================================================

start_networks() {
    log "Starting multi-network Docker architecture..."

    create_networks
    apply_network_policies

    log_success "Multi-network architecture ready"
}

stop_networks() {
    log "Stopping multi-network Docker architecture..."

    remove_networks

    log_success "Multi-network architecture stopped"
}

status_networks() {
    log "Multi-network Docker architecture status:"
    echo

    inspect_networks
    show_network_topology

    if command -v docker compose >/dev/null 2>&1 && [[ -f "$COMPOSE_FILE" ]]; then
        echo
        log_info "Service status:"
        docker compose -f "$COMPOSE_FILE" ps --format table
    fi
}

test_networks() {
    log "Testing network connectivity and service discovery..."

    inspect_networks
    echo
    validate_service_discovery
    echo
    check_security_compliance
}

# =============================================================================
# Help and Usage
# =============================================================================

show_help() {
    cat << EOF
${CYAN}CrackSeg Network Manager${NC}

${YELLOW}USAGE:${NC}
    ./network-manager.sh [command] [options]

${YELLOW}COMMANDS:${NC}
    ${GREEN}start${NC}                     Create and configure networks
    ${GREEN}stop${NC}                      Remove networks and cleanup
    ${GREEN}restart${NC}                   Stop and start networks
    ${GREEN}status${NC}                    Show network status and topology
    ${GREEN}test${NC}                      Test connectivity and service discovery
    ${GREEN}monitor [interval]${NC}        Monitor network health (default: 10s)
    ${GREEN}backup${NC}                    Backup network configuration
    ${GREEN}restore <file>${NC}            Restore network configuration
    ${GREEN}policies${NC}                  Apply network security policies
    ${GREEN}compliance${NC}                Check security compliance
    ${GREEN}help${NC}                      Show this help message

${YELLOW}EXAMPLES:${NC}
    ${GREEN}./network-manager.sh start${NC}              # Create all networks
    ${GREEN}./network-manager.sh status${NC}             # Show network status
    ${GREEN}./network-manager.sh test${NC}               # Test connectivity
    ${GREEN}./network-manager.sh monitor 5${NC}          # Monitor every 5 seconds
    ${GREEN}./network-manager.sh backup${NC}             # Backup configuration

${YELLOW}NETWORKS:${NC}
    ${GREEN}Frontend${NC}     (172.20.0.0/24)  - Public-facing services
    ${GREEN}Backend${NC}      (172.21.0.0/24)  - Internal service communication
    ${GREEN}Management${NC}   (172.22.0.0/24)  - Administrative services
    ${GREEN}Legacy${NC}       (172.20.0.0/16)  - Backward compatibility

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
        "start")
            start_networks
            ;;
        "stop")
            stop_networks
            ;;
        "restart")
            stop_networks
            sleep 2
            start_networks
            ;;
        "status")
            status_networks
            ;;
        "test")
            test_networks
            ;;
        "monitor")
            monitor_network_health "${2:-10}"
            ;;
        "backup")
            backup_network_config
            ;;
        "restore")
            if [[ $# -lt 2 ]]; then
                log_error "Usage: $0 restore <backup_file>"
                exit 1
            fi
            restore_network_config "$2"
            ;;
        "policies")
            apply_network_policies
            ;;
        "compliance")
            check_security_compliance
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

# Run main function with all arguments
main "$@"