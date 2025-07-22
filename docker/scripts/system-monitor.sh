#!/usr/bin/env bash
# =============================================================================
# CrackSeg System Monitor - Unified Monitoring and Dashboard
# =============================================================================
# Purpose: Real-time system monitoring, resource usage tracking, and
#          centralized dashboard for Docker infrastructure
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
HEALTH_MANAGER="$SCRIPT_DIR/health-check-manager.sh"
NETWORK_MANAGER="$SCRIPT_DIR/network-manager.sh"
ARTIFACT_MANAGER="$SCRIPT_DIR/artifact-manager.sh"

# Monitoring configuration
DEFAULT_REFRESH_INTERVAL=5
DEFAULT_LOG_RETENTION_DAYS=7
MONITORING_OUTPUT_DIR="$PROJECT_ROOT/test-results/monitoring"
DASHBOARD_DIR="$MONITORING_OUTPUT_DIR/dashboard"
METRICS_DIR="$MONITORING_OUTPUT_DIR/metrics"
LOGS_DIR="$MONITORING_OUTPUT_DIR/logs"

# Resource thresholds
CPU_THRESHOLD=80
MEMORY_THRESHOLD=85
DISK_THRESHOLD=90
NETWORK_THRESHOLD_MBPS=100

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly MAGENTA='\033[0;35m'
readonly WHITE='\033[1;37m'
readonly NC='\033[0m' # No Color

# =============================================================================
# Logging Functions
# =============================================================================

log() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] [MONITOR]${NC} $*"
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

print_banner() {
    echo -e "${CYAN}"
    echo "============================================================================="
    echo "  CrackSeg System Monitor - Unified Infrastructure Monitoring"
    echo "  Real-time Metrics, Resource Tracking & Integrated Dashboard"
    echo "============================================================================="
    echo -e "${NC}"
}

# =============================================================================
# Environment Setup
# =============================================================================

setup_monitoring_environment() {
    log "Setting up monitoring environment..."

    # Create monitoring directories
    mkdir -p "$MONITORING_OUTPUT_DIR" "$DASHBOARD_DIR" "$METRICS_DIR" "$LOGS_DIR"

    # Set up monitoring session
    export MONITORING_SESSION_ID="monitor_$(date +%Y%m%d_%H%M%S)"
    export MONITORING_START_TIME=$(date -Iseconds)

    log_info "Monitoring session ID: $MONITORING_SESSION_ID"
    log_info "Output directory: $MONITORING_OUTPUT_DIR"
    log_success "Monitoring environment setup completed"
}

# =============================================================================
# System Resource Monitoring
# =============================================================================

get_system_metrics() {
    local timestamp
    timestamp=$(date -Iseconds)

    # CPU usage
    local cpu_usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//' || echo "0")

    # Memory usage
    local memory_info
    memory_info=$(free -m | grep '^Mem:')
    local total_mem
    total_mem=$(echo "$memory_info" | awk '{print $2}')
    local used_mem
    used_mem=$(echo "$memory_info" | awk '{print $3}')
    local memory_usage
    memory_usage=$((used_mem * 100 / total_mem))

    # Disk usage
    local disk_usage
    disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')

    # Load average
    local load_avg
    load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')

    # Create metrics JSON
    cat << EOF
{
    "timestamp": "$timestamp",
    "cpu_usage": $cpu_usage,
    "memory": {
        "total_mb": $total_mem,
        "used_mb": $used_mem,
        "usage_percent": $memory_usage
    },
    "disk_usage_percent": $disk_usage,
    "load_average": $load_avg,
    "system": {
        "uptime": "$(uptime -p)",
        "processes": $(ps aux | wc -l)
    }
}
EOF
}

get_docker_metrics() {
    local timestamp
    timestamp=$(date -Iseconds)

    # Docker system information
    local docker_info
    docker_info=$(docker system df --format "{{json .}}" 2>/dev/null || echo '{}')

    # Container stats
    local container_stats=()
    if docker ps -q >/dev/null 2>&1; then
        local containers
        containers=$(docker ps --format "{{.Names}}")

        while IFS= read -r container; do
            if [[ -n "$container" ]]; then
                local stats
                stats=$(docker stats "$container" --no-stream --format "{{json .}}" 2>/dev/null || echo '{}')
                container_stats+=("$stats")
            fi
        done <<< "$containers"
    fi

    # Network usage (simplified)
    local network_stats
    network_stats=$(docker network ls --format "{{json .}}" | jq -s . 2>/dev/null || echo '[]')

    # Create Docker metrics JSON
    cat << EOF
{
    "timestamp": "$timestamp",
    "system_info": $docker_info,
    "containers": [$(IFS=','; echo "${container_stats[*]}")],
    "networks": $network_stats
}
EOF
}

get_service_health_metrics() {
    local timestamp
    timestamp=$(date -Iseconds)

    # Get service health from health manager
    local health_status
    if [[ -f "$HEALTH_MANAGER" ]]; then
        health_status=$("$HEALTH_MANAGER" check json 2>/dev/null || echo '{}')
    else
        health_status='{}'
    fi

    # Add timestamp
    echo "$health_status" | jq --arg ts "$timestamp" '. + {timestamp: $ts}'
}

# =============================================================================
# Real-time Dashboard
# =============================================================================

display_dashboard() {
    local interval="${1:-$DEFAULT_REFRESH_INTERVAL}"

    print_banner

    while true; do
        clear
        print_banner

        # Get current metrics
        local system_metrics
        system_metrics=$(get_system_metrics)
        local docker_metrics
        docker_metrics=$(get_docker_metrics)
        local health_metrics
        health_metrics=$(get_service_health_metrics)

        # Display system overview
        display_system_overview "$system_metrics"

        # Display Docker overview
        display_docker_overview "$docker_metrics"

        # Display service health
        display_service_health "$health_metrics"

        # Display alerts if any
        display_alerts "$system_metrics" "$docker_metrics"

        echo -e "\n${YELLOW}Last updated: $(date) | Refresh interval: ${interval}s${NC}"
        echo -e "${BLUE}Press Ctrl+C to stop monitoring${NC}"

        sleep "$interval"
    done
}

display_system_overview() {
    local metrics="$1"

    echo -e "\n${WHITE}📊 SYSTEM OVERVIEW${NC}"
    echo "=================================="

    local cpu_usage
    cpu_usage=$(echo "$metrics" | jq -r '.cpu_usage')
    local memory_usage
    memory_usage=$(echo "$metrics" | jq -r '.memory.usage_percent')
    local disk_usage
    disk_usage=$(echo "$metrics" | jq -r '.disk_usage_percent')
    local load_avg
    load_avg=$(echo "$metrics" | jq -r '.load_average')

    # CPU with color coding
    local cpu_color="$GREEN"
    if (( $(echo "$cpu_usage > $CPU_THRESHOLD" | bc -l) )); then
        cpu_color="$RED"
    elif (( $(echo "$cpu_usage > 60" | bc -l) )); then
        cpu_color="$YELLOW"
    fi

    # Memory with color coding
    local mem_color="$GREEN"
    if (( memory_usage > MEMORY_THRESHOLD )); then
        mem_color="$RED"
    elif (( memory_usage > 70 )); then
        mem_color="$YELLOW"
    fi

    # Disk with color coding
    local disk_color="$GREEN"
    if (( disk_usage > DISK_THRESHOLD )); then
        disk_color="$RED"
    elif (( disk_usage > 75 )); then
        disk_color="$YELLOW"
    fi

    printf "CPU Usage:    ${cpu_color}%6.1f%%%${NC}\n" "$cpu_usage"
    printf "Memory Usage: ${mem_color}%6d%%%${NC}\n" "$memory_usage"
    printf "Disk Usage:   ${disk_color}%6d%%%${NC}\n" "$disk_usage"
    printf "Load Average: %6.2f\n" "$load_avg"
}

display_docker_overview() {
    local metrics="$1"

    echo -e "\n${WHITE}🐳 DOCKER OVERVIEW${NC}"
    echo "=================================="

    # Count containers by status
    local total_containers
    total_containers=$(echo "$metrics" | jq '.containers | length')

    local running_containers=0
    local unhealthy_containers=0

    if [[ "$total_containers" -gt 0 ]]; then
        # Analyze container health (simplified)
        local containers_json
        containers_json=$(echo "$metrics" | jq -r '.containers[]')

        # This is a simplified analysis - in practice would parse individual container stats
        running_containers=$total_containers
    fi

    printf "Total Containers:   %3d\n" "$total_containers"
    printf "Running:           %3d\n" "$running_containers"
    printf "Unhealthy:         %3d\n" "$unhealthy_containers"

    # Docker system space usage
    if docker system df >/dev/null 2>&1; then
        echo -e "\n${BLUE}Storage Usage:${NC}"
        docker system df --format "table {{.Type}}\t{{.Total}}\t{{.Active}}\t{{.Size}}\t{{.Reclaimable}}" | head -5
    fi
}

display_service_health() {
    local metrics="$1"

    echo -e "\n${WHITE}❤️  SERVICE HEALTH${NC}"
    echo "=================================="

    # Check if we have health data
    if echo "$metrics" | jq -e '.services' >/dev/null 2>&1; then
        local services
        services=$(echo "$metrics" | jq -r '.services // {} | keys[]' 2>/dev/null || echo "")

        if [[ -n "$services" ]]; then
            while IFS= read -r service; do
                local status
                status=$(echo "$metrics" | jq -r ".services[\"$service\"].status // \"unknown\"")

                local status_color="$GREEN"
                local status_symbol="✅"

                case "$status" in
                    "healthy"|"running")
                        status_color="$GREEN"
                        status_symbol="✅"
                        ;;
                    "unhealthy"|"failed")
                        status_color="$RED"
                        status_symbol="❌"
                        ;;
                    "starting")
                        status_color="$YELLOW"
                        status_symbol="🔄"
                        ;;
                    *)
                        status_color="$YELLOW"
                        status_symbol="❓"
                        ;;
                esac

                printf "%-20s ${status_color}%s %s${NC}\n" "$service:" "$status_symbol" "$status"
            done <<< "$services"
        else
            echo "No service health data available"
        fi
    else
        echo "Health monitoring not available"
    fi
}

display_alerts() {
    local system_metrics="$1"
    local docker_metrics="$2"

    local alerts=()

    # Check system alerts
    local cpu_usage
    cpu_usage=$(echo "$system_metrics" | jq -r '.cpu_usage')
    if (( $(echo "$cpu_usage > $CPU_THRESHOLD" | bc -l) )); then
        alerts+=("HIGH CPU: ${cpu_usage}% (threshold: ${CPU_THRESHOLD}%)")
    fi

    local memory_usage
    memory_usage=$(echo "$system_metrics" | jq -r '.memory.usage_percent')
    if (( memory_usage > MEMORY_THRESHOLD )); then
        alerts+=("HIGH MEMORY: ${memory_usage}% (threshold: ${MEMORY_THRESHOLD}%)")
    fi

    local disk_usage
    disk_usage=$(echo "$system_metrics" | jq -r '.disk_usage_percent')
    if (( disk_usage > DISK_THRESHOLD )); then
        alerts+=("HIGH DISK: ${disk_usage}% (threshold: ${DISK_THRESHOLD}%)")
    fi

    # Display alerts if any
    if [[ ${#alerts[@]} -gt 0 ]]; then
        echo -e "\n${RED}🚨 ALERTS${NC}"
        echo "=================================="
        for alert in "${alerts[@]}"; do
            echo -e "${RED}⚠️  $alert${NC}"
        done
    fi
}

# =============================================================================
# Log Collection and Management
# =============================================================================

collect_system_logs() {
    local output_dir="${1:-$LOGS_DIR}"
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)

    log "Collecting system logs..."
    mkdir -p "$output_dir"

    # System logs
    journalctl -u docker --since "1 hour ago" > "$output_dir/docker-${timestamp}.log" 2>/dev/null || true

    # Docker logs
    if docker ps -q >/dev/null 2>&1; then
        local containers
        containers=$(docker ps --format "{{.Names}}")

        while IFS= read -r container; do
            if [[ -n "$container" ]]; then
                docker logs "$container" --since 1h > "$output_dir/${container}-${timestamp}.log" 2>&1 || true
            fi
        done <<< "$containers"
    fi

    # System metrics history
    if [[ -f "$METRICS_DIR/system-metrics.jsonl" ]]; then
        cp "$METRICS_DIR/system-metrics.jsonl" "$output_dir/system-metrics-${timestamp}.jsonl"
    fi

    log_success "System logs collected to: $output_dir"
}

start_metrics_collection() {
    local interval="${1:-$DEFAULT_REFRESH_INTERVAL}"
    local background="${2:-true}"

    log "Starting metrics collection (interval: ${interval}s)..."

    if [[ "$background" == "true" ]]; then
        # Start in background
        nohup bash -c "
            while true; do
                echo \"\$(get_system_metrics)\" >> \"$METRICS_DIR/system-metrics.jsonl\"
                echo \"\$(get_docker_metrics)\" >> \"$METRICS_DIR/docker-metrics.jsonl\"
                echo \"\$(get_service_health_metrics)\" >> \"$METRICS_DIR/health-metrics.jsonl\"
                sleep $interval
            done
        " > "$LOGS_DIR/metrics-collection.log" 2>&1 &

        local metrics_pid=$!
        echo "$metrics_pid" > "/tmp/system-monitor-metrics.pid"

        log_success "Metrics collection started in background (PID: $metrics_pid)"
    else
        # Start in foreground
        while true; do
            echo "$(get_system_metrics)" >> "$METRICS_DIR/system-metrics.jsonl"
            echo "$(get_docker_metrics)" >> "$METRICS_DIR/docker-metrics.jsonl"
            echo "$(get_service_health_metrics)" >> "$METRICS_DIR/health-metrics.jsonl"
            sleep "$interval"
        done
    fi
}

stop_metrics_collection() {
    log "Stopping metrics collection..."

    if [[ -f "/tmp/system-monitor-metrics.pid" ]]; then
        local metrics_pid
        metrics_pid=$(cat "/tmp/system-monitor-metrics.pid")

        if kill -0 "$metrics_pid" 2>/dev/null; then
            kill "$metrics_pid"
            rm -f "/tmp/system-monitor-metrics.pid"
            log_success "Metrics collection stopped (PID: $metrics_pid)"
        else
            log_warning "Metrics collection process not found"
            rm -f "/tmp/system-monitor-metrics.pid"
        fi
    else
        log_warning "Metrics collection is not running"
    fi
}

# =============================================================================
# Report Generation
# =============================================================================

generate_monitoring_report() {
    local output_file="${1:-$MONITORING_OUTPUT_DIR/monitoring-report-$(date +%Y%m%d_%H%M%S).html}"

    log "Generating monitoring report..."

    # Create HTML report
    cat > "$output_file" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CrackSeg System Monitoring Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        .header { background: #007bff; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .metric-card { background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }
        .metric-value { font-size: 2em; font-weight: bold; color: #007bff; }
        .metric-label { color: #666; font-size: 0.9em; }
        .alert { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 10px; border-radius: 5px; margin: 10px 0; }
        .success { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 10px; border-radius: 5px; margin: 10px 0; }
        .timestamp { color: #666; font-size: 0.9em; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>CrackSeg System Monitoring Report</h1>
            <p class="timestamp">Generated: $(date)</p>
        </div>

        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Current System Status</div>
                <div class="metric-value">$(if docker ps -q >/dev/null 2>&1; then echo "🟢 Online"; else echo "🔴 Offline"; fi)</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Active Containers</div>
                <div class="metric-value">$(docker ps -q | wc -l 2>/dev/null || echo "0")</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">System Uptime</div>
                <div class="metric-value">$(uptime -p | sed 's/up //')</div>
            </div>
        </div>

        <h2>Resource Usage</h2>
        <p>Current system resource utilization and Docker container metrics.</p>

        <h2>Service Health</h2>
        <p>Health status of all monitored services and containers.</p>

        <h2>Recent Metrics</h2>
        <p>Historical metrics data and trends.</p>

        <div class="success">
            <strong>Report Status:</strong> Monitoring report generated successfully
        </div>
    </div>
</body>
</html>
EOF

    log_success "Monitoring report generated: $output_file"
}

# =============================================================================
# Cleanup and Maintenance
# =============================================================================

cleanup_old_logs() {
    local retention_days="${1:-$DEFAULT_LOG_RETENTION_DAYS}"

    log "Cleaning up logs older than $retention_days days..."

    # Clean up log files
    find "$LOGS_DIR" -name "*.log" -type f -mtime +$retention_days -delete 2>/dev/null || true
    find "$METRICS_DIR" -name "*.jsonl" -type f -mtime +$retention_days -delete 2>/dev/null || true

    log_success "Log cleanup completed"
}

# =============================================================================
# Help and Usage
# =============================================================================

show_help() {
    cat << EOF
${CYAN}CrackSeg System Monitor - Unified Infrastructure Monitoring${NC}

${YELLOW}USAGE:${NC}
    $0 [COMMAND] [OPTIONS]

${YELLOW}COMMANDS:${NC}
    ${GREEN}dashboard [interval]${NC}               Show real-time dashboard
    ${GREEN}metrics [interval] [background]${NC}    Start metrics collection
    ${GREEN}stop-metrics${NC}                      Stop metrics collection
    ${GREEN}collect-logs [output-dir]${NC}         Collect system logs
    ${GREEN}report [output-file]${NC}              Generate monitoring report
    ${GREEN}cleanup [retention-days]${NC}          Clean up old logs
    ${GREEN}status${NC}                           Show current monitoring status
    ${GREEN}help${NC}                             Show this help

${YELLOW}OPTIONS:${NC}
    ${GREEN}interval${NC}         Refresh/collection interval in seconds (default: $DEFAULT_REFRESH_INTERVAL)
    ${GREEN}background${NC}       Run metrics collection in background (true/false)
    ${GREEN}retention-days${NC}   Log retention period (default: $DEFAULT_LOG_RETENTION_DAYS)

${YELLOW}EXAMPLES:${NC}
    ${GREEN}$0 dashboard${NC}                      # Show real-time dashboard
    ${GREEN}$0 dashboard 10${NC}                   # Dashboard with 10s refresh
    ${GREEN}$0 metrics 5 true${NC}                 # Start background metrics collection
    ${GREEN}$0 collect-logs${NC}                   # Collect all system logs
    ${GREEN}$0 report /tmp/report.html${NC}        # Generate custom report
    ${GREEN}$0 cleanup 3${NC}                      # Clean logs older than 3 days

${YELLOW}MONITORING FEATURES:${NC}
    • Real-time system resource monitoring
    • Docker container metrics and health
    • Service availability tracking
    • Alert system for threshold violations
    • Historical metrics collection
    • Automated log aggregation
    • HTML report generation

${YELLOW}THRESHOLDS:${NC}
    ${GREEN}CPU:${NC}     ${CPU_THRESHOLD}%
    ${GREEN}Memory:${NC}  ${MEMORY_THRESHOLD}%
    ${GREEN}Disk:${NC}    ${DISK_THRESHOLD}%

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

    # Setup monitoring environment
    setup_monitoring_environment

    case "$1" in
        "dashboard")
            local interval="${2:-$DEFAULT_REFRESH_INTERVAL}"
            display_dashboard "$interval"
            ;;
        "metrics")
            local interval="${2:-$DEFAULT_REFRESH_INTERVAL}"
            local background="${3:-true}"
            start_metrics_collection "$interval" "$background"
            ;;
        "stop-metrics")
            stop_metrics_collection
            ;;
        "collect-logs")
            local output_dir="${2:-$LOGS_DIR}"
            collect_system_logs "$output_dir"
            ;;
        "report")
            local output_file="${2:-}"
            generate_monitoring_report "$output_file"
            ;;
        "cleanup")
            local retention_days="${2:-$DEFAULT_LOG_RETENTION_DAYS}"
            cleanup_old_logs "$retention_days"
            ;;
        "status")
            print_banner
            echo -e "\n${WHITE}📊 MONITORING STATUS${NC}"
            echo "=================================="

            if [[ -f "/tmp/system-monitor-metrics.pid" ]] && kill -0 "$(cat "/tmp/system-monitor-metrics.pid")" 2>/dev/null; then
                echo -e "${GREEN}✅ Metrics collection is running${NC} (PID: $(cat "/tmp/system-monitor-metrics.pid"))"
            else
                echo -e "${YELLOW}❌ Metrics collection is not running${NC}"
            fi

            echo "Output directory: $MONITORING_OUTPUT_DIR"
            echo "Session ID: $MONITORING_SESSION_ID"
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