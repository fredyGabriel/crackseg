#!/bin/bash
# =============================================================================
# CrackSeg Artifact Management Script
# =============================================================================
# Purpose: Comprehensive artifact management for Docker testing infrastructure
# Features: Collection, cleanup, archiving, verification, and reporting
# Usage: ./artifact-manager.sh [command] [options]
# Subtask: 13.5 - Implement Artifact Management and Volume Configuration
# =============================================================================

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DOCKER_DIR="$PROJECT_ROOT/tests/docker"

# Default paths (aligned with environment variable management from 13.6)
DEFAULT_TEST_RESULTS="${PROJECT_ROOT}/test-results"
DEFAULT_TEST_DATA="${PROJECT_ROOT}/test-data"
DEFAULT_TEST_ARTIFACTS="${PROJECT_ROOT}/test-artifacts"
DEFAULT_SELENIUM_VIDEOS="${PROJECT_ROOT}/selenium-videos"
DEFAULT_ARCHIVE_PATH="${PROJECT_ROOT}/archived-artifacts"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

print_banner() {
    echo -e "${CYAN}"
    echo "============================================================================="
    echo "  CrackSeg - Artifact Management System (Subtask 13.5)"
    echo "============================================================================="
    echo -e "${NC}"
}

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${PURPLE}[DEBUG]${NC} $1"
    fi
}

show_help() {
    cat << EOF
CrackSeg Artifact Management Script

USAGE:
    $0 COMMAND [OPTIONS]

COMMANDS:
    collect         Collect artifacts from containers
    cleanup         Clean up artifacts and volumes
    archive         Archive artifacts for long-term storage
    verify          Verify artifact integrity
    status          Show artifact storage status
    list            List available artifacts
    restore         Restore artifacts from archive
    report          Generate artifact report
    help            Show this help message

COLLECTION OPTIONS:
    --container NAME        Collect from specific container [default: all]
    --type TYPE            Artifact type (logs, screenshots, reports, videos) [default: all]
    --timestamp TIMESTAMP  Collect artifacts from specific test run
    --output-dir DIR       Custom output directory
    --compress             Compress artifacts during collection

CLEANUP OPTIONS:
    --older-than DAYS      Remove artifacts older than N days [default: 7]
    --keep-latest N        Keep latest N artifact sets [default: 5]
    --dry-run             Show what would be deleted without deleting
    --force               Force cleanup without confirmation
    --volumes             Also clean Docker volumes

ARCHIVE OPTIONS:
    --format FORMAT        Archive format (tar.gz, zip) [default: tar.gz]
    --encryption           Enable encryption for sensitive artifacts
    --storage-path PATH    Custom archive storage path
    --retention-days DAYS  Archive retention period [default: 30]

VERIFICATION OPTIONS:
    --checksum            Generate and verify checksums
    --structure           Verify directory structure
    --completeness        Check for missing required artifacts

GLOBAL OPTIONS:
    --env-file FILE       Load environment variables from file
    --debug               Enable debug output
    --quiet               Suppress non-essential output
    --log-file FILE       Log operations to file

EXAMPLES:
    # Collect all artifacts
    $0 collect

    # Collect only screenshots from test-runner
    $0 collect --container test-runner --type screenshots

    # Clean up artifacts older than 3 days
    $0 cleanup --older-than 3

    # Archive current artifacts with compression
    $0 archive --format tar.gz --compression

    # Verify artifact integrity
    $0 verify --checksum --structure

    # Generate comprehensive status report
    $0 report --output-format html

ENVIRONMENT VARIABLES:
    TEST_RESULTS_PATH     Path to test results directory
    TEST_ARTIFACTS_PATH   Path to test artifacts directory
    SELENIUM_VIDEOS_PATH  Path to video recordings
    ARCHIVE_PATH          Path to archived artifacts
    CLEANUP_RETENTION     Default cleanup retention in days

EOF
}

# =============================================================================
# Environment Setup
# =============================================================================

setup_environment() {
    local env_file="${ENV_FILE:-}"

    # Load environment file if specified
    if [[ -n "$env_file" && -f "$env_file" ]]; then
        print_status "Loading environment from: $env_file"
        source "$env_file"
    fi

    # Set artifact paths (with environment variable integration)
    export TEST_RESULTS_PATH="${TEST_RESULTS_PATH:-$DEFAULT_TEST_RESULTS}"
    export TEST_DATA_PATH="${TEST_DATA_PATH:-$DEFAULT_TEST_DATA}"
    export TEST_ARTIFACTS_PATH="${TEST_ARTIFACTS_PATH:-$DEFAULT_TEST_ARTIFACTS}"
    export SELENIUM_VIDEOS_PATH="${SELENIUM_VIDEOS_PATH:-$DEFAULT_SELENIUM_VIDEOS}"
    export ARCHIVE_PATH="${ARCHIVE_PATH:-$DEFAULT_ARCHIVE_PATH}"

    # Create directories if they don't exist
    mkdir -p "$TEST_RESULTS_PATH" "$TEST_ARTIFACTS_PATH" "$SELENIUM_VIDEOS_PATH" "$ARCHIVE_PATH"

    print_debug "Environment configured:"
    print_debug "  Test Results: $TEST_RESULTS_PATH"
    print_debug "  Artifacts: $TEST_ARTIFACTS_PATH"
    print_debug "  Videos: $SELENIUM_VIDEOS_PATH"
    print_debug "  Archive: $ARCHIVE_PATH"
}

# =============================================================================
# Artifact Collection Functions
# =============================================================================

collect_artifacts() {
    local container="${CONTAINER:-all}"
    local artifact_type="${TYPE:-all}"
    local timestamp="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
    local output_dir="${OUTPUT_DIR:-$TEST_ARTIFACTS_PATH}"
    local compress="${COMPRESS:-false}"

    print_status "Collecting artifacts..."
    print_status "  Container: $container"
    print_status "  Type: $artifact_type"
    print_status "  Timestamp: $timestamp"

    local collection_dir="$output_dir/collection_$timestamp"
    mkdir -p "$collection_dir"

    # Collect from running containers
    if [[ "$container" == "all" ]]; then
        collect_from_all_containers "$collection_dir" "$artifact_type"
    else
        collect_from_container "$container" "$collection_dir" "$artifact_type"
    fi

    # Collect from volumes
    collect_from_volumes "$collection_dir" "$artifact_type"

    # Compress if requested
    if [[ "$compress" == "true" ]]; then
        compress_artifacts "$collection_dir"
    fi

    print_success "Artifact collection completed: $collection_dir"
}

collect_from_all_containers() {
    local output_dir="$1"
    local artifact_type="$2"

    local containers=(
        "crackseg-test-runner"
        "crackseg-streamlit-app"
        "crackseg-selenium-hub"
        "crackseg-chrome-node"
        "crackseg-firefox-node"
    )

    for container in "${containers[@]}"; do
        if docker ps -q -f name="$container" | grep -q .; then
            print_status "Collecting from container: $container"
            collect_from_container "$container" "$output_dir" "$artifact_type"
        else
            print_warning "Container not running: $container"
        fi
    done
}

collect_from_container() {
    local container="$1"
    local output_dir="$2"
    local artifact_type="$3"

    local container_dir="$output_dir/$container"
    mkdir -p "$container_dir"

    case "$artifact_type" in
        "logs"|"all")
            collect_container_logs "$container" "$container_dir"
            ;;
        "screenshots"|"all")
            collect_container_screenshots "$container" "$container_dir"
            ;;
        "reports"|"all")
            collect_container_reports "$container" "$container_dir"
            ;;
    esac
}

collect_container_logs() {
    local container="$1"
    local output_dir="$2"

    print_debug "Collecting logs from $container"

    # Container logs
    docker logs "$container" > "$output_dir/container.log" 2>&1 || true

    # Application logs from container filesystem
    if docker exec "$container" test -d "/app/logs" 2>/dev/null; then
        docker cp "$container:/app/logs" "$output_dir/" 2>/dev/null || true
    fi

    # Test execution logs
    if docker exec "$container" test -d "/app/test-results/logs" 2>/dev/null; then
        docker cp "$container:/app/test-results/logs" "$output_dir/" 2>/dev/null || true
    fi
}

collect_container_screenshots() {
    local container="$1"
    local output_dir="$2"

    print_debug "Collecting screenshots from $container"

    # Screenshots from test results
    if docker exec "$container" test -d "/app/test-results/screenshots" 2>/dev/null; then
        docker cp "$container:/app/test-results/screenshots" "$output_dir/" 2>/dev/null || true
    fi

    # Screenshots from test artifacts
    if docker exec "$container" test -d "/app/test-artifacts/screenshots" 2>/dev/null; then
        docker cp "$container:/app/test-artifacts/screenshots" "$output_dir/" 2>/dev/null || true
    fi
}

collect_container_reports() {
    local container="$1"
    local output_dir="$2"

    print_debug "Collecting reports from $container"

    # Test reports
    if docker exec "$container" test -d "/app/test-results/reports" 2>/dev/null; then
        docker cp "$container:/app/test-results/reports" "$output_dir/" 2>/dev/null || true
    fi

    # Coverage reports
    if docker exec "$container" test -d "/app/test-results/coverage" 2>/dev/null; then
        docker cp "$container:/app/test-results/coverage" "$output_dir/" 2>/dev/null || true
    fi
}

collect_from_volumes() {
    local output_dir="$1"
    local artifact_type="$2"

    print_debug "Collecting from Docker volumes"

    local volumes_dir="$output_dir/volumes"
    mkdir -p "$volumes_dir"

    # Named volumes collection
    local volumes=(
        "crackseg-test-results:test-results"
        "crackseg-test-artifacts:test-artifacts"
        "crackseg-selenium-videos:selenium-videos"
    )

    for volume_mapping in "${volumes[@]}"; do
        local volume_name="${volume_mapping%%:*}"
        local volume_alias="${volume_mapping##*:}"

        if docker volume ls -q | grep -q "$volume_name"; then
            print_debug "Collecting from volume: $volume_name"
            collect_volume_data "$volume_name" "$volumes_dir/$volume_alias" "$artifact_type"
        fi
    done
}

collect_volume_data() {
    local volume_name="$1"
    local output_dir="$2"
    local artifact_type="$3"

    mkdir -p "$output_dir"

    # Use temporary container to access volume data
    docker run --rm \
        -v "$volume_name:/source:ro" \
        -v "$output_dir:/dest" \
        alpine:latest \
        sh -c "cp -r /source/* /dest/ 2>/dev/null || true"
}

compress_artifacts() {
    local collection_dir="$1"
    local archive_name="$(basename "$collection_dir").tar.gz"
    local archive_path="$(dirname "$collection_dir")/$archive_name"

    print_status "Compressing artifacts to: $archive_name"

    tar -czf "$archive_path" -C "$(dirname "$collection_dir")" "$(basename "$collection_dir")"

    if [[ -f "$archive_path" ]]; then
        rm -rf "$collection_dir"
        print_success "Artifacts compressed and original directory removed"
    fi
}

# =============================================================================
# Cleanup Functions
# =============================================================================

cleanup_artifacts() {
    local older_than="${OLDER_THAN:-7}"
    local keep_latest="${KEEP_LATEST:-5}"
    local dry_run="${DRY_RUN:-false}"
    local force="${FORCE:-false}"
    local clean_volumes="${VOLUMES:-false}"

    print_status "Cleaning up artifacts..."
    print_status "  Older than: $older_than days"
    print_status "  Keep latest: $keep_latest"
    print_status "  Dry run: $dry_run"

    # Confirm if not dry run and not forced
    if [[ "$dry_run" != "true" && "$force" != "true" ]]; then
        echo -n "Are you sure you want to proceed with cleanup? [y/N]: "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            print_status "Cleanup cancelled"
            return 0
        fi
    fi

    # Clean up artifact directories
    cleanup_directory "$TEST_RESULTS_PATH" "$older_than" "$keep_latest" "$dry_run"
    cleanup_directory "$TEST_ARTIFACTS_PATH" "$older_than" "$keep_latest" "$dry_run"
    cleanup_directory "$SELENIUM_VIDEOS_PATH" "$older_than" "$keep_latest" "$dry_run"

    # Clean up Docker volumes if requested
    if [[ "$clean_volumes" == "true" ]]; then
        cleanup_docker_volumes "$dry_run"
    fi

    print_success "Cleanup completed"
}

cleanup_directory() {
    local target_dir="$1"
    local older_than="$2"
    local keep_latest="$3"
    local dry_run="$4"

    if [[ ! -d "$target_dir" ]]; then
        print_warning "Directory does not exist: $target_dir"
        return 0
    fi

    print_debug "Cleaning up directory: $target_dir"

    # Find files older than specified days
    local old_files
    if old_files=$(find "$target_dir" -type f -mtime +$older_than 2>/dev/null); then
        if [[ -n "$old_files" ]]; then
            if [[ "$dry_run" == "true" ]]; then
                print_status "Would delete old files in $target_dir:"
                echo "$old_files"
            else
                echo "$old_files" | xargs rm -f
                print_debug "Deleted old files in $target_dir"
            fi
        fi
    fi

    # Keep only latest N directories (collection_* pattern)
    local collection_dirs
    if collection_dirs=$(find "$target_dir" -name "collection_*" -type d 2>/dev/null | sort -r | tail -n +$((keep_latest + 1))); then
        if [[ -n "$collection_dirs" ]]; then
            if [[ "$dry_run" == "true" ]]; then
                print_status "Would delete old collections in $target_dir:"
                echo "$collection_dirs"
            else
                echo "$collection_dirs" | xargs rm -rf
                print_debug "Deleted old collections in $target_dir"
            fi
        fi
    fi
}

cleanup_docker_volumes() {
    local dry_run="$1"

    print_debug "Cleaning up Docker volumes"

    local volumes=(
        "crackseg-test-results"
        "crackseg-test-artifacts"
        "crackseg-selenium-videos"
    )

    for volume in "${volumes[@]}"; do
        if docker volume ls -q | grep -q "$volume"; then
            if [[ "$dry_run" == "true" ]]; then
                print_status "Would clean volume: $volume"
            else
                # Don't remove volumes, just clean their contents
                docker run --rm \
                    -v "$volume:/data" \
                    alpine:latest \
                    sh -c "find /data -type f -mtime +7 -delete 2>/dev/null || true"
                print_debug "Cleaned volume: $volume"
            fi
        fi
    done
}

# =============================================================================
# Archive Functions
# =============================================================================

archive_artifacts() {
    local format="${FORMAT:-tar.gz}"
    local encryption="${ENCRYPTION:-false}"
    local storage_path="${STORAGE_PATH:-$ARCHIVE_PATH}"
    local retention_days="${RETENTION_DAYS:-30}"

    print_status "Archiving artifacts..."
    print_status "  Format: $format"
    print_status "  Storage: $storage_path"
    print_status "  Retention: $retention_days days"

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local archive_name="crackseg_artifacts_$timestamp"

    mkdir -p "$storage_path"

    case "$format" in
        "tar.gz")
            create_tar_archive "$archive_name" "$storage_path"
            ;;
        "zip")
            create_zip_archive "$archive_name" "$storage_path"
            ;;
        *)
            print_error "Unsupported archive format: $format"
            return 1
            ;;
    esac

    if [[ "$encryption" == "true" ]]; then
        encrypt_archive "$storage_path/$archive_name.$format"
    fi

    # Apply retention policy
    apply_archive_retention "$storage_path" "$retention_days"

    print_success "Archive created: $storage_path/$archive_name.$format"
}

create_tar_archive() {
    local archive_name="$1"
    local storage_path="$2"
    local archive_file="$storage_path/$archive_name.tar.gz"

    print_debug "Creating tar.gz archive: $archive_file"

    tar -czf "$archive_file" \
        -C "$TEST_RESULTS_PATH" . \
        -C "$TEST_ARTIFACTS_PATH" . \
        -C "$SELENIUM_VIDEOS_PATH" . 2>/dev/null || true

    # Generate checksum
    if command -v sha256sum >/dev/null; then
        sha256sum "$archive_file" > "$archive_file.sha256"
    fi
}

create_zip_archive() {
    local archive_name="$1"
    local storage_path="$2"
    local archive_file="$storage_path/$archive_name.zip"

    print_debug "Creating zip archive: $archive_file"

    if command -v zip >/dev/null; then
        cd "$PROJECT_ROOT"
        zip -r "$archive_file" \
            "$(basename "$TEST_RESULTS_PATH")" \
            "$(basename "$TEST_ARTIFACTS_PATH")" \
            "$(basename "$SELENIUM_VIDEOS_PATH")" \
            2>/dev/null || true
    else
        print_error "zip command not available"
        return 1
    fi
}

apply_archive_retention() {
    local storage_path="$1"
    local retention_days="$2"

    print_debug "Applying archive retention policy: $retention_days days"

    find "$storage_path" -name "crackseg_artifacts_*" -type f -mtime +$retention_days -delete 2>/dev/null || true
}

# =============================================================================
# Verification Functions
# =============================================================================

verify_artifacts() {
    local checksum="${CHECKSUM:-false}"
    local structure="${STRUCTURE:-false}"
    local completeness="${COMPLETENESS:-false}"

    print_status "Verifying artifacts..."

    local verification_passed=true

    if [[ "$structure" == "true" ]]; then
        if ! verify_directory_structure; then
            verification_passed=false
        fi
    fi

    if [[ "$checksum" == "true" ]]; then
        if ! verify_checksums; then
            verification_passed=false
        fi
    fi

    if [[ "$completeness" == "true" ]]; then
        if ! verify_completeness; then
            verification_passed=false
        fi
    fi

    if [[ "$verification_passed" == "true" ]]; then
        print_success "All artifact verification checks passed"
        return 0
    else
        print_error "Some artifact verification checks failed"
        return 1
    fi
}

verify_directory_structure() {
    print_debug "Verifying directory structure"

    local required_dirs=(
        "$TEST_RESULTS_PATH"
        "$TEST_ARTIFACTS_PATH"
        "$SELENIUM_VIDEOS_PATH"
    )

    local structure_valid=true

    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            print_error "Required directory missing: $dir"
            structure_valid=false
        else
            print_debug "Directory exists: $dir"
        fi
    done

    return $([ "$structure_valid" == "true" ])
}

verify_checksums() {
    print_debug "Verifying checksums"

    local checksum_valid=true

    # Verify archive checksums
    if [[ -d "$ARCHIVE_PATH" ]]; then
        for checksum_file in "$ARCHIVE_PATH"/*.sha256; do
            if [[ -f "$checksum_file" ]]; then
                if ! sha256sum -c "$checksum_file" >/dev/null 2>&1; then
                    print_error "Checksum verification failed: $checksum_file"
                    checksum_valid=false
                else
                    print_debug "Checksum verified: $checksum_file"
                fi
            fi
        done
    fi

    return $([ "$checksum_valid" == "true" ])
}

verify_completeness() {
    print_debug "Verifying artifact completeness"

    local completeness_valid=true

    # Check if recent test runs have all expected artifacts
    local latest_collections=$(find "$TEST_ARTIFACTS_PATH" -name "collection_*" -type d -mtime -1 2>/dev/null)

    for collection in $latest_collections; do
        if [[ ! -d "$collection/volumes" ]]; then
            print_warning "Missing volume artifacts in: $collection"
            completeness_valid=false
        fi

        if [[ ! -f "$collection/*/container.log" ]]; then
            print_warning "Missing container logs in: $collection"
            completeness_valid=false
        fi
    done

    return $([ "$completeness_valid" == "true" ])
}

# =============================================================================
# Status and Reporting Functions
# =============================================================================

show_status() {
    print_status "Artifact Storage Status"
    echo ""

    # Directory sizes
    if [[ -d "$TEST_RESULTS_PATH" ]]; then
        local results_size=$(du -sh "$TEST_RESULTS_PATH" 2>/dev/null | cut -f1)
        echo "Test Results: $results_size ($TEST_RESULTS_PATH)"
    fi

    if [[ -d "$TEST_ARTIFACTS_PATH" ]]; then
        local artifacts_size=$(du -sh "$TEST_ARTIFACTS_PATH" 2>/dev/null | cut -f1)
        echo "Test Artifacts: $artifacts_size ($TEST_ARTIFACTS_PATH)"
    fi

    if [[ -d "$SELENIUM_VIDEOS_PATH" ]]; then
        local videos_size=$(du -sh "$SELENIUM_VIDEOS_PATH" 2>/dev/null | cut -f1)
        echo "Selenium Videos: $videos_size ($SELENIUM_VIDEOS_PATH)"
    fi

    if [[ -d "$ARCHIVE_PATH" ]]; then
        local archive_size=$(du -sh "$ARCHIVE_PATH" 2>/dev/null | cut -f1)
        echo "Archives: $archive_size ($ARCHIVE_PATH)"
    fi

    echo ""

    # Docker volumes
    print_status "Docker Volume Status"
    docker volume ls --filter "name=crackseg-" --format "table {{.Name}}\t{{.Driver}}\t{{.Size}}" 2>/dev/null || true

    echo ""

    # Recent collections
    print_status "Recent Collections"
    find "$TEST_ARTIFACTS_PATH" -name "collection_*" -type d -mtime -7 2>/dev/null | sort -r | head -5
}

# =============================================================================
# Main Function
# =============================================================================

main() {
    local command="${1:-help}"
    shift || true

    # Parse command line options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --container)
                CONTAINER="$2"
                shift 2
                ;;
            --type)
                TYPE="$2"
                shift 2
                ;;
            --timestamp)
                TIMESTAMP="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --compress)
                COMPRESS="true"
                shift
                ;;
            --older-than)
                OLDER_THAN="$2"
                shift 2
                ;;
            --keep-latest)
                KEEP_LATEST="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --force)
                FORCE="true"
                shift
                ;;
            --volumes)
                VOLUMES="true"
                shift
                ;;
            --format)
                FORMAT="$2"
                shift 2
                ;;
            --encryption)
                ENCRYPTION="true"
                shift
                ;;
            --storage-path)
                STORAGE_PATH="$2"
                shift 2
                ;;
            --retention-days)
                RETENTION_DAYS="$2"
                shift 2
                ;;
            --checksum)
                CHECKSUM="true"
                shift
                ;;
            --structure)
                STRUCTURE="true"
                shift
                ;;
            --completeness)
                COMPLETENESS="true"
                shift
                ;;
            --env-file)
                ENV_FILE="$2"
                shift 2
                ;;
            --debug)
                DEBUG="true"
                shift
                ;;
            --quiet)
                QUIET="true"
                shift
                ;;
            --log-file)
                LOG_FILE="$2"
                shift 2
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Setup environment
    setup_environment

    # Execute command
    case "$command" in
        collect)
            collect_artifacts
            ;;
        cleanup)
            cleanup_artifacts
            ;;
        archive)
            archive_artifacts
            ;;
        verify)
            verify_artifacts
            ;;
        status)
            show_status
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# =============================================================================
# Script Execution
# =============================================================================

# Only run main if script is executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    print_banner
    main "$@"
fi
