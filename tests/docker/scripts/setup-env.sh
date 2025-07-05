#!/bin/bash
# =============================================================================
# CrackSeg Environment Setup Script
# =============================================================================
# Purpose: Configure environment variables for different deployment environments
# Usage: ./setup-env.sh [local|staging|production|test] [options]
# Designed for Subtask 13.6 - Configure Environment Variable Management
# =============================================================================

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$DOCKER_DIR")")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="local"
VALIDATE_ONLY=false
EXPORT_FILE=""
APPLY_CONFIG=false
VERBOSE=false
FORCE=false

# Function to print colored output
print_info() {
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

# Function to show usage
show_usage() {
    cat << EOF
CrackSeg Environment Setup Script

USAGE:
    $0 [ENVIRONMENT] [OPTIONS]

ENVIRONMENTS:
    local       Local development environment (default)
    staging     Staging/CI environment
    production  Production environment
    test        Automated testing environment

OPTIONS:
    --validate        Validate configuration only
    --export FILE     Export configuration to file
    --apply          Apply configuration to current shell
    --verbose        Show detailed output
    --force          Force overwrite existing files
    --help           Show this help message

EXAMPLES:
    $0 local --apply                    # Setup local development
    $0 staging --validate               # Validate staging config
    $0 test --export test-config.json   # Export test configuration
    $0 production --apply --verbose     # Setup production with details

ENVIRONMENT FILES:
    The script looks for environment files in the following order:
    1. .env.{environment}              # Actual environment file
    2. env.{environment}.template      # Template file

CREATED FILES:
    .env.{environment}   Environment-specific variables
    env-config.json      Exported configuration (if --export used)

For more information, see: docs/guides/docker-testing-infrastructure.md
EOF
}

# Function to validate prerequisites
validate_prerequisites() {
    print_info "Validating prerequisites..."

    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi

    # Check if env_manager.py exists
    if [[ ! -f "$DOCKER_DIR/env_manager.py" ]]; then
        print_error "env_manager.py not found at $DOCKER_DIR/env_manager.py"
        exit 1
    fi

    # Check Python dependencies
    if ! python3 -c "import dataclasses, enum, pathlib, json" 2>/dev/null; then
        print_error "Required Python modules not available"
        exit 1
    fi

    print_success "Prerequisites validated"
}

# Function to detect current environment
detect_environment() {
    local detected_env="local"

    if [[ -n "${CRACKSEG_ENV:-}" ]]; then
        detected_env="$CRACKSEG_ENV"
    elif [[ -n "${NODE_ENV:-}" ]]; then
        detected_env="$NODE_ENV"
    elif [[ -n "${CI:-}" ]]; then
        detected_env="staging"
    fi

    print_info "Detected environment: $detected_env"
    echo "$detected_env"
}

# Function to check if environment file exists
check_env_file() {
    local env_name="$1"
    local env_file="$DOCKER_DIR/.env.$env_name"
    local template_file="$DOCKER_DIR/env.$env_name.template"

    if [[ -f "$env_file" ]]; then
        echo "$env_file"
        return 0
    elif [[ -f "$template_file" ]]; then
        echo "$template_file"
        return 0
    else
        return 1
    fi
}

# Function to create environment file from template
create_env_file() {
    local env_name="$1"
    local template_file="$DOCKER_DIR/env.$env_name.template"
    local env_file="$DOCKER_DIR/.env.$env_name"

    if [[ ! -f "$template_file" ]]; then
        print_error "Template file not found: $template_file"
        return 1
    fi

    if [[ -f "$env_file" ]] && [[ "$FORCE" != "true" ]]; then
        print_warning "Environment file already exists: $env_file"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Keeping existing file"
            return 0
        fi
    fi

    print_info "Creating environment file from template..."
    cp "$template_file" "$env_file"

    # Make sure .env files are in .gitignore
    local gitignore_file="$DOCKER_DIR/.gitignore"
    if [[ ! -f "$gitignore_file" ]] || ! grep -q "\.env\." "$gitignore_file" 2>/dev/null; then
        echo ".env.*" >> "$gitignore_file"
        print_info "Added .env.* to .gitignore"
    fi

    print_success "Created: $env_file"
    print_warning "Please review and update the configuration with your specific values"
}

# Function to validate configuration
validate_configuration() {
    local env_name="$1"

    print_info "Validating configuration for environment: $env_name"

    if ! python3 "$DOCKER_DIR/env_manager.py" --env "$env_name" --validate; then
        print_error "Configuration validation failed"
        return 1
    fi

    print_success "Configuration is valid"
}

# Function to export configuration
export_configuration() {
    local env_name="$1"
    local export_file="$2"

    print_info "Exporting configuration to: $export_file"

    if ! python3 "$DOCKER_DIR/env_manager.py" --env "$env_name" --export "$export_file"; then
        print_error "Configuration export failed"
        return 1
    fi

    print_success "Configuration exported successfully"
}

# Function to apply configuration
apply_configuration() {
    local env_name="$1"

    print_info "Applying configuration for environment: $env_name"

    # Create a temporary script that sets environment variables
    local temp_script=$(mktemp)
    cat > "$temp_script" << 'EOF'
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from env_manager import EnvironmentManager, Environment

manager = EnvironmentManager()
env = Environment(sys.argv[1])
config = manager.create_config_from_env(env)
docker_env = manager.export_to_docker_compose(config)

for key, value in docker_env.items():
    print(f"export {key}='{value}'")
EOF

    # Get the environment variables and evaluate them
    local env_commands
    if env_commands=$(python3 "$temp_script" "$env_name"); then
        # Save to a file that can be sourced
        local env_script="$DOCKER_DIR/.env-$env_name.sh"
        echo "#!/bin/bash" > "$env_script"
        echo "# Generated environment variables for $env_name" >> "$env_script"
        echo "# Source this file: source $env_script" >> "$env_script"
        echo "" >> "$env_script"
        echo "$env_commands" >> "$env_script"
        chmod +x "$env_script"

        print_success "Environment variables prepared"
        print_info "To apply in current shell, run: source $env_script"

        if [[ "$VERBOSE" == "true" ]]; then
            print_info "Generated environment variables:"
            cat "$env_script"
        fi
    else
        print_error "Failed to generate environment variables"
        rm -f "$temp_script"
        return 1
    fi

    rm -f "$temp_script"
}

# Function to show environment status
show_status() {
    local env_name="$1"

    print_info "Environment Status for: $env_name"
    echo "----------------------------------------"

    # Check for environment file
    if env_file=$(check_env_file "$env_name"); then
        echo "✅ Environment file: $env_file"
    else
        echo "❌ Environment file: Not found"
    fi

    # Check configuration validity
    if python3 "$DOCKER_DIR/env_manager.py" --env "$env_name" --validate >/dev/null 2>&1; then
        echo "✅ Configuration: Valid"
    else
        echo "❌ Configuration: Invalid"
    fi

    # Show current environment variables
    local env_script="$DOCKER_DIR/.env-$env_name.sh"
    if [[ -f "$env_script" ]]; then
        echo "✅ Applied script: $env_script"
    else
        echo "❌ Applied script: Not found"
    fi

    echo "----------------------------------------"
}

# Main function
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            local|staging|production|test)
                ENVIRONMENT="$1"
                shift
                ;;
            --validate)
                VALIDATE_ONLY=true
                shift
                ;;
            --export)
                EXPORT_FILE="$2"
                shift 2
                ;;
            --apply)
                APPLY_CONFIG=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # Show header
    print_info "CrackSeg Environment Setup"
    print_info "Environment: $ENVIRONMENT"
    print_info "Docker directory: $DOCKER_DIR"

    # Validate prerequisites
    validate_prerequisites

    # Check if environment file exists, create if needed
    if ! env_file=$(check_env_file "$ENVIRONMENT"); then
        print_warning "Environment file not found for: $ENVIRONMENT"
        if [[ -f "$DOCKER_DIR/env.$ENVIRONMENT.template" ]]; then
            create_env_file "$ENVIRONMENT"
        else
            print_error "No template file found for environment: $ENVIRONMENT"
            exit 1
        fi
    else
        if [[ "$VERBOSE" == "true" ]]; then
            print_info "Using environment file: $env_file"
        fi
    fi

    # Execute requested actions
    if [[ "$VALIDATE_ONLY" == "true" ]]; then
        validate_configuration "$ENVIRONMENT"
    elif [[ -n "$EXPORT_FILE" ]]; then
        validate_configuration "$ENVIRONMENT" && export_configuration "$ENVIRONMENT" "$EXPORT_FILE"
    elif [[ "$APPLY_CONFIG" == "true" ]]; then
        validate_configuration "$ENVIRONMENT" && apply_configuration "$ENVIRONMENT"
    else
        # Default: show status
        show_status "$ENVIRONMENT"
    fi

    print_success "Setup completed for environment: $ENVIRONMENT"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi