"""File generation utilities for packaging system.

This module handles creation of application files, entrypoints,
health checks, configuration files, and deployment scripts.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import DeploymentConfig

logger = logging.getLogger(__name__)


class FileGenerator:
    """Generates application files for packaging."""

    def __init__(self) -> None:
        """Initialize file generator."""
        self.logger = logging.getLogger(__name__)

    def create_app_entrypoint(
        self, package_dir: Path, config: "DeploymentConfig"
    ) -> None:
        """Create application entrypoint script.

        Args:
            package_dir: Package directory
            config: Deployment configuration
        """
        entrypoint_content = self._generate_entrypoint_content(config)
        entrypoint_path = package_dir / "app.py"
        entrypoint_path.write_text(entrypoint_content)
        entrypoint_path.chmod(0o755)

        self.logger.info(f"Created entrypoint: {entrypoint_path}")

    def create_health_check(self, package_dir: Path) -> None:
        """Create health check endpoint.

        Args:
            package_dir: Package directory
        """
        health_check_content = '''"""Health check endpoint for deployment."""

import json
import time
from pathlib import Path

def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    print(json.dumps(health_check()))
'''
        health_path = package_dir / "health_check.py"
        health_path.write_text(health_check_content)
        health_path.chmod(0o755)

        self.logger.info(f"Created health check: {health_path}")

    def create_config_files(
        self, package_dir: Path, config: "DeploymentConfig"
    ) -> None:
        """Create configuration files.

        Args:
            package_dir: Package directory
            config: Deployment configuration
        """
        # Create config directory
        config_dir = package_dir / "config"
        config_dir.mkdir(exist_ok=True)

        # Environment configuration
        env_config = self._generate_env_config(config)
        env_path = config_dir / "environment.yaml"
        env_path.write_text(env_config)

        # Logging configuration
        logging_config = self._generate_logging_config()
        logging_path = config_dir / "logging.yaml"
        logging_path.write_text(logging_config)

        self.logger.info(f"Created config files in: {config_dir}")

    def create_deployment_scripts(
        self, package_dir: Path, config: "DeploymentConfig"
    ) -> None:
        """Create deployment scripts.

        Args:
            package_dir: Package directory
            config: Deployment configuration
        """
        scripts_dir = package_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)

        # Start script
        start_script = self._generate_start_script(config)
        start_path = scripts_dir / "start.sh"
        start_path.write_text(start_script)
        start_path.chmod(0o755)

        # Stop script
        stop_script = self._generate_stop_script()
        stop_path = scripts_dir / "stop.sh"
        stop_path.write_text(stop_script)
        stop_path.chmod(0o755)

        self.logger.info(f"Created deployment scripts in: {scripts_dir}")

    def _generate_entrypoint_content(self, config: "DeploymentConfig") -> str:
        """Generate entrypoint script content."""
        return f'''"""CrackSeg application entrypoint."""

import os
import sys
from pathlib import Path

# Add package to path
package_dir = Path(__file__).parent
sys.path.insert(0, str(package_dir))

def main():
    """Main application entrypoint."""
    # Set environment variables
    os.environ.setdefault("CRACKSEG_ENV", "{config.target_environment}")
    os.environ.setdefault(
        "CRACKSEG_DEPLOYMENT_TYPE", "{config.deployment_type}"
    )

    # Import and run application
    from crackseg.app import create_app
    app = create_app()

    # Run based on deployment type
    if "{config.deployment_type}" == "serverless":
        # Lambda/Cloud Function entry point
        return app
    else:
        # Container/Server deployment
        host = os.environ.get("HOST", "0.0.0.0")
        port = int(os.environ.get("PORT", 8000))
        app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    main()
'''

    def _generate_env_config(self, config: "DeploymentConfig") -> str:
        """Generate environment configuration."""
        return f"""# Environment configuration
environment: {config.target_environment}
deployment_type: {config.deployment_type}
model_path: /app/models/crackseg_model.pth
log_level: INFO
enable_health_checks: {config.enable_health_checks}
enable_metrics_collection: {config.enable_metrics_collection}
"""

    def _generate_logging_config(self) -> str:
        """Generate logging configuration."""
        return """# Logging configuration
version: 1
formatters:
  standard:
    format: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout
loggers:
  crackseg:
    level: INFO
    handlers: [console]
    propagate: no
root:
  level: INFO
  handlers: [console]
"""

    def _generate_start_script(self, config: "DeploymentConfig") -> str:
        """Generate start script."""
        return f"""#!/bin/bash
# Start script for CrackSeg deployment

set -e

echo "Starting CrackSeg {config.deployment_type} deployment..."

# Set environment
export CRACKSEG_ENV={config.target_environment}
export CRACKSEG_DEPLOYMENT_TYPE={config.deployment_type}

# Start application
python app.py
"""

    def _generate_stop_script(self) -> str:
        """Generate stop script."""
        return """#!/bin/bash
# Stop script for CrackSeg deployment

echo "Stopping CrackSeg deployment..."

# Find and kill the application process
pkill -f "python app.py" || true

echo "CrackSeg deployment stopped."
"""
