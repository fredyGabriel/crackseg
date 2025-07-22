"""Base executor for performance maintenance operations."""

import logging
import subprocess
from pathlib import Path

from .utils import MaintenanceLog, log_action


class BaseMaintenanceExecutor:
    """Base class for performance maintenance executors."""

    def __init__(self, paths: dict[str, Path], logger: logging.Logger) -> None:
        """Initialize base maintenance executor.

        Args:
            paths: Dictionary of project paths
            logger: Logger instance for persistent logging
        """
        self.paths = paths
        self.logger = logger
        self.maintenance_log: MaintenanceLog = []

    def run_command(
        self, command: list[str], description: str
    ) -> tuple[bool, str]:
        """Run a system command with logging.

        Args:
            command: List of command arguments
            description: Description of the command for logging

        Returns:
            Tuple of (success: bool, output: str)
        """
        try:
            self.logger.info(f"Running: {description}")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )

            if result.returncode == 0:
                log_action(f"SUCCESS: {description}", "INFO")
                self.logger.info(f"Command succeeded: {description}")
                return True, result.stdout
            else:
                log_action(f"FAILED: {description}", "ERROR")
                self.logger.error(f"Command failed: {description}")
                self.logger.error(f"Error output: {result.stderr}")
                return False, result.stderr

        except subprocess.TimeoutExpired:
            log_action(f"TIMEOUT: {description}", "ERROR")
            self.logger.error(f"Command timed out: {description}")
            return False, "Command timed out"
        except Exception as e:
            log_action(f"EXCEPTION: {description} - {e}", "ERROR")
            self.logger.error(f"Command exception: {description} - {e}")
            return False, str(e)

    def check_file_exists(self, file_path: Path, description: str) -> bool:
        """Check if a file exists with logging.

        Args:
            file_path: Path to the file to check
            description: Description for logging

        Returns:
            True if file exists, False otherwise
        """
        exists = file_path.exists()
        if exists:
            log_action(f"FOUND: {description}", "INFO")
            self.logger.info(f"File found: {file_path}")
        else:
            log_action(f"MISSING: {description}", "WARNING")
            self.logger.warning(f"File not found: {file_path}")
        return exists
