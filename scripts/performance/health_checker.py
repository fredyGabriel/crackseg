"""Health checker for performance benchmarking system."""

import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from .base_executor import BaseMaintenanceExecutor


class HealthChecker(BaseMaintenanceExecutor):
    """Performs comprehensive health checks for the performance system."""

    def __init__(self, paths: dict[str, Path], logger: logging.Logger) -> None:
        """Initialize the health checker.

        Args:
            paths: Dictionary of project paths
            logger: Logger instance for persistent logging
        """
        super().__init__(paths, logger)

    def health_check(self) -> dict[str, Any]:
        """Perform comprehensive system health check.

        Returns:
            Dictionary containing health check results and overall status
        """
        self.logger.info("Starting performance system health check...")

        health_results: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "checks": {},
            "warnings": [],
            "errors": [],
        }

        # Check Python environment and dependencies
        try:
            import_cmd = (
                "from src.crackseg.utils.monitoring.resource_monitor "
                "import ResourceMonitor; print('OK')"
            )
            result = subprocess.run(
                ["python", "-c", import_cmd],
                capture_output=True,
                text=True,
                timeout=30,
            )
            health_results["checks"]["python_environment"] = {
                "status": "success" if result.returncode == 0 else "error",
                "details": (
                    result.stdout if result.returncode == 0 else result.stderr
                ),
            }
        except subprocess.TimeoutExpired:
            health_results["checks"]["python_environment"] = {
                "status": "error",
                "details": "Python environment check timed out",
            }
        except Exception as e:
            health_results["checks"]["python_environment"] = {
                "status": "error",
                "details": f"Python environment check failed: {e}",
            }

        # Check project structure integrity
        try:
            structure_check = self._check_project_structure()
            health_results["checks"]["project_structure"] = structure_check
        except Exception as e:
            health_results["errors"].append(
                f"Project structure check failed: {e}"
            )

        # Check essential imports and modules
        try:
            imports_check = self._check_essential_imports()
            health_results["checks"]["essential_imports"] = imports_check
        except Exception as e:
            health_results["errors"].append(
                f"Essential imports check failed: {e}"
            )

        # Determine overall status
        if health_results["errors"]:
            health_results["overall_status"] = "error"
        elif health_results["warnings"]:
            health_results["overall_status"] = "warning"
        else:
            health_results["overall_status"] = "healthy"

        status = health_results["overall_status"]
        self.logger.info(f"Health check completed with status: {status}")
        return health_results

    def _check_python_environment(self) -> dict[str, Any]:
        """Check Python environment and key dependencies."""
        try:
            result = subprocess.run(
                [
                    "python",
                    "-c",
                    (
                        "from src.crackseg.utils.monitoring.resource_monitor "
                        "import ResourceMonitor; print('OK')"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return {
                "status": "success" if result.returncode == 0 else "error",
                "details": (
                    result.stdout if result.returncode == 0 else result.stderr
                ),
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "details": "Python environment check timed out",
            }
        except Exception as e:
            return {
                "status": "error",
                "details": f"Python environment check failed: {e}",
            }

    def _check_project_structure(self) -> dict[str, Any]:
        """Check essential project directories and files."""
        essential_paths = [
            self.paths.get("project_root", Path(".")),
            self.paths.get("src_dir", Path("src")),
            self.paths.get("tests_dir", Path("tests")),
            self.paths.get("scripts_dir", Path("scripts")),
        ]

        missing_paths = []
        for path in essential_paths:
            if not path.exists():
                missing_paths.append(str(path))

        return {
            "status": "success" if not missing_paths else "error",
            "details": {"missing_paths": missing_paths},
        }

    def _check_essential_imports(self) -> dict[str, Any]:
        """Check if essential modules can be imported."""
        essential_modules = [
            "src.crackseg",
            "src.crackseg.data",
            "src.crackseg.model",
            "src.crackseg.training",
        ]

        import_errors = []
        for module in essential_modules:
            try:
                __import__(module)
            except ImportError as e:
                import_errors.append(f"{module}: {e}")

        return {
            "status": "success" if not import_errors else "error",
            "details": {"import_errors": import_errors},
        }
