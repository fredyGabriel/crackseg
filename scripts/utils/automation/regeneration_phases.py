"""Regeneration phases for post-phase workflow.

This module contains the individual regeneration functions for different
project phases like documentation, API docs, test reports, etc.
"""

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "scripts" / "utils" / "automation"))

from simple_mapping_registry import SimpleMappingRegistry  # noqa: E402


def regenerate_project_tree(registry: SimpleMappingRegistry) -> dict[str, Any]:
    """Regenerate the project tree documentation.

    Args:
        registry: Mapping registry for path validation

    Returns:
        Dictionary with regeneration results
    """
    logger = logging.getLogger(__name__)
    logger.info("🌳 Regenerating project tree...")

    try:
        # Run the project tree generation script
        result = subprocess.run(
            [
                sys.executable,
                "scripts/utils/documentation/generate_project_tree.py",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=300,  # 5 minutes timeout
        )

        if result.returncode == 0:
            logger.info("✅ Project tree regenerated successfully")
            return {
                "success": True,
                "output": result.stdout,
                "timestamp": datetime.now().isoformat(),
            }
        else:
            logger.error(
                f"❌ Project tree regeneration failed: {result.stderr}"
            )
            return {
                "success": False,
                "error": result.stderr,
                "timestamp": datetime.now().isoformat(),
            }

    except subprocess.TimeoutExpired:
        error_msg = "Project tree regeneration timed out"
        logger.error(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        error_msg = f"Project tree regeneration failed: {e}"
        logger.error(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        }


def regenerate_api_documentation(
    registry: SimpleMappingRegistry,
) -> dict[str, Any]:
    """Regenerate API documentation.

    Args:
        registry: Mapping registry for path validation

    Returns:
        Dictionary with regeneration results
    """
    logger = logging.getLogger(__name__)
    logger.info("📚 Regenerating API documentation...")

    try:
        # Run the API documentation generation script
        result = subprocess.run(
            [
                sys.executable,
                "scripts/utils/documentation/generate_api_docs.py",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=600,  # 10 minutes timeout
        )

        if result.returncode == 0:
            logger.info("✅ API documentation regenerated successfully")
            return {
                "success": True,
                "output": result.stdout,
                "timestamp": datetime.now().isoformat(),
            }
        else:
            logger.error(
                f"❌ API documentation regeneration failed: {result.stderr}"
            )
            return {
                "success": False,
                "error": result.stderr,
                "timestamp": datetime.now().isoformat(),
            }

    except subprocess.TimeoutExpired:
        error_msg = "API documentation regeneration timed out"
        logger.error(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        error_msg = f"API documentation regeneration failed: {e}"
        logger.error(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        }


def regenerate_test_reports(registry: SimpleMappingRegistry) -> dict[str, Any]:
    """Regenerate test reports and coverage documentation.

    Args:
        registry: Mapping registry for path validation

    Returns:
        Dictionary with regeneration results
    """
    logger = logging.getLogger(__name__)
    logger.info("🧪 Regenerating test reports...")

    try:
        # Run the test report generation script
        result = subprocess.run(
            [
                sys.executable,
                "scripts/utils/documentation/generate_test_reports.py",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=300,  # 5 minutes timeout
        )

        if result.returncode == 0:
            logger.info("✅ Test reports regenerated successfully")
            return {
                "success": True,
                "output": result.stdout,
                "timestamp": datetime.now().isoformat(),
            }
        else:
            logger.error(
                f"❌ Test reports regeneration failed: {result.stderr}"
            )
            return {
                "success": False,
                "error": result.stderr,
                "timestamp": datetime.now().isoformat(),
            }

    except subprocess.TimeoutExpired:
        error_msg = "Test reports regeneration timed out"
        logger.error(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        error_msg = f"Test reports regeneration failed: {e}"
        logger.error(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        }


def update_mapping_registry(registry: SimpleMappingRegistry) -> dict[str, Any]:
    """Update the mapping registry with any new path changes.

    Args:
        registry: Mapping registry to update

    Returns:
        Dictionary with update results
    """
    logger = logging.getLogger(__name__)
    logger.info("🗂️  Updating mapping registry...")

    try:
        # Save the current registry
        registry.save_registry()

        logger.info("✅ Mapping registry updated successfully")
        return {
            "success": True,
            "mappings_count": len(registry.mappings),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        error_msg = f"Mapping registry update failed: {e}"
        logger.error(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        }


def run_consistency_checks(registry: SimpleMappingRegistry) -> dict[str, Any]:
    """Run consistency checks after regeneration.

    Args:
        registry: Mapping registry for validation

    Returns:
        Dictionary with check results
    """
    logger = logging.getLogger(__name__)
    logger.info("🔍 Running consistency checks...")

    try:
        # Run the CI consistency checker
        result = subprocess.run(
            [
                sys.executable,
                "scripts/utils/quality/guardrails/ci_consistency_checker.py",
                "--skip-imports",  # Skip import checks for now
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=300,  # 5 minutes timeout
        )

        # Parse the output to extract statistics
        output = result.stdout
        errors = 0
        warnings = 0

        # Extract error and warning counts from output
        for line in output.split("\n"):
            if "Total Errors:" in line:
                try:
                    errors = int(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
            elif "Total Warnings:" in line:
                try:
                    warnings = int(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass

        logger.info(
            f"✅ Consistency checks completed: {errors} errors, {warnings} warnings"
        )
        return {
            "success": True,
            "errors": errors,
            "warnings": warnings,
            "output": output,
            "timestamp": datetime.now().isoformat(),
        }

    except subprocess.TimeoutExpired:
        error_msg = "Consistency checks timed out"
        logger.error(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        error_msg = f"Consistency checks failed: {e}"
        logger.error(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        }
