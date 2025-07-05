"""Configuration workflow component for modular integration testing.

This module provides reusable workflow components for testing configuration
loading, validation, and directory setup scenarios.
"""

from pathlib import Path
from typing import Any, Protocol

import yaml


class TestUtilities(Protocol):
    """Protocol for test utilities needed by workflow components."""

    temp_path: Path


class ConfigurationWorkflowComponent:
    """Modular component for configuration workflow testing."""

    def __init__(self, test_utilities: TestUtilities) -> None:
        """Initialize with test utilities for shared resources."""
        self.test_utilities = test_utilities

    def execute_config_loading_workflow(
        self, config_path: Path, expect_success: bool = True
    ) -> dict[str, Any]:
        """Execute complete configuration loading workflow.

        Args:
            config_path: Path to configuration file
            expect_success: Whether to expect successful loading

        Returns:
            Result dictionary with workflow outcome and details
        """
        workflow_result: dict[str, Any] = {
            "step": "config_loading",
            "success": False,
            "config_path": str(config_path),
            "config_loaded": False,
            "validation_errors": [],
            "session_state_updated": False,
        }

        try:
            # Step 1: Validate file exists
            if not config_path.exists():
                workflow_result["validation_errors"].append(
                    "File does not exist"
                )
                return workflow_result

            # Step 2: Load and validate YAML
            with open(config_path, encoding="utf-8") as f:
                config_content = yaml.safe_load(f)

            # Step 3: Validate required sections
            required_sections = ["model", "training"]
            missing_sections = [
                section
                for section in required_sections
                if section not in config_content
            ]

            if missing_sections:
                workflow_result["validation_errors"].extend(
                    [
                        f"Missing section: {section}"
                        for section in missing_sections
                    ]
                )

            # Step 4: Simulate session state update
            if not workflow_result["validation_errors"]:
                workflow_result["config_loaded"] = True
                workflow_result["session_state_updated"] = True
                workflow_result["success"] = True

        except yaml.YAMLError as e:
            workflow_result["validation_errors"].append(
                f"YAML parsing error: {e}"
            )
        except Exception as e:
            workflow_result["validation_errors"].append(
                f"Unexpected error: {e}"
            )

        # Verify expected outcome
        if expect_success and not workflow_result["success"]:
            raise AssertionError(
                f"Expected successful config loading but failed: "
                f"{workflow_result['validation_errors']}"
            )
        elif not expect_success and workflow_result["success"]:
            raise AssertionError(
                "Expected config loading to fail but it succeeded"
            )

        return workflow_result

    def execute_directory_setup_workflow(
        self, directory_path: Path, create_if_missing: bool = True
    ) -> dict[str, Any]:
        """Execute complete directory setup workflow.

        Args:
            directory_path: Path to working directory
            create_if_missing: Whether to create directory if it doesn't exist

        Returns:
            Result dictionary with workflow outcome
        """
        workflow_result: dict[str, Any] = {
            "step": "directory_setup",
            "success": False,
            "directory_path": str(directory_path),
            "directory_exists": False,
            "directory_created": False,
            "subdirectories_created": False,
        }

        try:
            # Step 1: Check if directory exists
            workflow_result["directory_exists"] = directory_path.exists()

            # Step 2: Create directory if needed and allowed
            if not workflow_result["directory_exists"] and create_if_missing:
                directory_path.mkdir(parents=True, exist_ok=True)
                workflow_result["directory_created"] = True
                workflow_result["directory_exists"] = True

            # Step 3: Create standard subdirectories
            if workflow_result["directory_exists"]:
                subdirs = ["logs", "checkpoints", "results", "configs"]
                for subdir in subdirs:
                    (directory_path / subdir).mkdir(exist_ok=True)
                workflow_result["subdirectories_created"] = True
                workflow_result["success"] = True

        except Exception as e:
            workflow_result["error"] = str(e)

        return workflow_result
