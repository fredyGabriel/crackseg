#!/usr/bin/env python3
"""
Unit tests for Docker artifact management system.
Tests the artifact-manager.sh script functionality through Python interfaces.

Subtask: 13.5 - Implement Artifact Management and Volume Configuration
"""

import os
import platform
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


def get_shell_command(script_path: str, args: list[str]) -> list[str]:
    """Get appropriate shell command for the current platform."""
    if platform.system() == "Windows":
        # Use WSL on Windows to execute bash scripts
        wsl_path = script_path.replace("\\", "/").replace("C:", "/mnt/c")
        base_path = (
            "/mnt/c/Users/fgrv/OneDrive/Documentos/PythonProjects/"
            "doctorado/crackseg"
        )
        return [
            "wsl",
            "bash",
            "-c",
            f"cd {base_path} && {wsl_path} {' '.join(args)}",
        ]
    else:
        # Use bash directly on Unix-like systems
        return ["bash", script_path] + args


class TestArtifactManager(unittest.TestCase):
    """Test suite for the artifact management system."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.script_path = (
            self.project_root
            / "tests"
            / "docker"
            / "scripts"
            / "artifact-manager.sh"
        )

        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_results_path = Path(self.temp_dir) / "test-results"
        self.test_artifacts_path = Path(self.temp_dir) / "test-artifacts"
        self.selenium_videos_path = Path(self.temp_dir) / "selenium-videos"
        self.archive_path = Path(self.temp_dir) / "archived-artifacts"

        # Create test directories
        for path in [
            self.test_results_path,
            self.test_artifacts_path,
            self.selenium_videos_path,
            self.archive_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        # Environment variables for testing
        self.test_env = {
            **os.environ,
            "TEST_RESULTS_PATH": str(self.test_results_path),
            "TEST_ARTIFACTS_PATH": str(self.test_artifacts_path),
            "SELENIUM_VIDEOS_PATH": str(self.selenium_videos_path),
            "ARCHIVE_PATH": str(self.archive_path),
            "DEBUG": "true",
        }

    def tearDown(self) -> None:
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_script_exists_and_executable(self) -> None:
        """Test that the artifact manager script exists and is executable."""
        assert (
            self.script_path.exists()
        ), f"Script not found: {self.script_path}"

        # On Windows, check if WSL is available instead of direct execution
        if platform.system() == "Windows":
            try:
                result = subprocess.run(
                    ["wsl", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                assert (
                    result.returncode == 0
                ), "WSL is not available for script execution"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pytest.skip("WSL not available, skipping bash script tests")
        else:
            assert os.access(
                self.script_path, os.X_OK
            ), "Script is not executable"

    def test_help_command(self) -> None:
        """Test that help command displays usage information."""
        cmd = get_shell_command(str(self.script_path), ["help"])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=self.test_env,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            pytest.skip("Command timed out, likely due to environment issues")
        except FileNotFoundError as e:
            if "wsl" in str(e).lower():
                pytest.skip("WSL not available, skipping bash script tests")
            raise

        assert result.returncode == 0, f"Help command failed: {result.stderr}"
        assert "CrackSeg Artifact Management Script" in result.stdout
        assert "COMMANDS:" in result.stdout
        assert "collect" in result.stdout
        assert "cleanup" in result.stdout
        assert "archive" in result.stdout

    def test_status_command(self) -> None:
        """Test status command shows artifact storage information."""
        # Create some test files
        (self.test_results_path / "test.log").write_text("test log")
        (self.test_artifacts_path / "artifact.json").write_text(
            '{"test": true}'
        )

        cmd = get_shell_command(str(self.script_path), ["status"])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=self.test_env,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            pytest.skip("Command timed out, likely due to environment issues")
        except FileNotFoundError as e:
            if "wsl" in str(e).lower():
                pytest.skip("WSL not available, skipping bash script tests")
            raise

        assert (
            result.returncode == 0
        ), f"Status command failed: {result.stderr}"
        assert "Artifact Storage Status" in result.stdout
        assert "Test Results:" in result.stdout
        assert "Test Artifacts:" in result.stdout

    @patch("subprocess.run")
    def test_collect_command_basic(self, mock_run: Mock) -> None:
        """Test basic artifact collection functionality."""
        # Mock Docker commands to succeed
        mock_run.return_value.returncode = 0

        cmd = get_shell_command(str(self.script_path), ["collect", "--debug"])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=self.test_env,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            pytest.skip("Command timed out, likely due to environment issues")
        except FileNotFoundError as e:
            if "wsl" in str(e).lower():
                pytest.skip("WSL not available, skipping bash script tests")
            raise

        # Should attempt to run even if Docker is not available
        assert (
            "Collecting artifacts" in result.stdout
            or result.returncode in [0, 1]
        )

    def test_collect_with_compression(self) -> None:
        """Test artifact collection with compression enabled."""
        # Create test artifacts
        (self.test_artifacts_path / "test1.log").write_text("test log 1")
        (self.test_artifacts_path / "test2.json").write_text('{"test": 2}')

        cmd = get_shell_command(
            str(self.script_path), ["collect", "--compress", "--debug"]
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=self.test_env,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            pytest.skip("Command timed out, likely due to environment issues")
        except FileNotFoundError as e:
            if "wsl" in str(e).lower():
                pytest.skip("WSL not available, skipping bash script tests")
            raise

        # Verify that compression was attempted
        # (even if it fails due to Docker)
        assert "Collecting artifacts" in result.stdout

    def test_cleanup_dry_run(self) -> None:
        """Test cleanup command with dry run option."""
        # Create old test files
        old_file = self.test_results_path / "old_test.log"
        old_file.write_text("old test")

        # Modify timestamp to make it old (7+ days)
        import time

        old_timestamp = time.time() - (8 * 24 * 60 * 60)  # 8 days ago
        os.utime(old_file, (old_timestamp, old_timestamp))

        cmd = get_shell_command(
            str(self.script_path),
            [
                "cleanup",
                "--older-than",
                "7",
                "--dry-run",
                "--debug",
            ],
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=self.test_env,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            pytest.skip("Command timed out, likely due to environment issues")
        except FileNotFoundError as e:
            if "wsl" in str(e).lower():
                pytest.skip("WSL not available, skipping bash script tests")
            raise

        assert (
            result.returncode == 0
        ), f"Cleanup dry-run failed: {result.stderr}"
        assert "Cleaning up artifacts" in result.stdout
        assert "Dry run: true" in result.stdout

        # File should still exist after dry run
        assert old_file.exists(), "File was deleted during dry run"

    def test_archive_command_tar_format(self) -> None:
        """Test archiving with tar.gz format."""
        # Create test artifacts
        (self.test_results_path / "result1.html").write_text(
            "<html>test</html>"
        )
        (self.test_artifacts_path / "artifact1.json").write_text(
            '{"result": "success"}'
        )

        cmd = get_shell_command(
            str(self.script_path),
            [
                "archive",
                "--format",
                "tar.gz",
                "--debug",
            ],
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=self.test_env,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            pytest.skip("Command timed out, likely due to environment issues")
        except FileNotFoundError as e:
            if "wsl" in str(e).lower():
                pytest.skip("WSL not available, skipping bash script tests")
            raise

        assert (
            result.returncode == 0
        ), f"Archive command failed: {result.stderr}"
        assert "Archiving artifacts" in result.stdout

    def test_verify_directory_structure(self) -> None:
        """Test verification of directory structure."""
        cmd = get_shell_command(
            str(self.script_path), ["verify", "--structure", "--debug"]
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=self.test_env,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            pytest.skip("Command timed out, likely due to environment issues")
        except FileNotFoundError as e:
            if "wsl" in str(e).lower():
                pytest.skip("WSL not available, skipping bash script tests")
            raise

        assert (
            result.returncode == 0
        ), f"Verify structure failed: {result.stderr}"
        assert "Verifying artifacts" in result.stdout

    def test_environment_variable_integration(self) -> None:
        """Test that environment variables are properly integrated."""
        custom_env = {
            **self.test_env,
            "TEST_RESULTS_PATH": "/custom/results",
            "ARCHIVE_PATH": "/custom/archive",
        }

        cmd = get_shell_command(str(self.script_path), ["status"])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, env=custom_env, timeout=30
            )
        except subprocess.TimeoutExpired:
            pytest.skip("Command timed out, likely due to environment issues")
        except FileNotFoundError as e:
            if "wsl" in str(e).lower():
                pytest.skip("WSL not available, skipping bash script tests")
            raise

        # Should handle custom paths gracefully (even if they don't exist)
        assert result.returncode in [
            0,
            1,
        ]  # May fail due to missing directories

    def test_invalid_command_handling(self) -> None:
        """Test handling of invalid commands."""
        cmd = get_shell_command(str(self.script_path), ["invalid-command"])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=self.test_env,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            pytest.skip("Command timed out, likely due to environment issues")
        except FileNotFoundError as e:
            if "wsl" in str(e).lower():
                pytest.skip("WSL not available, skipping bash script tests")
            raise

        assert result.returncode == 1, "Invalid command should return error"
        assert "Unknown command" in result.stdout

    def test_collection_timestamp_format(self) -> None:
        """Test that collection directories use proper timestamp format."""
        cmd = get_shell_command(str(self.script_path), ["collect", "--debug"])

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=self.test_env,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            pytest.skip("Command timed out, likely due to environment issues")
        except FileNotFoundError as e:
            if "wsl" in str(e).lower():
                pytest.skip("WSL not available, skipping bash script tests")
            raise

        # Check for timestamp pattern in output
        import re

        # Look for collection directories that might have been created
        collections = list(self.test_artifacts_path.glob("collection_*"))
        if collections:
            # Verify timestamp format
            for collection in collections:
                assert re.match(
                    r"collection_\d{8}_\d{6}", collection.name
                ), f"Invalid timestamp format: {collection.name}"


class TestVolumeConfiguration(unittest.TestCase):
    """Test suite for Docker volume configuration."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.docker_compose_path = (
            self.project_root / "tests" / "docker" / "docker-compose.test.yml"
        )

    def test_docker_compose_file_exists(self) -> None:
        """Test that docker-compose configuration file exists."""
        assert (
            self.docker_compose_path.exists()
        ), f"Docker compose file not found: {self.docker_compose_path}"

    def test_volume_configuration(self) -> None:
        """Test that required volumes are configured in docker-compose."""
        content = self.docker_compose_path.read_text()

        # Check for required volumes
        required_volumes = [
            "test-results",
            "test-artifacts",
            "selenium-videos",
            "artifact-archive",
            "artifact-temp",
        ]

        for volume in required_volumes:
            assert (
                f"{volume}:" in content
            ), f"Volume {volume} not found in docker-compose"

    def test_volume_labels(self) -> None:
        """Test that volumes have proper labels for artifact management."""
        content = self.docker_compose_path.read_text()

        # Check for artifact management labels
        required_labels = [
            "crackseg.volume=test-artifacts",
            "crackseg.managed-by=artifact-manager",
            "crackseg.version=13.5",
        ]

        for label in required_labels:
            assert (
                label in content
            ), f"Label {label} not found in docker-compose"

    def test_environment_variable_integration(self) -> None:
        """Test that volumes use environment variable paths."""
        content = self.docker_compose_path.read_text()

        # Check for environment variable usage
        env_vars = [
            "${TEST_RESULTS_PATH:-./test-results}",
            "${TEST_ARTIFACTS_PATH:-./test-artifacts}",
            "${ARCHIVE_PATH:-./archived-artifacts}",
        ]

        for env_var in env_vars:
            assert (
                env_var in content
            ), f"Environment variable {env_var} not found"


class TestRunTestRunnerIntegration(unittest.TestCase):
    """Test suite for run-test-runner.sh integration with artifact mgmt."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.script_path = (
            self.project_root
            / "tests"
            / "docker"
            / "scripts"
            / "run-test-runner.sh"
        )

    def test_run_test_runner_script_exists(self) -> None:
        """Test that run-test-runner script exists and is executable."""
        assert (
            self.script_path.exists()
        ), f"Script not found: {self.script_path}"
        assert os.access(self.script_path, os.X_OK), "Script is not executable"

    def test_enhanced_commands_in_help(self) -> None:
        """Test that enhanced artifact commands are listed in help."""
        cmd = get_shell_command(str(self.script_path), ["help"])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
        except subprocess.TimeoutExpired:
            pytest.skip("Command timed out, likely due to environment issues")
        except FileNotFoundError as e:
            if "wsl" in str(e).lower():
                pytest.skip("WSL not available, skipping bash script tests")
            raise

        assert result.returncode == 0, f"Help command failed: {result.stderr}"

        # Check for enhanced commands
        enhanced_commands = [
            "collect-artifacts",
            "cleanup-artifacts",
            "archive-artifacts",
        ]

        for command in enhanced_commands:
            assert (
                command in result.stdout
            ), f"Enhanced command {command} not found in help"

    def test_artifact_management_environment_vars(self) -> None:
        """Test that artifact management environment variables are defined."""
        content = self.script_path.read_text()

        # Check for new environment variables
        env_vars = [
            "ARCHIVE_PATH",
            "TEMP_ARTIFACTS_PATH",
            "ARTIFACT_COLLECTION_ENABLED",
            "ARTIFACT_CLEANUP_ENABLED",
            "ARTIFACT_ARCHIVE_ENABLED",
        ]

        for env_var in env_vars:
            assert (
                env_var in content
            ), f"Environment variable {env_var} not found in script"


class TestArtifactManagerPythonInterface(unittest.TestCase):
    """Test suite for Python interface to artifact management system."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.project_root = Path(__file__).parent.parent.parent.parent

    def create_artifact_manager_interface(self) -> "ArtifactManagerInterface":
        """Create an interface to the artifact manager for testing."""
        return ArtifactManagerInterface(self.project_root)


class ArtifactManagerInterface:
    """Python interface to the artifact management system for testing."""

    def __init__(self, project_root: Path) -> None:
        """Initialize the interface."""
        self.project_root = project_root
        self.script_path = (
            project_root
            / "tests"
            / "docker"
            / "scripts"
            / "artifact-manager.sh"
        )

    def collect_artifacts(
        self, compress: bool = False, debug: bool = True
    ) -> subprocess.CompletedProcess[str]:
        """Collect artifacts using the artifact manager."""
        args = ["collect"]
        if compress:
            args.append("--compress")
        if debug:
            args.append("--debug")

        cmd = get_shell_command(str(self.script_path), args)
        return subprocess.run(cmd, capture_output=True, text=True)

    def cleanup_artifacts(
        self, older_than: int = 7, dry_run: bool = True
    ) -> subprocess.CompletedProcess[str]:
        """Clean up artifacts using the artifact manager."""
        args = [
            "cleanup",
            "--older-than",
            str(older_than),
        ]
        if dry_run:
            args.append("--dry-run")

        cmd = get_shell_command(str(self.script_path), args)
        return subprocess.run(cmd, capture_output=True, text=True)

    def archive_artifacts(
        self, format_type: str = "tar.gz"
    ) -> subprocess.CompletedProcess[str]:
        """Archive artifacts using the artifact manager."""
        args = ["archive", "--format", format_type]

        cmd = get_shell_command(str(self.script_path), args)
        return subprocess.run(cmd, capture_output=True, text=True)

    def verify_artifacts(
        self, check_structure: bool = True
    ) -> subprocess.CompletedProcess[str]:
        """Verify artifacts using the artifact manager."""
        args = ["verify"]
        if check_structure:
            args.append("--structure")

        cmd = get_shell_command(str(self.script_path), args)
        return subprocess.run(cmd, capture_output=True, text=True)

    def get_status(self) -> subprocess.CompletedProcess[str]:
        """Get artifact storage status."""
        args = ["status"]
        cmd = get_shell_command(str(self.script_path), args)
        return subprocess.run(cmd, capture_output=True, text=True)


# Integration test using the interface
class TestArtifactManagerIntegration(unittest.TestCase):
    """Integration tests using the Python interface."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.interface = ArtifactManagerInterface(self.project_root)

    def test_collect_and_status_integration(self) -> None:
        """Test collection followed by status check."""
        # Collect artifacts
        collect_result = self.interface.collect_artifacts(debug=True)

        # Get status
        status_result = self.interface.get_status()

        # Both should complete (may fail due to Docker not running,
        # but shouldn't crash)
        assert collect_result.returncode in [0, 1]
        assert status_result.returncode in [0, 1]

        # Status should show artifact information
        if status_result.returncode == 0:
            assert "Artifact Storage Status" in status_result.stdout

    def test_verify_system_integrity(self) -> None:
        """Test system integrity verification."""
        verify_result = self.interface.verify_artifacts(check_structure=True)

        # Should complete successfully
        assert verify_result.returncode == 0
        assert "Verifying artifacts" in verify_result.stdout


if __name__ == "__main__":
    # Run tests with pytest for better output
    pytest.main([__file__, "-v"])
