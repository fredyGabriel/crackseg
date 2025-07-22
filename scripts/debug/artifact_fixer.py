"""Artifact fixing utilities for training experiments."""

import json
import shutil
from pathlib import Path

import yaml

from .checkpoint_validator import CheckpointValidator
from .utils import DiagnosticResult


class ArtifactFixer:
    """Tools to fix common artifact issues."""

    @staticmethod
    def fix_configuration_directory(config_dir: Path) -> bool:
        """
        Attempt to fix common configuration directory issues.

        Args:
            config_dir: Path to configuration directory to fix

        Returns:
            True if fixes were applied successfully
        """
        print(f"🔧 Attempting to fix configuration directory: {config_dir}")

        if not config_dir.exists():
            print("Creating missing configuration directory...")
            config_dir.mkdir(parents=True, exist_ok=True)
            return True

        # Look for orphaned config files
        yaml_files = list(config_dir.glob("**/*.yaml"))
        json_files = list(config_dir.glob("**/*.json"))

        print(f"Found {len(yaml_files)} YAML and {len(json_files)} JSON files")

        # Basic validation of found files
        for config_file in yaml_files + json_files:
            try:
                if config_file.suffix == ".yaml":
                    with open(config_file, encoding="utf-8") as f:
                        yaml.safe_load(f)
                else:
                    with open(config_file, encoding="utf-8") as f:
                        json.load(f)
                print(f"✅ {config_file.name} is valid")
            except Exception as e:
                print(f"❌ {config_file.name} is corrupted: {e}")
                # Optionally attempt to fix the corrupted file
                ArtifactFixer._attempt_config_repair(config_file)

        return True

    @staticmethod
    def _attempt_config_repair(config_file: Path) -> bool:
        """
        Attempt to repair a corrupted configuration file.

        Args:
            config_file: Path to the corrupted configuration file

        Returns:
            True if repair was successful
        """
        print(f"🔧 Attempting to repair: {config_file.name}")

        try:
            # Create a backup
            backup_file = config_file.with_suffix(
                config_file.suffix + ".backup"
            )
            config_file.rename(backup_file)

            # Try to read and clean the file
            with open(backup_file, encoding="utf-8") as f:
                content = f.read()

            # Basic cleanup: remove invalid characters, fix syntax errors
            content = content.replace("\0", "")  # Remove null bytes
            content = content.replace(
                "\ufffd", ""
            )  # Remove replacement characters

            # Try to save cleaned version
            with open(config_file, "w", encoding="utf-8") as f:
                f.write(content)

            # Validate the cleaned version
            if config_file.suffix == ".yaml":
                with open(config_file, encoding="utf-8") as f:
                    yaml.safe_load(f)
            else:
                with open(config_file, encoding="utf-8") as f:
                    json.load(f)

            print(f"✅ Successfully repaired {config_file.name}")
            return True

        except Exception as e:
            print(f"❌ Failed to repair {config_file.name}: {e}")
            # Restore backup if repair failed
            if backup_file.exists():
                backup_file.rename(config_file)
            return False

    @staticmethod
    def validate_checkpoint_standalone(
        checkpoint_path: Path,
    ) -> DiagnosticResult:
        """
        Validate a standalone checkpoint file with detailed output.

        Args:
            checkpoint_path: Path to checkpoint file to validate

        Returns:
            Dictionary containing validation results
        """
        return CheckpointValidator._validate_checkpoint_file(checkpoint_path)

    @staticmethod
    def fix_checkpoint_permissions(checkpoint_path: Path) -> bool:
        """
        Fix common checkpoint file permission issues.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            True if permissions were fixed successfully
        """
        print(f"🔧 Checking permissions for: {checkpoint_path}")

        try:
            # Check if file is readable
            if not checkpoint_path.is_file():
                print(f"❌ File does not exist: {checkpoint_path}")
                return False

            # Try to read the file
            with open(checkpoint_path, "rb") as f:
                f.read(1024)  # Read first 1KB to test access

            print(f"✅ Permissions appear correct for {checkpoint_path.name}")
            return True

        except PermissionError:
            print(f"❌ Permission denied accessing {checkpoint_path}")
            try:
                # On Unix systems, try to fix permissions
                checkpoint_path.chmod(0o644)
                print(f"✅ Fixed permissions for {checkpoint_path.name}")
                return True
            except Exception as e:
                print(f"❌ Failed to fix permissions: {e}")
                return False

        except Exception as e:
            print(f"❌ Error accessing file: {e}")
            return False

    @staticmethod
    def cleanup_temporary_files(experiment_dir: Path) -> int:
        """
        Clean up temporary and cache files in experiment directory.

        Args:
            experiment_dir: Path to experiment directory

        Returns:
            Number of files cleaned up
        """
        print(f"🧹 Cleaning temporary files in: {experiment_dir}")

        # Patterns for temporary files to clean
        temp_patterns = [
            "*.tmp",
            "*.temp",
            "*~",
            ".*.swp",
            "__pycache__/*",
            "*.pyc",
            ".DS_Store",
            "Thumbs.db",
        ]

        cleanup_count = 0

        for pattern in temp_patterns:
            temp_files = list(experiment_dir.glob(f"**/{pattern}"))
            for temp_file in temp_files:
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                        cleanup_count += 1
                        print(f"🗑️  Removed: {temp_file.name}")
                    elif temp_file.is_dir() and pattern == "__pycache__/*":
                        # Remove __pycache__ directories
                        shutil.rmtree(temp_file)
                        cleanup_count += 1
                        print(f"🗑️  Removed directory: {temp_file}")
                except Exception as e:
                    print(f"⚠️ Could not remove {temp_file}: {e}")

        print(f"✅ Cleaned up {cleanup_count} temporary files")
        return cleanup_count
