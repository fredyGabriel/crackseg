"""Checkpoint validation utilities."""

from pathlib import Path

import torch

from .utils import DiagnosticResult, IssueList


class CheckpointValidator:
    """Utilities for validating PyTorch checkpoint files."""

    @staticmethod
    def diagnose_checkpoints(experiment_dir: Path) -> DiagnosticResult:
        """Diagnose checkpoint artifacts in an experiment directory.

        Args:
            experiment_dir: Path to experiment directory

        Returns:
            Dictionary containing checkpoint diagnosis results
        """
        print("\n📦 Diagnosing Checkpoints")
        print("-" * 30)

        checkpoint_dir = experiment_dir / "checkpoints"
        if not checkpoint_dir.exists():
            issue = "Checkpoint directory not found"
            print(f"❌ {issue}")
            return {"status": "missing", "issues": [issue], "files": []}

        checkpoint_files = list(checkpoint_dir.glob("*.pth*"))
        if not checkpoint_files:
            issue = "No checkpoint files found"
            print(f"❌ {issue}")
            return {"status": "empty", "issues": [issue], "files": []}

        issues: IssueList = []
        file_info = []

        for checkpoint_file in checkpoint_files:
            file_result = CheckpointValidator._validate_checkpoint_file(
                checkpoint_file
            )
            file_info.append(file_result)

            file_issues = file_result.get("issues", [])
            if isinstance(file_issues, list):
                issues.extend(file_issues)

            status_icon = "✅" if not file_result.get("issues") else "❌"
            print(f"{status_icon} {checkpoint_file.name}")

        return {
            "status": "healthy" if not issues else "issues",
            "issues": issues,
            "files": file_info,
            "file_count": len(file_info),
        }

    @staticmethod
    def _validate_checkpoint_file(checkpoint_file: Path) -> DiagnosticResult:
        """Validate a single checkpoint file.

        Args:
            checkpoint_file: Path to checkpoint file

        Returns:
            Dictionary containing validation results
        """
        issues: IssueList = []

        try:
            # Try to load checkpoint
            checkpoint = torch.load(checkpoint_file, map_location="cpu")

            # Check required keys
            required_keys = ["model_state_dict", "epoch"]
            for key in required_keys:
                if key not in checkpoint:
                    issues.append(f"Missing required key: {key}")

            # Check model state dict
            if "model_state_dict" in checkpoint:
                model_state = checkpoint["model_state_dict"]
                if not isinstance(model_state, dict):
                    issues.append("model_state_dict is not a dictionary")
                elif len(model_state) == 0:
                    issues.append("model_state_dict is empty")

            return {
                "file": str(checkpoint_file),
                "status": "valid" if not issues else "invalid",
                "issues": issues,
                "size_mb": checkpoint_file.stat().st_size / (1024 * 1024),
            }

        except Exception as e:
            issues.append(f"Failed to load checkpoint: {e}")
            return {
                "file": str(checkpoint_file),
                "status": "corrupted",
                "issues": issues,
            }
