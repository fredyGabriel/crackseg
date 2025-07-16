#!/usr/bin/env python3
"""Debug and diagnostic utilities for training artifacts.

This script provides utilities to diagnose and resolve common problems with
training artifacts including checkpoints, configurations, and metrics.

Usage:
    python scripts/debug_artifacts.py --help
    python scripts/debug_artifacts.py diagnose \\
        --experiment-dir outputs/experiments/exp_001
    python scripts/debug_artifacts.py validate \\
        --checkpoint checkpoints/model_best.pth.tar
    python scripts/debug_artifacts.py fix-config \\
        --config-dir configurations/exp_001
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crackseg.utils.config.standardized_storage import (
    StandardizedConfigStorage,
    validate_configuration_completeness,
)


class ArtifactDiagnostics:
    """Diagnostic tools for training artifacts."""

    def __init__(self, experiment_dir: Path) -> None:
        """Initialize diagnostics for a specific experiment directory."""
        self.experiment_dir = experiment_dir
        self.issues: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def diagnose_all(self) -> dict[str, Any]:
        """Run comprehensive diagnostics on all artifacts."""
        print(f"🔍 Diagnosing artifacts in: {self.experiment_dir}")
        print("=" * 60)

        results = {
            "experiment_dir": str(self.experiment_dir),
            "checkpoints": self._diagnose_checkpoints(),
            "configurations": self._diagnose_configurations(),
            "metrics": self._diagnose_metrics(),
            "overall_health": "unknown",
        }

        # Determine overall health
        total_issues = 0
        for category in ["checkpoints", "configurations", "metrics"]:
            category_result = results[category]
            if (
                isinstance(category_result, dict)
                and "issues" in category_result
            ):
                total_issues += len(category_result["issues"])

        if total_issues == 0:
            results["overall_health"] = "healthy"
            print("\n✅ All artifacts appear healthy!")
        elif total_issues <= 3:
            results["overall_health"] = "warning"
            print(f"\n⚠️ Found {total_issues} minor issues")
        else:
            results["overall_health"] = "critical"
            print(f"\n❌ Found {total_issues} issues requiring attention")

        return results

    def _diagnose_checkpoints(self) -> dict[str, Any]:
        """Diagnose checkpoint artifacts."""
        print("\n📦 Diagnosing Checkpoints")
        print("-" * 30)

        checkpoint_dir = self.experiment_dir / "checkpoints"
        if not checkpoint_dir.exists():
            issue = "Checkpoint directory not found"
            print(f"❌ {issue}")
            return {"status": "missing", "issues": [issue], "files": []}

        checkpoint_files = list(checkpoint_dir.glob("*.pth*"))
        if not checkpoint_files:
            issue = "No checkpoint files found"
            print(f"❌ {issue}")
            return {"status": "empty", "issues": [issue], "files": []}

        issues = []
        file_info = []

        for checkpoint_file in checkpoint_files:
            file_result = self._validate_checkpoint_file(checkpoint_file)
            file_info.append(file_result)
            issues.extend(file_result.get("issues", []))

            status_icon = "✅" if not file_result.get("issues") else "⚠️"
            print(
                f"{status_icon} {checkpoint_file.name}: "
                f"{file_result['status']}"
            )

        return {
            "status": "healthy" if not issues else "issues",
            "issues": issues,
            "files": file_info,
            "file_count": len(checkpoint_files),
        }

    def _validate_checkpoint_file(
        self, checkpoint_path: Path
    ) -> dict[str, Any]:
        """Validate a single checkpoint file."""
        issues = []

        try:
            # Try to load checkpoint
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

            # Check required fields
            required_fields = [
                "model_state_dict",
                "optimizer_state_dict",
                "epoch",
                "best_metric_value",
            ]

            for field in required_fields:
                if field not in checkpoint_data:
                    issues.append(f"Missing required field: {field}")

            # Validate data types
            if "epoch" in checkpoint_data and not isinstance(
                checkpoint_data["epoch"], int
            ):
                issues.append("Invalid epoch type (should be int)")

            if "best_metric_value" in checkpoint_data and not isinstance(
                checkpoint_data["best_metric_value"], int | float
            ):
                issues.append(
                    "Invalid best_metric_value type (should be numeric)"
                )

            # Check file size (shouldn't be too small)
            file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            if file_size_mb < 0.1:
                issues.append(
                    f"Suspiciously small file size: {file_size_mb:.2f}MB"
                )

            return {
                "file": str(checkpoint_path),
                "status": "valid" if not issues else "invalid",
                "issues": issues,
                "size_mb": round(file_size_mb, 2),
                "epoch": checkpoint_data.get("epoch", "unknown"),
            }

        except Exception as e:
            issues.append(f"Failed to load checkpoint: {e}")
            return {
                "file": str(checkpoint_path),
                "status": "corrupted",
                "issues": issues,
            }

    def _diagnose_configurations(self) -> dict[str, Any]:
        """Diagnose configuration artifacts."""
        print("\n⚙️ Diagnosing Configurations")
        print("-" * 30)

        config_dir = self.experiment_dir / "configurations"
        if not config_dir.exists():
            issue = "Configuration directory not found"
            print(f"❌ {issue}")
            return {"status": "missing", "issues": [issue], "experiments": []}

        try:
            config_storage = StandardizedConfigStorage(base_dir=config_dir)
            experiment_ids = config_storage.list_experiments()

            if not experiment_ids:
                issue = "No configuration experiments found"
                print(f"❌ {issue}")
                return {
                    "status": "empty",
                    "issues": [issue],
                    "experiments": [],
                }

            issues = []
            experiment_info = []

            for exp_id in experiment_ids:
                exp_result = self._validate_configuration_experiment(
                    config_storage, exp_id
                )
                experiment_info.append(exp_result)
                issues.extend(exp_result.get("issues", []))

                status_icon = "✅" if not exp_result.get("issues") else "⚠️"
                print(f"{status_icon} {exp_id}: {exp_result['status']}")

            return {
                "status": "healthy" if not issues else "issues",
                "issues": issues,
                "experiments": experiment_info,
                "experiment_count": len(experiment_ids),
            }

        except Exception as e:
            issue = f"Failed to access configuration storage: {e}"
            print(f"❌ {issue}")
            return {"status": "error", "issues": [issue], "experiments": []}

    def _validate_configuration_experiment(
        self, config_storage: StandardizedConfigStorage, experiment_id: str
    ) -> dict[str, Any]:
        """Validate a single configuration experiment."""
        issues = []

        try:
            config = config_storage.load_configuration(experiment_id)

            # Validate configuration completeness
            validation_result = validate_configuration_completeness(config)

            if validation_result.get("has_critical_missing", False):
                issues.extend(
                    f"Critical missing: {field}"
                    for field in validation_result.get("critical_missing", [])
                )

            if validation_result.get("has_recommended_missing", False):
                # Don't treat recommended as issues, just warnings
                pass

            # Check for required sections
            required_sections = ["experiment", "model", "training"]
            for section in required_sections:
                if section not in config:
                    issues.append(f"Missing required section: {section}")

            # Check environment metadata
            if "environment" not in config:
                issues.append("Missing environment metadata")

            return {
                "experiment_id": experiment_id,
                "status": "valid" if not issues else "invalid",
                "issues": issues,
                "validation_score": validation_result.get(
                    "completeness_score", 0
                ),
            }

        except Exception as e:
            issues.append(f"Failed to load configuration: {e}")
            return {
                "experiment_id": experiment_id,
                "status": "corrupted",
                "issues": issues,
            }

    def _diagnose_metrics(self) -> dict[str, Any]:
        """Diagnose metrics artifacts."""
        print("\n📊 Diagnosing Metrics")
        print("-" * 30)

        # Look for metrics directories
        metrics_dirs = list(self.experiment_dir.glob("**/metrics"))

        if not metrics_dirs:
            issue = "No metrics directories found"
            print(f"⚠️ {issue}")
            return {"status": "missing", "issues": [issue], "files": []}

        issues = []
        file_info = []

        for metrics_dir in metrics_dirs:
            # Look for metrics files
            metrics_files = (
                list(metrics_dir.glob("*.csv"))
                + list(metrics_dir.glob("*.json"))
                + list(metrics_dir.glob("metrics_summary.json"))
            )

            for metrics_file in metrics_files:
                file_result = self._validate_metrics_file(metrics_file)
                file_info.append(file_result)
                issues.extend(file_result.get("issues", []))

                status_icon = "✅" if not file_result.get("issues") else "⚠️"
                print(
                    f"{status_icon} {metrics_file.name}: "
                    f"{file_result['status']}"
                )

        if not file_info:
            issue = "No metrics files found"
            print(f"⚠️ {issue}")
            return {"status": "empty", "issues": [issue], "files": []}

        return {
            "status": "healthy" if not issues else "issues",
            "issues": issues,
            "files": file_info,
            "file_count": len(file_info),
        }

    def _validate_metrics_file(self, metrics_path: Path) -> dict[str, Any]:
        """Validate a single metrics file."""
        issues = []

        try:
            if metrics_path.suffix == ".json":
                with open(metrics_path, encoding="utf-8") as f:
                    data = json.load(f)

                # Validate metrics summary structure
                if metrics_path.name == "metrics_summary.json":
                    required_fields = ["experiment_info", "training_summary"]
                    for field in required_fields:
                        if field not in data:
                            issues.append(f"Missing required field: {field}")

            elif metrics_path.suffix == ".csv":
                # Basic CSV validation (file should not be empty)
                file_size = metrics_path.stat().st_size
                if file_size == 0:
                    issues.append("CSV file is empty")
                elif file_size < 50:  # Probably just headers
                    issues.append("CSV file appears to contain only headers")

            return {
                "file": str(metrics_path),
                "status": "valid" if not issues else "invalid",
                "issues": issues,
                "size_bytes": metrics_path.stat().st_size,
            }

        except Exception as e:
            issues.append(f"Failed to validate metrics file: {e}")
            return {
                "file": str(metrics_path),
                "status": "corrupted",
                "issues": issues,
            }


class ArtifactFixer:
    """Tools to fix common artifact issues."""

    @staticmethod
    def fix_configuration_directory(config_dir: Path) -> bool:
        """Attempt to fix common configuration directory issues."""
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

        return True

    @staticmethod
    def validate_checkpoint_standalone(
        checkpoint_path: Path,
    ) -> dict[str, Any]:
        """Validate a standalone checkpoint file."""
        print(f"🔍 Validating checkpoint: {checkpoint_path}")

        if not checkpoint_path.exists():
            return {"status": "missing", "issues": ["File does not exist"]}

        try:
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
            issues = []

            # Check required fields
            required_fields = [
                "model_state_dict",
                "optimizer_state_dict",
                "epoch",
                "best_metric_value",
            ]

            for field in required_fields:
                if field not in checkpoint_data:
                    issues.append(f"Missing required field: {field}")

            # Check model state dict
            if "model_state_dict" in checkpoint_data:
                model_state = checkpoint_data["model_state_dict"]
                if not isinstance(model_state, dict):
                    issues.append("model_state_dict is not a dictionary")
                elif len(model_state) == 0:
                    issues.append("model_state_dict is empty")

            print("Checkpoint analysis:")
            print(f"  Epoch: {checkpoint_data.get('epoch', 'N/A')}")
            print(
                f"  Best metric: "
                f"{checkpoint_data.get('best_metric_value', 'N/A')}"
            )
            print(
                f"  Model parameters: "
                f"{len(checkpoint_data.get('model_state_dict', {}))}"
            )

            if issues:
                print("Issues found:")
                for issue in issues:
                    print(f"  ❌ {issue}")
            else:
                print("✅ Checkpoint appears valid")

            return {
                "status": "valid" if not issues else "invalid",
                "issues": issues,
                "epoch": checkpoint_data.get("epoch"),
                "best_metric": checkpoint_data.get("best_metric_value"),
            }

        except Exception as e:
            error_msg = f"Failed to load checkpoint: {e}"
            print(f"❌ {error_msg}")
            return {"status": "corrupted", "issues": [error_msg]}


def main() -> None:
    """Main CLI interface for artifact debugging tools."""
    parser = argparse.ArgumentParser(
        description="Debug and validate training artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s diagnose \\
    --experiment-dir outputs/experiments/20241201-143045-crackseg
  %(prog)s validate --checkpoint outputs/checkpoints/model_best.pth.tar
  %(prog)s fix-config --config-dir outputs/experiments/exp_001/configurations
        """,
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    # Diagnose command
    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="Run comprehensive diagnostics on experiment artifacts",
    )
    diagnose_parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to experiment directory to diagnose",
    )
    diagnose_parser.add_argument(
        "--output",
        type=Path,
        help="Save diagnostic results to JSON file",
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate a specific checkpoint file"
    )
    validate_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint file to validate",
    )

    # Fix config command
    fix_parser = subparsers.add_parser(
        "fix-config", help="Attempt to fix configuration directory issues"
    )
    fix_parser.add_argument(
        "--config-dir",
        type=Path,
        required=True,
        help="Path to configuration directory to fix",
    )

    args = parser.parse_args()

    if args.command == "diagnose":
        diagnostics = ArtifactDiagnostics(args.experiment_dir)
        results = diagnostics.diagnose_all()

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"\n📄 Results saved to: {args.output}")

    elif args.command == "validate":
        ArtifactFixer.validate_checkpoint_standalone(args.checkpoint)

    elif args.command == "fix-config":
        ArtifactFixer.fix_configuration_directory(args.config_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
