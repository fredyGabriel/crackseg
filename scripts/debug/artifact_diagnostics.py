"""Artifact diagnostics for training experiments."""

from pathlib import Path

from crackseg.utils.config.standardized_storage import (
    StandardizedConfigStorage,
    validate_configuration_completeness,
)

from .checkpoint_validator import CheckpointValidator
from .utils import DiagnosticResult, IssueList, validate_metrics_file


class ArtifactDiagnostics:
    """Diagnostic tools for training artifacts."""

    def __init__(self, experiment_dir: Path) -> None:
        """
        Initialize diagnostics for a specific experiment directory. Args:
        experiment_dir: Path to experiment directory to diagnose
        """
        self.experiment_dir = experiment_dir
        self.issues: IssueList = []
        self.warnings: IssueList = []
        self.info: IssueList = []

    def diagnose_all(self) -> DiagnosticResult:
        """
        Run comprehensive diagnostics on all artifacts. Returns: Dictionary
        containing complete diagnostic results
        """
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

    def _diagnose_checkpoints(self) -> DiagnosticResult:
        """
        Diagnose checkpoint artifacts. Returns: Dictionary containing
        checkpoint diagnosis results
        """
        return CheckpointValidator.diagnose_checkpoints(self.experiment_dir)

    def _diagnose_configurations(self) -> DiagnosticResult:
        """
        Diagnose configuration artifacts. Returns: Dictionary containing
        configuration diagnosis results
        """
        print("\n⚙️ Diagnosing Configurations")
        print("-" * 30)

        config_dir = self.experiment_dir / "configurations"
        if not config_dir.exists():
            issue = "Configuration directory not found"
            print(f"❌ {issue}")
            return {"status": "missing", "issues": [issue], "experiments": []}

        try:
            config_storage = StandardizedConfigStorage(config_dir)
            experiment_ids = config_storage.list_experiments()

            if not experiment_ids:
                issue = "No configuration experiments found"
                print(f"❌ {issue}")
                return {
                    "status": "empty",
                    "issues": [issue],
                    "experiments": [],
                }

            issues: IssueList = []
            experiment_info = []

            for exp_id in experiment_ids:
                exp_result = self._validate_configuration_experiment(
                    config_storage, exp_id
                )
                experiment_info.append(exp_result)  # type: ignore[misc]
                exp_issues = exp_result.get("issues", [])
                if isinstance(exp_issues, list):
                    issues.extend(exp_issues)  # type: ignore[misc]

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
        self,
        config_storage: StandardizedConfigStorage,
        experiment_id: str,
    ) -> DiagnosticResult:
        """
        Validate a single configuration experiment. Args: config_storage:
        Configuration storage instance experiment_id: Experiment ID to
        validate Returns: Dictionary containing experiment validation results
        """
        issues: IssueList = []

        try:
            config = config_storage.load_configuration(experiment_id)

            # Validate configuration completeness
            validation_result = validate_configuration_completeness(config)

            if validation_result.get("has_critical_missing", False):
                critical_missing = validation_result.get(
                    "critical_missing", []
                )
                for field in critical_missing:
                    issues.append(f"Critical missing: {field}")

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

    def _diagnose_metrics(self) -> DiagnosticResult:
        """
        Diagnose metrics artifacts. Returns: Dictionary containing metrics
        diagnosis results
        """
        print("\n📊 Diagnosing Metrics")
        print("-" * 30)

        # Look for metrics directories
        metrics_dirs = list(self.experiment_dir.glob("**/metrics"))

        if not metrics_dirs:
            issue = "No metrics directories found"
            print(f"⚠️ {issue}")
            return {"status": "missing", "issues": [issue], "files": []}

        issues: IssueList = []
        file_info = []

        for metrics_dir in metrics_dirs:
            metrics_files = [
                f
                for f in metrics_dir.iterdir()
                if f.suffix in [".json", ".jsonl", ".csv"]
            ]

            for metrics_file in metrics_files:
                file_result = validate_metrics_file(metrics_file)
                file_info.append(file_result)  # type: ignore[misc]

                file_issues = file_result.get("issues", [])
                if isinstance(file_issues, list):
                    issues.extend(file_issues)  # type: ignore[misc]

                status_icon = "✅" if not file_result.get("issues") else "❌"
                print(f"{status_icon} {metrics_file.name}")

        return {
            "status": "healthy" if not issues else "issues",
            "issues": issues,
            "files": file_info,
            "file_count": len(file_info),
        }
