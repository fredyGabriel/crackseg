#!/usr/bin/env python3
"""
Automated Experiment Comparison Script

This script provides automated comparison of multiple experiments using the
AutomatedComparisonEngine. It generates comprehensive analysis reports,
identifies best performers, and provides actionable recommendations.

Usage:
    python scripts/experiments/automated_comparison.py --experiments \
        exp1,exp2,exp3
    python scripts/experiments/automated_comparison.py --experiment-dirs \
        path1,path2,path3
    python scripts/experiments/automated_comparison.py --auto-find \
        --max-experiments 5
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from crackseg.reporting.comparison import AutomatedComparisonEngine  # noqa
from crackseg.reporting.config import OutputFormat, ReportConfig  # noqa
from crackseg.reporting.core import ExperimentReporter  # noqa


class AutomatedExperimentComparer:
    """Automated experiment comparison and analysis tool."""

    def __init__(self, project_root: Path) -> None:
        """Initialize the automated comparer."""
        self.project_root = project_root
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = (
            project_root
            / "docs"
            / "reports"
            / "comparison"
            / f"automated_comparison_{timestamp}"
        )

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.comparison_engine = AutomatedComparisonEngine()
        self.reporter = ExperimentReporter()

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def find_experiments_by_names(
        self, experiment_names: list[str]
    ) -> list[Path]:
        """Find experiment directories by names."""
        experiments_dir = self.project_root / "artifacts" / "experiments"
        if not experiments_dir.exists():
            self.logger.warning(
                f"Experiments directory not found: {experiments_dir}"
            )
            return []

        experiment_dirs = []
        for exp_name in experiment_names:
            # Look for exact matches or partial matches
            found = False
            for exp_dir in experiments_dir.iterdir():
                if (
                    exp_dir.is_dir()
                    and exp_name.lower() in exp_dir.name.lower()
                ):
                    experiment_dirs.append(exp_dir)
                    found = True
                    break

            if not found:
                self.logger.warning(f"Experiment not found: {exp_name}")

        return experiment_dirs

    def find_experiments_by_dirs(
        self, experiment_dirs: list[str]
    ) -> list[Path]:
        """Convert directory paths to Path objects."""
        paths = []
        for dir_path in experiment_dirs:
            path = Path(dir_path)
            if path.exists() and path.is_dir():
                paths.append(path)
            else:
                self.logger.warning(f"Directory not found: {dir_path}")

        return paths

    def auto_find_experiments(self, max_experiments: int = 5) -> list[Path]:
        """Automatically find recent experiments."""
        experiments_dir = self.project_root / "artifacts" / "experiments"
        if not experiments_dir.exists():
            self.logger.warning(
                f"Experiments directory not found: {experiments_dir}"
            )
            return []

        # Get all experiment directories sorted by modification time
        experiment_dirs = []
        for exp_dir in experiments_dir.iterdir():
            if exp_dir.is_dir():
                experiment_dirs.append(exp_dir)

        # Sort by modification time (most recent first)
        experiment_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        return experiment_dirs[:max_experiments]

    def load_experiment_data(self, experiment_dirs: list[Path]) -> list[Any]:
        """Load experiment data using the reporter."""
        self.logger.info(
            f"Loading data for {len(experiment_dirs)} experiments"
        )

        experiments_data = []
        for exp_dir in experiment_dirs:
            try:
                if (
                    self.reporter
                    and hasattr(self.reporter, "data_loader")
                    and self.reporter.data_loader is not None
                ):
                    exp_data = self.reporter.data_loader.load_experiment_data(
                        exp_dir
                    )
                    experiments_data.append(exp_data)
                    self.logger.info(
                        f"Loaded experiment: {exp_data.experiment_id}"
                    )
                else:
                    self.logger.warning(
                        f"Reporter not properly initialized for {exp_dir}"
                    )
            except Exception as e:
                self.logger.error(f"Failed to load experiment {exp_dir}: {e}")

        return experiments_data

    def run_comparison_analysis(
        self, experiments_data: list[Any]
    ) -> dict[str, Any]:
        """Run comprehensive comparison analysis."""
        self.logger.info("Running automated comparison analysis")

        # Create report configuration
        config = ReportConfig(
            output_formats=[
                OutputFormat.MARKDOWN,
                OutputFormat.HTML,
                OutputFormat.JSON,
            ],
            include_performance_analysis=True,
            include_recommendations=True,
        )

        # Run comparison analysis
        comparison_results = self.comparison_engine.compare_experiments(
            experiments_data, config
        )

        # Identify best performer
        best_performer = self.comparison_engine.identify_best_performing(
            experiments_data, config
        )

        # Generate comparison table
        comparison_table = self.comparison_engine.generate_comparison_table(
            experiments_data, config
        )

        return {
            "comparison_results": comparison_results,
            "best_performer": best_performer,
            "comparison_table": comparison_table,
            "analysis_timestamp": datetime.now().isoformat(),
        }

    def generate_comparison_report(
        self, analysis_results: dict[str, Any]
    ) -> None:
        """Generate comprehensive comparison report."""
        self.logger.info("Generating comparison report")

        # Create report content
        report_content = {
            "title": "Automated Experiment Comparison Report",
            "timestamp": analysis_results["analysis_timestamp"],
            "comparison_results": analysis_results["comparison_results"],
            "best_performer": analysis_results["best_performer"],
            "comparison_table": analysis_results["comparison_table"],
        }

        # Save JSON report
        json_path = self.output_dir / "comparison_analysis.json"
        with open(json_path, "w") as f:
            json.dump(report_content, f, indent=2, default=str)

        # Save comparison table as CSV
        if "table_data" in analysis_results["comparison_table"]:
            table_data = analysis_results["comparison_table"]["table_data"]
            df = pd.DataFrame(table_data)
            csv_path = self.output_dir / "comparison_table.csv"
            df.to_csv(csv_path, index=False)

        # Generate markdown report
        self._generate_markdown_report(report_content)

        self.logger.info(f"Reports saved to: {self.output_dir}")

    def _generate_markdown_report(
        self, report_content: dict[str, Any]
    ) -> None:
        """Generate markdown report."""
        md_path = self.output_dir / "comparison_report.md"

        with open(md_path, "w") as f:
            f.write(f"# {report_content['title']}\n\n")
            f.write(f"**Generated:** {report_content['timestamp']}\n\n")

            # Best performer section
            if "best_performer" in report_content:
                best = report_content["best_performer"]
                f.write("## 🏆 Best Performing Experiment\n\n")
                exp_id = best.get("experiment_id", "N/A")
                f.write(f"- **Experiment ID:** {exp_id}\n")
                composite_score = best.get("composite_score", 0)
                f.write(f"- **Composite Score:** {composite_score:.4f}\n")
                confidence_level = best.get("confidence_level", 0)
                f.write(f"- **Confidence Level:** {confidence_level:.2%}\n")
                significance = best.get("statistical_significance", False)
                f.write(f"- **Statistical Significance:** {significance}\n\n")

                if "runner_up" in best:
                    runner_up = best["runner_up"]
                    runner_id = runner_up.get("experiment_id", "N/A")
                    f.write(f"- **Runner-up:** {runner_id}\n")
                    score_diff = runner_up.get("score_difference", 0)
                    f.write(f"- **Score Difference:** {score_diff:.4f}\n")
                    improvement = runner_up.get("percentage_improvement", 0)
                    f.write(f"- **Improvement:** {improvement:.2f}%\n\n")

            # Statistical analysis section
            if "comparison_results" in report_content:
                comp_results = report_content["comparison_results"]

                f.write("## 📊 Statistical Analysis\n\n")
                exp_count = comp_results.get("experiment_count", 0)
                f.write(f"- **Experiments Compared:** {exp_count}\n")
                timestamp = comp_results.get("comparison_timestamp", "N/A")
                f.write(f"- **Analysis Timestamp:** {timestamp}\n\n")

                # Ranking analysis
                if "ranking_analysis" in comp_results:
                    ranking = comp_results["ranking_analysis"]
                    f.write("### Ranking Analysis\n\n")
                    f.write("| Position | Experiment ID | Total Score |\n")
                    f.write("|----------|---------------|-------------|\n")

                    for rank in ranking.get("ranking", []):
                        position = rank.get("position", "N/A")
                        exp_id = rank.get("experiment_id", "N/A")
                        total_score = rank.get("total_score", 0)
                        f.write(
                            f"| {position} | {exp_id} | {total_score:.4f} |\n"
                        )
                    f.write("\n")

                # Recommendations
                if "recommendations" in comp_results:
                    f.write("### 💡 Recommendations\n\n")
                    for rec in comp_results["recommendations"]:
                        f.write(f"- {rec}\n")
                    f.write("\n")

            # Anomaly detection
            if "anomaly_detection" in comp_results:
                anomalies = comp_results["anomaly_detection"]
                if anomalies.get("outliers"):
                    f.write("## ⚠️ Anomaly Detection\n\n")
                    for metric, outliers in anomalies["outliers"].items():
                        f.write(f"### {metric.upper()} Outliers\n\n")
                        for exp_id, value in outliers:
                            f.write(f"- {exp_id}: {value:.4f}\n")
                        f.write("\n")

    def print_summary(self, analysis_results: dict[str, Any]) -> None:
        """Print analysis summary to console."""
        print("\n" + "=" * 80)
        print("🔍 AUTOMATED EXPERIMENT COMPARISON SUMMARY")
        print("=" * 80)

        # Best performer
        if "best_performer" in analysis_results:
            best = analysis_results["best_performer"]
            print("\n🏆 BEST PERFORMER:")
            print(f"   Experiment ID: {best.get('experiment_id', 'N/A')}")
            print(f"   Composite Score: {best.get('composite_score', 0):.4f}")
            print(
                f"   Confidence Level: {best.get('confidence_level', 0):.2%}"
            )
            print(
                f"   Statistical Significance: "
                f"{best.get('statistical_significance', False)}"
            )

        # Ranking summary
        if "comparison_results" in analysis_results:
            comp_results = analysis_results["comparison_results"]
            print("\n📊 ANALYSIS SUMMARY:")
            print(
                f"   Experiments Compared: "
                f"{comp_results.get('experiment_count', 0)}"
            )
            print(
                f"   Analysis Timestamp: "
                f"{comp_results.get('comparison_timestamp', 'N/A')}"
            )

            # Top 3 performers
            if "ranking_analysis" in comp_results:
                ranking = comp_results["ranking_analysis"]["ranking"]
                print("\n🥇 TOP PERFORMERS:")
                for i, rank in enumerate(ranking[:3]):
                    print(
                        f"   {i + 1}. {rank.get('experiment_id', 'N/A')} "
                        f"(Score: {rank.get('total_score', 0):.4f})"
                    )

            # Recommendations
            if "recommendations" in comp_results:
                print("\n💡 KEY RECOMMENDATIONS:")
                for rec in comp_results["recommendations"][:3]:
                    print(f"   • {rec}")

        print(f"\n📁 Reports saved to: {self.output_dir}")
        print("=" * 80)

    def run_complete_analysis(
        self,
        experiment_names: list[str] | None = None,
        experiment_dirs: list[str] | None = None,
        auto_find: bool = False,
        max_experiments: int = 5,
    ) -> None:
        """Run complete automated comparison analysis."""
        print("🚀 Automated Experiment Comparison")
        print("=" * 50)

        # Find experiment directories
        if experiment_names:
            experiment_paths = self.find_experiments_by_names(experiment_names)
        elif experiment_dirs:
            experiment_paths = self.find_experiments_by_dirs(experiment_dirs)
        elif auto_find:
            experiment_paths = self.auto_find_experiments(max_experiments)
        else:
            print("❌ No experiment selection method specified!")
            return

        if not experiment_paths:
            print("❌ No experiments found for comparison!")
            return

        print(f"📋 Found {len(experiment_paths)} experiments for comparison:")
        for exp_path in experiment_paths:
            print(f"   • {exp_path.name}")

        # Load experiment data
        experiments_data = self.load_experiment_data(experiment_paths)

        if not experiments_data:
            print("❌ No experiment data could be loaded!")
            return

        # Run comparison analysis
        analysis_results = self.run_comparison_analysis(experiments_data)

        # Generate reports
        self.generate_comparison_report(analysis_results)

        # Print summary
        self.print_summary(analysis_results)


def main() -> None:
    """Main function for automated experiment comparison."""
    parser = argparse.ArgumentParser(
        description="Automated experiment comparison tool"
    )

    # Experiment selection options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--experiments",
        type=str,
        help="Comma-separated list of experiment names",
    )
    group.add_argument(
        "--experiment-dirs",
        type=str,
        help="Comma-separated list of experiment directory paths",
    )
    group.add_argument(
        "--auto-find",
        action="store_true",
        help="Automatically find recent experiments",
    )

    # Additional options
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=5,
        help="Maximum number of experiments to analyze (default: 5)",
    )

    args = parser.parse_args()

    # Initialize comparer
    project_root = Path(__file__).parent.parent.parent
    comparer = AutomatedExperimentComparer(project_root)

    # Parse experiment selection
    experiment_names = None
    experiment_dirs = None

    if args.experiments:
        experiment_names = [
            name.strip() for name in args.experiments.split(",")
        ]
    elif args.experiment_dirs:
        experiment_dirs = [
            path.strip() for path in args.experiment_dirs.split(",")
        ]

    # Run analysis
    comparer.run_complete_analysis(
        experiment_names=experiment_names,
        experiment_dirs=experiment_dirs,
        auto_find=args.auto_find,
        max_experiments=args.max_experiments,
    )


if __name__ == "__main__":
    main()
