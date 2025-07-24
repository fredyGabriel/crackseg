#!/usr/bin/env python3
"""
SwinV2 Hybrid Experiment Analysis Wrapper

This script provides a comprehensive analysis workflow for the SwinV2 Hybrid
experiment, integrating with the existing experiment_visualizer.py and other
analysis tools.

Usage:
    python scripts/experiments/swinv2_hybrid/swinv2_hybrid_analysis.py
    python scripts/experiments/swinv2_hybrid/swinv2_hybrid_analysis.py \
        --compare-baseline
    python scripts/experiments/swinv2_hybrid/swinv2_hybrid_analysis.py \
        --export-report
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


class SwinV2HybridAnalyzer:
    """Comprehensive analyzer for SwinV2 Hybrid experiment results."""

    def __init__(self, project_root: Path) -> None:
        """Initialize the analyzer."""
        self.project_root = project_root
        self.experiment_name = "swinv2_hybrid"
        self.output_dir = (
            project_root
            / "docs"
            / "reports"
            / "analysis"
            / "swinv2_hybrid_analysis"
        )

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_experiment_results(self) -> list[Path]:
        """Find all SwinV2 Hybrid experiment results."""
        experiments_dir = self.project_root / "outputs" / "experiments"
        if not experiments_dir.exists():
            return []

        # Look for experiments with swinv2_hybrid in the name
        experiment_dirs = []
        for exp_dir in experiments_dir.iterdir():
            if exp_dir.is_dir() and "swinv2_hybrid" in exp_dir.name:
                experiment_dirs.append(exp_dir)

        return sorted(experiment_dirs, key=lambda x: x.name, reverse=True)

    def run_visualization_analysis(self, experiment_dirs: list[Path]) -> None:
        """
        Run the generic experiment visualizer on SwinV2 Hybrid experiments.
        """
        if not experiment_dirs:
            print("❌ No SwinV2 Hybrid experiments found!")
            return

        print(f"🔍 Found {len(experiment_dirs)} SwinV2 Hybrid experiments:")
        for exp_dir in experiment_dirs:
            print(f"   📂 {exp_dir.name}")

        # Build command for generic visualizer
        experiment_paths = ",".join(
            str(exp_dir) for exp_dir in experiment_dirs
        )

        cmd = [
            sys.executable,
            "scripts/experiments/experiment_visualizer.py",
            "--experiment-dirs",
            experiment_paths,
            "--output-dir",
            str(self.output_dir),
            "--title",
            "SwinV2 Hybrid Architecture Analysis",
        ]

        print("\n🚀 Running visualization analysis...")
        print(f"   Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True
            )
            print("✅ Visualization analysis completed successfully!")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Visualization analysis failed: {e}")
            print(f"Error output: {e.stderr}")

    def generate_performance_summary(
        self, experiment_dirs: list[Path]
    ) -> None:
        """Generate a performance summary report."""
        if not experiment_dirs:
            return

        print("\n📊 Generating performance summary...")

        summary_data = []
        for exp_dir in experiment_dirs:
            summary_file = exp_dir / "metrics" / "complete_summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    data = json.load(f)

                # Extract key metrics
                best_metrics = data.get("best_metrics", {})
                experiment_info = data.get("experiment_info", {})

                summary_data.append(
                    {
                        "experiment": exp_dir.name,
                        "final_loss": best_metrics.get("loss", {}).get(
                            "value", "N/A"
                        ),
                        "final_iou": best_metrics.get("iou", {}).get(
                            "value", "N/A"
                        ),
                        "final_f1": best_metrics.get("f1", {}).get(
                            "value", "N/A"
                        ),
                        "final_precision": best_metrics.get(
                            "precision", {}
                        ).get("value", "N/A"),
                        "final_recall": best_metrics.get("recall", {}).get(
                            "value", "N/A"
                        ),
                        "total_epochs": experiment_info.get(
                            "total_epochs", "N/A"
                        ),
                        "best_epoch": experiment_info.get("best_epoch", "N/A"),
                        "training_time": experiment_info.get(
                            "training_time", "N/A"
                        ),
                    }
                )

        if summary_data:
            # Create DataFrame and save
            df = pd.DataFrame(summary_data)
            summary_path = (
                self.output_dir / "swinv2_hybrid_performance_summary.csv"
            )
            df.to_csv(summary_path, index=False)

            print(f"✅ Performance summary saved to: {summary_path}")

            # Print summary table
            print("\n📋 Performance Summary:")
            print("=" * 80)
            print(df.to_string(index=False))

    def compare_with_baseline(self) -> None:
        """Compare SwinV2 Hybrid results with baseline experiments."""
        print("\n🔍 Comparing with baseline experiments...")

        # Find baseline experiments (tutorial_02, etc.)
        experiments_dir = self.project_root / "outputs" / "experiments"
        baseline_dirs = []

        for exp_dir in experiments_dir.iterdir():
            if exp_dir.is_dir() and any(
                baseline in exp_dir.name
                for baseline in ["tutorial_02", "baseline", "default"]
            ):
                baseline_dirs.append(exp_dir)

        if baseline_dirs:
            print(
                f"Found {len(baseline_dirs)} baseline experiments for "
                f"comparison"
            )

            # Run comparison analysis
            all_experiments = self.find_experiment_results() + baseline_dirs
            experiment_paths = ",".join(
                str(exp_dir) for exp_dir in all_experiments
            )

            cmd = [
                sys.executable,
                "scripts/experiments/experiment_visualizer.py",
                "--experiment-dirs",
                experiment_paths,
                "--output-dir",
                str(self.output_dir / "comparison"),
                "--title",
                "SwinV2 Hybrid vs Baseline Comparison",
            ]

            try:
                subprocess.run(cmd, check=True)
                print("✅ Baseline comparison completed!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Baseline comparison failed: {e}")
        else:
            print("⚠️ No baseline experiments found for comparison")

    def export_comprehensive_report(self) -> None:
        """Export a comprehensive analysis report."""
        print("\n📄 Generating comprehensive report...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = (
            self.output_dir / f"swinv2_hybrid_analysis_report_{timestamp}.md"
        )

        experiment_dirs = self.find_experiment_results()

        with open(report_path, "w") as f:
            f.write("# SwinV2 Hybrid Experiment Analysis Report\n\n")
            f.write(
                f"**Generated:** "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            f.write("## Experiment Overview\n\n")
            f.write("- **Architecture:** SwinV2 + ASPP + CNN Hybrid\n")
            f.write("- **Loss Function:** Focal Dice Loss\n")
            f.write(
                f"- **Total Experiments Found:** {len(experiment_dirs)}\n\n"
            )

            f.write("## Experiment Results\n\n")
            for exp_dir in experiment_dirs:
                f.write(f"### {exp_dir.name}\n")
                f.write(f"- **Path:** `{exp_dir}`\n")

                # Try to load metrics
                summary_file = exp_dir / "metrics" / "complete_summary.json"
                if summary_file.exists():
                    with open(summary_file) as sf:
                        data = json.load(sf)
                        best_metrics = data.get("best_metrics", {})
                        f.write(
                            f"- **Best IoU:** "
                            f"{best_metrics.get('iou', {}).get('value', 'N/A')}\n"  # noqa: E501
                        )
                        f.write(
                            f"- **Best F1:** "
                            f"{best_metrics.get('f1', {}).get('value', 'N/A')}\n"  # noqa: E501
                        )
                        f.write(
                            f"- **Best Loss:** "
                            f"{best_metrics.get('loss', {}).get('value', 'N/A')}\n"  # noqa: E501
                        )

                f.write("\n")

            f.write("## Analysis Files\n\n")
            f.write("The following analysis files have been generated:\n\n")
            f.write(
                "- `training_curves_*.png` - Training progress visualization\n"
            )
            f.write(
                "- `performance_radar_*.png` - Performance comparison "
                "radar chart\n"
            )
            f.write(
                "- `experiment_comparison_*.csv` - Tabular comparison data\n"
            )
            f.write(
                "- `swinv2_hybrid_performance_summary.csv` - Performance "
                "summary\n\n"
            )

            f.write("## Next Steps\n\n")
            f.write("1. Review the generated visualizations\n")
            f.write("2. Analyze performance trends across experiments\n")
            f.write("3. Compare with baseline models\n")
            f.write("4. Document insights and recommendations\n")

        print(f"✅ Comprehensive report saved to: {report_path}")

    def run_complete_analysis(
        self, compare_baseline: bool = False, export_report: bool = False
    ) -> None:
        """Run complete analysis workflow."""
        print("🚀 SwinV2 Hybrid Experiment Analysis")
        print("=" * 50)

        # Find experiment results
        experiment_dirs = self.find_experiment_results()

        if not experiment_dirs:
            print("❌ No SwinV2 Hybrid experiments found!")
            print("Please run the experiment first using:")
            print(
                "   python scripts/experiments/swinv2_hybrid/"
                "run_swinv2_hybrid_experiment.py"
            )
            return

        # Run visualization analysis
        self.run_visualization_analysis(experiment_dirs)

        # Generate performance summary
        self.generate_performance_summary(experiment_dirs)

        # Compare with baseline if requested
        if compare_baseline:
            self.compare_with_baseline()

        # Export comprehensive report if requested
        if export_report:
            self.export_comprehensive_report()

        print(f"\n✅ Analysis complete! Results saved to: {self.output_dir}")
        print("\n📋 Generated Files:")
        for file_path in self.output_dir.rglob("*"):
            if file_path.is_file():
                print(f"   📄 {file_path.relative_to(self.output_dir)}")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="SwinV2 Hybrid Experiment Analysis"
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare results with baseline experiments",
    )
    parser.add_argument(
        "--export-report",
        action="store_true",
        help="Export comprehensive analysis report",
    )

    args = parser.parse_args()

    # Initialize analyzer
    project_root = Path(__file__).parent.parent.parent.parent
    analyzer = SwinV2HybridAnalyzer(project_root)

    # Run analysis
    analyzer.run_complete_analysis(
        compare_baseline=args.compare_baseline,
        export_report=args.export_report,
    )


if __name__ == "__main__":
    main()
