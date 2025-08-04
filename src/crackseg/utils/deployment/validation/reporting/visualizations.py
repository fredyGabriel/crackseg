"""Visualization generation for validation reporting."""

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from .config import ValidationReportData


class ChartGenerator:
    """Generator for validation report charts."""

    def create_performance_radar_chart(
        self, report_data: "ValidationReportData", save_dir: Path
    ) -> Path:
        """Create performance radar chart.

        Args:
            report_data: Validation report data
            save_dir: Directory to save chart

        Returns:
            Path to generated chart
        """
        # Performance metrics for radar chart
        categories = [
            "Performance Score",
            "Security Score",
            "Compatibility Score",
            "Functional Tests",
            "Resource Efficiency",
        ]

        values = [
            report_data.performance_score * 100,
            report_data.security_score * 10,
            report_data.compatibility_score * 100,
            100 if report_data.functional_tests_passed else 0,
            max(0, 100 - (report_data.memory_usage_mb / 2048) * 100),
        ]

        # Create radar chart
        angles = [
            i / len(categories) * 2 * 3.14159 for i in range(len(categories))
        ]
        angles += angles[:1]  # Close the loop
        values += values[:1]

        _, ax = plt.subplots(
            figsize=(8, 8), subplot_kw={"projection": "polar"}
        )
        ax.plot(angles, values, "o-", linewidth=2, label="Validation Scores")
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title("Validation Performance Radar Chart", size=16, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        chart_path = save_dir / "performance_radar.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()

        return chart_path

    def create_resource_utilization_chart(
        self, report_data: "ValidationReportData", save_dir: Path
    ) -> Path:
        """Create resource utilization bar chart.

        Args:
            report_data: Validation report data
            save_dir: Directory to save chart

        Returns:
            Path to generated chart
        """
        resources = ["CPU", "GPU", "Memory (GB)", "Disk (MB)"]
        values = [
            report_data.cpu_usage_percent,
            report_data.gpu_usage_percent,
            report_data.memory_usage_mb / 1024,  # Convert to GB
            report_data.disk_usage_mb,
        ]

        _, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(
            resources,
            values,
            color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"],
        )

        # Add value labels on bars
        for bar, value in zip(bars, values, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{value:.1f}",
                ha="center",
                va="bottom",
            )

        ax.set_title("Resource Utilization", fontsize=14, pad=20)
        ax.set_ylabel("Usage (%)")
        ax.set_ylim(0, max(values) * 1.1)

        chart_path = save_dir / "resource_utilization.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()

        return chart_path

    def create_security_score_chart(
        self, report_data: "ValidationReportData", save_dir: Path
    ) -> Path:
        """Create security score visualization.

        Args:
            report_data: Validation report data
            save_dir: Directory to save chart

        Returns:
            Path to generated chart
        """
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Security score gauge
        score = report_data.security_score
        ax1.pie(
            [score, 10 - score],
            colors=["#FF6B6B" if score < 8 else "#4ECDC4", "#F0F0F0"],
            startangle=90,
            counterclock=False,
        )
        ax1.text(
            0,
            0,
            f"{score:.1f}/10",
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
        )
        ax1.set_title("Security Score", fontsize=14)

        # Vulnerabilities bar chart
        vulnerabilities = report_data.vulnerabilities_found
        ax2.bar(
            ["Vulnerabilities"],
            [vulnerabilities],
            color="#FF6B6B" if vulnerabilities > 0 else "#4ECDC4",
        )
        ax2.set_title("Security Vulnerabilities Found", fontsize=14)
        ax2.set_ylabel("Count")
        ax2.text(
            0,
            vulnerabilities + 0.1,
            str(vulnerabilities),
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()
        chart_path = save_dir / "security_score.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()

        return chart_path

    def create_compatibility_heatmap(
        self, report_data: "ValidationReportData", save_dir: Path
    ) -> Path:
        """Create compatibility matrix heatmap.

        Args:
            report_data: Validation report data
            save_dir: Directory to save chart

        Returns:
            Path to generated chart
        """
        compatibility_matrix = [
            ["Python", "Dependencies", "Environment"],
            [
                report_data.python_compatible,
                report_data.dependencies_compatible,
                report_data.environment_compatible,
            ],
        ]

        df = pd.DataFrame(
            compatibility_matrix[1:], columns=compatibility_matrix[0]
        )
        df_numeric = df.astype(int)

        _fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(
            df_numeric,
            annot=df,
            fmt="s",
            cmap="RdYlGn",
            cbar_kws={"label": "Compatibility Status"},
        )
        ax.set_title("Compatibility Matrix", fontsize=14, pad=20)
        ax.set_ylabel("Components")

        chart_path = save_dir / "compatibility_heatmap.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()

        return chart_path
