#!/usr/bin/env python3
"""
Tutorial 02 Visualization Script

This script uses the generic experiment visualizer to analyze Tutorial 02
experiments. It provides a convenient wrapper for the tutorial-specific
experiments.

Reference: docs/tutorials/02_custom_experiment_cli.md

Usage:
    python scripts/experiments/tutorial_02/tutorial_02_visualize.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run visualization for Tutorial 02 experiments."""
    print("Tutorial 02 - Experiment Visualization")
    print("=" * 50)
    print("Reference: docs/tutorials/02_custom_experiment_cli.md")
    print("=" * 50)

    # Define Tutorial 02 experiment directories
    tutorial_02_experiments = [
        "src/crackseg/outputs/experiments/20250723-005521-default",
        "src/crackseg/outputs/experiments/20250723-005704-default",
        "src/crackseg/outputs/experiments/20250723-010032-default",
    ]

    # Check which experiments exist
    existing_experiments = []
    for exp_dir in tutorial_02_experiments:
        if Path(exp_dir).exists():
            existing_experiments.append(exp_dir)
            print(f"✅ Found experiment: {exp_dir}")
        else:
            print(f"❌ Experiment not found: {exp_dir}")

    if not existing_experiments:
        print("❌ No Tutorial 02 experiments found!")
        print("Please run the experiments first using the tutorial scripts.")
        return

    # Build command for generic visualizer
    experiment_dirs = ",".join(existing_experiments)

    cmd = [
        sys.executable,
        "scripts/experiments/experiment_visualizer.py",
        "--experiment-dirs",
        experiment_dirs,
        "--output-dir",
        "docs/reports/tutorial_02_analysis",
        "--title",
        "Tutorial 02: Custom Experiments Analysis",
    ]

    print("\n🚀 Running generic visualizer with command:")
    print(f"   {' '.join(cmd)}")
    print()

    # Run the generic visualizer
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print("\n✅ Tutorial 02 visualization completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running visualization: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
