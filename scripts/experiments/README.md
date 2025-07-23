# Experiment Scripts

This directory contains scripts for running, analyzing, and visualizing experiments in the CrackSeg project.

## Structure

```bash
scripts/experiments/
├── README.md                           # This file
├── experiment_visualizer.py            # Generic experiment visualization tool
├── tutorial_02/                        # Tutorial 02 specific scripts
│   ├── tutorial_02_compare.py          # Simple comparison for tutorial 02
│   ├── tutorial_02_visualize.py        # Tutorial 02 visualization wrapper
│   └── tutorial_02_batch.ps1           # Batch execution for tutorial 02
└── [other experiment scripts...]       # Other experiment-specific scripts
```

## Generic Tools

### `experiment_visualizer.py`

A **generic, reusable visualization tool** that can analyze any set of experiments.

**Features:**

- Loads experiment data from any experiment directory
- Creates training curves, performance radar charts, and detailed analysis
- Supports multiple input methods (experiment names, directory paths, auto-discovery)
- Configurable output directory and analysis title
- Handles any number of experiments

**Usage Examples:**

```bash
# Analyze specific experiments by name
python scripts/experiments/experiment_visualizer.py --experiments exp1,exp2,exp3

# Analyze experiments by directory paths
python scripts/experiments/experiment_visualizer.py --experiment-dirs path1,path2,path3

# Automatically find and analyze recent experiments
python scripts/experiments/experiment_visualizer.py --auto-find --max-experiments 5

# Custom output and title
python scripts/experiments/experiment_visualizer.py --experiments exp1,exp2 --output-dir my_analysis --title "My Experiment Analysis"
```

**Arguments:**

- `--experiments`: Comma-separated list of experiment names
- `--experiment-dirs`: Comma-separated list of experiment directory paths
- `--output-dir`: Output directory for plots and analysis (default: docs/reports/experiment_analysis)
- `--title`: Title for the analysis (default: "Experiment Analysis")
- `--auto-find`: Automatically find recent experiment directories
- `--max-experiments`: Maximum number of experiments to analyze (when using --auto-find)

## Tutorial-Specific Scripts

### `tutorial_02/`

Scripts specifically designed for Tutorial 02: "Creating Custom Experiments (CLI Only)".

**Scripts:**

- `tutorial_02_compare.py`: Simple text-based comparison of tutorial 02 experiments
- `tutorial_02_visualize.py`: Wrapper that uses the generic visualizer for tutorial 02 experiments
- `tutorial_02_batch.ps1`: PowerShell script to run all tutorial 02 experiments

**Usage:**

```bash
# Run comparison
python scripts/experiments/tutorial_02/tutorial_02_compare.py

# Run visualization (uses generic visualizer)
python scripts/experiments/tutorial_02/tutorial_02_visualize.py

# Run batch execution
.\scripts\experiments\tutorial_02\tutorial_02_batch.ps1
```

## Best Practices

### Creating New Experiment Scripts

1. **Use the generic visualizer** when possible instead of creating custom visualization code
2. **Create tutorial-specific wrappers** that use the generic tools
3. **Keep experiment-specific logic** in dedicated subdirectories
4. **Follow the naming convention**: `tutorial_XX_` for tutorial-specific scripts

### Example: Creating Tutorial 03 Scripts

```bash
# Create directory
mkdir scripts/experiments/tutorial_03

# Create wrapper script
cat > scripts/experiments/tutorial_03/tutorial_03_visualize.py << 'EOF'
#!/usr/bin/env python3
"""Tutorial 03 visualization wrapper."""
import subprocess
import sys

def main():
    cmd = [
        sys.executable,
        "scripts/experiments/experiment_visualizer.py",
        "--experiments", "tutorial_03_exp1,tutorial_03_exp2",
        "--output-dir", "docs/reports/tutorial_03_analysis",
        "--title", "Tutorial 03: Advanced Experiments"
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
EOF
```

## Output Structure

The generic visualizer creates the following output structure:

```bash
docs/reports/experiment_analysis/
├── training_curves.png          # Training curves comparison
├── performance_radar.png        # Performance radar chart
└── experiment_comparison.csv    # Tabular comparison data
```

## Dependencies

The generic visualizer requires:

- `matplotlib`: For creating plots
- `pandas`: For data manipulation
- `seaborn`: For plot styling
- `numpy`: For numerical operations

Install with:

```bash
conda activate crackseg
pip install matplotlib pandas seaborn numpy
```

## References

- **Tutorial 02**: [docs/tutorials/02_custom_experiment_cli.md](../../../docs/tutorials/02_custom_experiment_cli.md)
- **Configurations**: [configs/experiments/tutorial_02/](../../../configs/experiments/tutorial_02/)
- **Project Structure**: [docs/reports/project_tree.md](../../../docs/reports/project_tree.md)
