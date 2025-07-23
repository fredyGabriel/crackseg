# CrackSeg Tutorials

Welcome to the CrackSeg tutorials! These guides will help you get started with the project, from
basic training to advanced customization.

## Tutorial Overview

The tutorials are organized into two tracks:

### 🖥️ **GUI Track** (Recommended for Beginners)

Use the interactive web interface for a visual, user-friendly experience.

### 💻 **CLI Track** (Recommended for Advanced Users)

Use command-line tools for automation, scripting, and server environments.

## Tutorial Tracks

### GUI Track

| Tutorial | Description | Prerequisites |
|----------|-------------|---------------|
| [01_basic_training.md](01_basic_training.md) | Basic training workflow using the GUI | Project installation |
| [02_custom_experiment.md](02_custom_experiment.md) | Creating custom experiments with GUI editor | Tutorial 1 |
| [03_extending_project.md](03_extending_project.md) | Adding custom components (loss functions, models) | Tutorial 2 |

### CLI Track

| Tutorial | Description | Prerequisites |
|----------|-------------|---------------|
| [01_basic_training_cli.md](01_basic_training_cli.md) | Basic training workflow using CLI only | Project installation |
| [02_custom_experiment_cli.md](02_custom_experiment_cli.md) | Creating custom experiments with CLI and YAML | Tutorial 1 CLI |
| [03_extending_project_cli.md](03_extending_project_cli.md) | Adding custom components using CLI | Tutorial 2 CLI |

## Prerequisites

Before starting any tutorial, ensure you have:

1. **Project Installed**: Follow the [CLEAN_INSTALLATION.md](../guides/CLEAN_INSTALLATION.md) guide
2. **Conda Environment Activated**: `conda activate crackseg`
3. **Package Installed**: `pip install -e . --no-deps` (for conda environments)
4. **Installation Verified**: `python -c "import crackseg; print('✅ Success')"`

## Quick Start

### For GUI Users (Beginners)

```bash
# 1. Install and verify
conda activate crackseg
pip install -e . --no-deps
python -c "import crackseg; print('✅ Success')"

# 2. Start GUI
streamlit run gui/app.py

# 3. Follow Tutorial 1: Basic Training
```

### For CLI Users (Advanced)

```bash
# 1. Install and verify
conda activate crackseg
pip install -e . --no-deps
python -c "import crackseg; print('✅ Success')"

# 2. Run basic training
python run.py --config-name basic_verification

# 3. Follow Tutorial 1 CLI: Basic Training Workflow
```

## Tutorial Progression

### Beginner Path (GUI)

```bash
01_basic_training.md → 02_custom_experiment.md → 03_extending_project.md
```

### Advanced Path (CLI)

```bash
01_basic_training_cli.md → 02_custom_experiment_cli.md → 03_extending_project_cli.md
```

### Mixed Path (Recommended)

```bash
01_basic_training.md → 02_custom_experiment_cli.md → 03_extending_project_cli.md
```

## What You'll Learn

### Tutorial 1: Basic Training

- ✅ Install and verify the project
- ✅ Run your first training experiment
- ✅ Monitor training progress
- ✅ View and analyze results

### Tutorial 2: Custom Experiments

- ✅ Create custom configurations
- ✅ Modify training parameters
- ✅ Compare multiple experiments
- ✅ Manage experiment outputs

### Tutorial 3: Extending the Project

- ✅ Add custom loss functions
- ✅ Create new model architectures
- ✅ Implement custom optimizers
- ✅ Use the registry pattern

## Common Commands Reference

### Installation & Verification

```bash
conda activate crackseg
pip install -e . --no-deps
python -c "import crackseg; print('✅ Success')"
```

### Basic Training

```bash
# GUI
streamlit run gui/app.py

# CLI
python run.py --config-name basic_verification
```

### Custom Experiments

```bash
# CLI with overrides
python run.py --config-name basic_verification training.learning_rate=0.001

# CLI with custom config
python run.py --config-name my_experiment
```

### Quality Gates

```bash
conda activate crackseg
black .
python -m ruff . --fix
basedpyright .
```

## Troubleshooting

### Common Issues

1. **Import Error**: No module named 'crackseg'

    - Solution: Run `pip install -e . --no-deps` from project root (for conda environments)
    - Alternative: Run `pip install -e .` (for pip environments)

2. **GUI Not Starting**

    - Solution: Install Streamlit: `pip install streamlit`
    - Alternative: Use CLI tutorials

3. **Configuration Errors**

    - Solution: Check YAML syntax and file paths
    - Verify all referenced components exist

4. **Out of Memory**

    - Solution: Reduce batch size in configuration
    - Use smaller model architecture

### Getting Help

1. **Check the troubleshooting sections** in each tutorial
2. **Review the [CLEAN_INSTALLATION.md](../guides/CLEAN_INSTALLATION.md)** guide
3. **Examine the [project documentation](../index.md)**
4. **Check the [configuration examples](../configs/)** directory

## Next Steps

After completing the tutorials:

1. **Explore the Configuration System**: Check `configs/` directory for available options
2. **Review the API Documentation**: See `docs/api/` for detailed component documentation
3. **Run the Test Suite**: Execute `python -m pytest tests/` to verify everything works
4. **Join the Community**: Contribute your custom components and experiments

## Tutorial Feedback

If you encounter issues or have suggestions for improving the tutorials:

1. Check the troubleshooting sections first
2. Review the project documentation
3. Report issues with specific error messages and steps to reproduce

---

Happy Training! 🚀

Start with [Tutorial 1: Basic Training](01_basic_training.md) or
[Tutorial 1 CLI: Basic Training Workflow](01_basic_training_cli.md) depending on your preference.
