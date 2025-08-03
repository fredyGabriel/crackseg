# Contribution Guide

Thank you for your interest in contributing to the CrackSeg project. This document provides
specific guidelines to contribute effectively, complementing our professional development standards.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Quality Standards](#quality-standards)
- [Submitting Changes](#submitting-changes)
- [Code Review](#code-review)

## Environment Setup

### 1. Clone and Configure

```bash
git clone https://github.com/your-user/crackseg.git
cd crackseg

# Create conda environment
conda env create -f environment.yml
conda activate torch

# Configure environment variables
cp .env.example .env
# Edit .env as needed
```

### 2. Verify Setup

```bash
# Check quality tools (mandatory)
black --version
ruff --version
basedpyright --version

# Run initial checks
black .
ruff . --fix
basedpyright .
```

## Project Structure

The project follows a modular architecture for extensibility:

```txt
crackseg/
├── .cursor/rules/        # Professional development rules
├── configs/             # Hydra configurations by component
│   ├── data/            # Dataset and dataloaders
│   ├── model/           # Architectures and components
│   └── training/        # Training and evaluation
├── src/                 # Main source code
│   ├── data/            # Data modules and transforms
│   ├── model/           # Modular models and components
│   ├── training/        # Training and loss modules
│   ├── evaluation/      # Evaluation and metrics
│   └── utils/           # Common utilities
├── tests/               # Unit and integration tests
└── docs/guides/         # Project-specific documentation (organized by category)
```

### Architectural Principles

1. **Modularity**: Each component has a unique, well-defined responsibility
2. **Extensibility**: New components can be added without modifying existing code
3. **Configurability**: All parameters are configurable via YAML files
4. **Quality**: Code must pass basedpyright, Black, and Ruff with no errors

## Development Workflow

### 1. Planning

Before you start coding:

- Review existing issues or create a new one
- Discuss the approach if it is a significant change
- Clearly define the scope of the change
- **Refer to our development rules located in the `.cursor/rules/` directory for the detailed process.**

### 2. Implementation

```bash
# Create a branch for your work
git checkout -b feature/feature-name  # For new features
git checkout -b fix/bug-name         # For bug fixes

# During development, follow our professional standards.
# See the files in `.cursor/rules/` for full details.
```

### 3. Continuous Verification

```bash
# Run quality checks (mandatory before commit)
black .
ruff . --fix
basedpyright .

# Run tests
pytest tests/ --cov=src --cov-report=term-missing
```

## Quality Standards

**The project maintains strict professional standards. See our specific rule files in the
`.cursor/rules/` directory for details.**

### 📋 Mandatory Development Rules

The following topics are covered by our rule system:

- **Code Preferences**: Technical standards, mandatory typing, quality tools.
- **Testing Standards**: Testing strategies, coverage, mocking.
- **Git Standards**: Commit format, branching, collaboration.
- **ML Standards**: Reproducibility, experiments, VRAM optimization.

### ⚡ Quick Verification

```bash
# All tools must pass with no errors
black .                    # Auto-formatting
ruff . --fix              # Linting and autofix
basedpyright .            # Strict type checking
pytest tests/ --cov=src   # Tests with coverage
```

### 🎯 ML-Specific Requirements

- **Complete type annotations**: All tensors, models, and functions
- **VRAM management**: Optimized for RTX 3070 Ti (8GB)
- **Reproducibility**: Seeds, deterministic configurations
- **Documentation**: Detailed docstrings for model architectures

## Submitting Changes

### Commit Process

```bash
# Mandatory pre-commit verification
black .
ruff . --fix
basedpyright .
pytest

# Commit following conventions (see git-standards.mdc)
git add .
git commit -m "feat(model): Implement SwinV2-Tiny encoder

- Add hierarchical attention for long-range dependencies
- Optimize for 8GB VRAM with gradient accumulation
- Achieve IoU: 0.847 on validation set"
```

### Creating a Pull Request

1. **Update your branch**:

   ```bash
   git fetch origin
   git rebase origin/main
   git push origin branch-name
   ```

2. **PR format**:
   - Title: `type(scope): short description`
   - Detailed description of technical changes
   - Link issue: `Fixes #issueNum`
   - Include performance metrics if applicable

## Code Review

### Automatic Checks

All contributions must pass:

- ✅ **basedpyright**: No typing errors
- ✅ **Black**: Consistent formatting
- ✅ **Ruff**: No linting violations
- ✅ **pytest**: All tests pass
- ✅ **Coverage**: >80% on modified modules

### Manual Review Criteria

- **Functionality**: Does it meet technical requirements?
- **Architecture**: Does it follow established modular patterns?
- **ML/Research**: Does it maintain reproducibility and optimization?
- **Documentation**: Are relevant docs updated?
- **Integration**: Does it integrate correctly with existing components?

### ML-Specific Standards

- **Model Validation**: Tests with synthetic data
- **Memory Management**: VRAM usage monitoring
- **Metrics**: IoU, F1-Score, baseline comparison
- **Configurability**: Parameters accessible via Hydra configs

## Development Resources

### 📚 Essential Documentation

- **Configuration**: [WORKFLOW_TRAINING.md](WORKFLOW_TRAINING.md) - Training workflow
- **Loss Registry**: [loss_registry_usage.md](loss_registry_usage.md) - Loss system
- **Project Structure**: Refer to our project structure guidelines for file organization.

### 🛠️ Development Tools

```bash
# Main tools (installed with environment.yml)
basedpyright    # Type checker (replaces mypy)
black           # Formatter
ruff            # Linter (replaces flake8, isort, pylint)
pytest          # Testing framework
tensorboard     # Experiment monitoring
```

### 🔧 Utility Scripts

```bash
# Full quality check
python -c "
import subprocess
tools = ['black .', 'ruff . --fix', 'basedpyright .']
for tool in tools:
    result = subprocess.run(tool.split(), capture_output=True, text=True)
    print(f'{tool}: {'✅ PASS' if result.returncode == 0 else ' FAIL'}')
"
```

---

## Contact and Support

For questions about:

- **Technical standards**: See rules in `.cursor/rules/`
- **Specific issues**: Open an issue in the repository
- **Implementation doubts**: See documentation in `docs/guides/` (organized by category)

**Thank you for contributing to the advancement of crack segmentation research!** 🚀
