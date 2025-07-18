# Pavement Crack Segmentation Project

> **Note:** This project was developed with the assistance of AI tools.

## Overview

Advanced deep learning system for semantic segmentation of cracks in asphalt pavement.
Features a modular, reproducible, and extensible codebase designed for both research and production environments.

### Key features

- 🧠 **Modular Architecture**: Composable encoders, decoders, and training components
- 🔧 **Production Ready**: Type-safe, tested, and documented codebase
- 📊 **Research Focused**: Experiment tracking, reproducible configurations
- 🎯 **High Quality**: 66% test coverage, strict code quality standards
- ⚙️ **Configurable**: Hydra-based configuration system
- 📋 **Task Management**: Integrated Task Master AI workflow for development

## Quickstart

### Prerequisites

**System Dependencies Required:**

- Git, Conda/Miniconda, Graphviz
- Optional: CUDA Toolkit (for GPU acceleration)

For detailed installation instructions, see [**System Dependencies Guide**](docs/guides/SYSTEM_DEPENDENCIES.md).

**Quick verification:**

```bash
# Verify all system dependencies
python scripts/verify_system_dependencies.py
```

### 1. Environment Setup

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate crackseg

# Copy environment template and configure
cp .env.example .env  # Edit with your specific settings
```

### 2. Data Preparation

Place your crack segmentation data in the following structure:

```txt
data/
├── train/
│   ├── images/    # Training images (.jpg, .png)
│   └── masks/     # Binary masks (.png)
├── val/
│   ├── images/    # Validation images
│   └── masks/     # Validation masks
└── test/
    ├── images/    # Test images
    └── masks/     # Test masks
```

### 3. Train a Model

You can train a model using the command line for reproducibility and scripting, or use the
interactive GUI for experimentation and visualization.

#### Using the Command Line

```bash
# Basic training with default U-Net architecture
python run.py

# Train with specific configuration
python run.py model=architectures/swin_unet data.batch_size=4

# Advanced configuration for 8GB VRAM
python run.py data.batch_size=4 \
              training.mixed_precision=true \
              training.gradient_accumulation_steps=4
```

#### Using the Interactive GUI

For a user-friendly experience, launch the Streamlit-based GUI:

```bash
streamlit run scripts/gui/app.py
```

The GUI allows you to:

- Visually browse and load configuration files.
- Edit configurations with a real-time validation editor.
- Monitor training progress with live logs and charts.
- View results and model predictions.

Refer to the [tutorials in `docs/tutorials/`](docs/tutorials/) for a detailed walkthrough.

### 4. Evaluate a Model

```bash
# Evaluate using the evaluation module
python -m src.evaluation \
    model.checkpoint_path=outputs/experiments/{timestamp}-{config}/checkpoints/best_model.pth

# Or use the wrapper script
python src/evaluate.py
```

For detailed workflow and advanced configuration options, see [**Training Workflow Guide**](docs/guides/WORKFLOW_TRAINING.md).

## Project Structure

For a comprehensive view of the project organization, see [**Project Structure Guide**](.cursor/rules/project-structure.mdc).

### Core Directories

- **`src/`** — Core application code (models, data, training, evaluation)
- **`configs/`** — Hydra YAML configurations (modular and composable)
- **`tests/`** — Comprehensive test suite (66% coverage, 866 tests)
- **`docs/`** — Documentation, guides, and organized reports
- **`outputs/`** — Training results, logs, checkpoints (git ignored)
- **`scripts/`** — Utility scripts and experimental code
- **`tasks/`** — Task Master project management files

### Key Components

- **Model Architectures**: `src/model/architectures/` (U-Net, SwinUNet variants)
- **Data Pipeline**: `src/data/` (datasets, transforms, loaders)
- **Training System**: `src/training/` (losses, optimizers, schedulers)
- **Evaluation Suite**: `src/evaluation/` (metrics, visualization, reporting)
- **Configuration**: `configs/` (model, training, data configurations)

> **Note:** Scripts in `scripts/` are for experimentation and utilities only. Do not import them in
> core modules. Clean up temporary files like `__pycache__` regularly.

## Reports & Analytics

All project reports, analysis, and documentation are organized in `docs/reports/` with the following
structure:

- **`testing/`** — Test coverage reports, improvement plans, and testing priorities
- **`coverage/`** — Code coverage analysis and gap reports
- **`tasks/`** — Task completion summaries and complexity analysis
- **`models/`** — Model architecture analysis and import catalogs
- **`project/`** — Project-level plans and verification reports
- **`archive/`** — Historical reports and deprecated documentation

### Current Metrics

- 📊 **Test Coverage**: 66% global (target >80% for new code and critical modules, see CI coverage badge)
- 🧪 **Tests Implemented**: 866 tests across unit and integration suites
- 🏗️ **Architecture**: Modular design with factory patterns
- 🔧 **Code Quality**: 100% type coverage, Black/Ruff compliance

For complete report navigation, see [`docs/reports/README.md`](docs/reports/README.md).

### Report Organization Tool

Maintain organized reports using the automated organizer:

```bash
# Check current organization status
python scripts/utils/organize_reports.py --report

# Preview organization changes (dry run)
python scripts/utils/organize_reports.py --dry-run

# Apply organization changes
python scripts/utils/organize_reports.py
```

## Configuration System

The project uses **Hydra** for hierarchical configuration management:

### Configuration Structure

- **Model Architectures**: `configs/model/architectures/` (U-Net, SwinUNet, etc.)
- **Training Components**: `configs/training/` (losses, metrics, schedulers)
- **Data Processing**: `configs/data/` (dataloaders, transforms)
- **Evaluation**: `configs/evaluation/` (metrics, thresholds)

### Usage Examples

```bash
# Switch model architecture
python run.py model=architectures/swin_unet

# Modify training parameters
python run.py training.optimizer.lr=0.001 training.batch_size=8

# Combine multiple overrides
python run.py model=architectures/unet \
              training.loss=dice_focal \
              data.batch_size=4 \
              experiment_name="unet_focal_exp"
```

All configurations are composable and can be overridden via CLI or configuration files.

## Task Management & Development Workflow

This project integrates **Task Master AI** for structured, AI-driven development workflow management:

### Key Features

- **🎯 Structured Tasks**: Break down complex features into manageable subtasks
- **📋 Dependency Management**: Automatic task ordering and prerequisite tracking
- **🤖 AI Integration**: Task generation from requirements and intelligent progress tracking
- **📊 Progress Analytics**: Complexity analysis and development insights

### Quick Start

```bash
# View current tasks and project status
task-master list

# Find the next task to work on
task-master next

# View detailed task information
task-master show <task-id>

# Mark tasks as completed
task-master set-status --id=<task-id> --status=done
```

### Documentation

- **📖 Integration Guide**: Task Master commands integrated for structured development workflow
- **🔧 Command Reference**: CLI commands available through `task-master` package
- **🏗️ Project Structure**: Task files organized in `tasks/` directory

The Task Master system helps maintain development momentum through structured task management,
automated progress tracking, and intelligent workflow guidance.

## Code Quality & Development Standards

This project maintains strict code quality standards:

### Quality Tools

- **🔧 Black**: Automatic code formatting (79 char line length)
- **🔍 Ruff**: Comprehensive linting and style checking
- **⚡ Basedpyright**: Static type checking (mandatory type annotations)
- **🧪 Pytest**: Testing framework with coverage reporting

### Quality Requirements

- **Type Annotations**: Required for all functions, methods, and variables
- **Test Coverage**: Target >80% for new code, current 66% overall
- **Code Formatting**: Must pass Black and Ruff checks
- **Documentation**: Docstrings required for all public APIs

### Pre-commit Workflow

```bash
# Manual quality check workflow
black .
ruff . --fix
basedpyright .

# Run tests with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Only commit if all checks pass
```

### Optional Pre-commit Hooks

```bash
# Install pre-commit hooks (optional)
conda install -c conda-forge pre-commit
pre-commit install
```

Configuration files:

- **Centralized config**: `configs/linting/config.yaml`
- **Tool configuration**: `pyproject.toml`
- **Type checking**: `pyrightconfig.json`

For detailed code quality guidelines, see [**Coding Standards**](.cursor/rules/coding-preferences.mdc).

## Training & Evaluation Workflow

### Training Process

1. **Configure Environment**: Set up conda environment and data structure
2. **Select Architecture**: Choose from available models in `configs/model/architectures/`
3. **Configure Training**: Set loss functions, optimizers, and hyperparameters
4. **Monitor Progress**: Training generates logs, metrics, and checkpoints in `outputs/`
5. **Evaluate Results**: Use comprehensive evaluation suite for analysis

### Experiment Tracking

Training outputs are organized by timestamp and configuration:

```txt
outputs/experiments/
└── {timestamp}-{config_name}/
    ├── checkpoints/    # Model checkpoints (.pth files)
    ├── logs/          # TensorBoard training logs
    ├── metrics/       # Training/validation metrics (CSV/JSON)
    └── results/       # Prediction outputs and visualizations
```

### Monitoring Tools

```bash
# Visualize training progress
tensorboard --logdir outputs/experiments/

# Monitor GPU usage during training
python run.py training.log_gpu_memory=true
```

For complete workflow details, see [**Training Workflow Guide**](docs/guides/WORKFLOW_TRAINING.md).

## Testing Framework

Comprehensive test suite ensuring code reliability:

### Test Organization

- **Unit Tests**: `tests/unit/` - Individual component testing
- **Integration Tests**: `tests/integration/` - Module interaction testing
- **End-to-End Tests**: `tests/integration/end_to_end/` - Complete pipeline testing

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only

# Run tests with detailed output
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Test Coverage

- **Current Coverage**: 66% (5,333/8,065 lines)
- **Test Count**: 866 implemented tests
- **Coverage Reports**: Available in `htmlcov/` after running with `--cov-report=html`

For testing guidelines and details, see [`tests/README.md`](tests/README.md).

## Environment Variables

Configure the project using environment variables in `.env`:

```bash
# Copy template and edit
cp .env.example .env
```

### Key Variables

- **`ANTHROPIC_API_KEY`**: API key for Task Master integration
- **`DEBUG`**: Enable/disable debug mode (`true`/`false`)
- **`CUDA_VISIBLE_DEVICES`**: GPU device selection
- **`PYTHONPATH`**: Automatically configured by `run.py`

## Dependency Management

### Environment Updates

```bash
# Check for package updates
python scripts/utils/check_updates.py

# Update conda environment
conda env update -f environment.yml --prune

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

## Task Management Integration

This project integrates with **Task Master** for structured development:

### Key Features 3

- 📋 **Structured Tasks**: Hierarchical task breakdown with dependencies
- 📊 **Progress Tracking**: Real-time task status and completion tracking
- 🤖 **AI Integration**: Automated task generation from PRDs
- 🔄 **Workflow Management**: Integration with development workflow

### Task Master Commands

```bash
# View current tasks
task-master list

# Show next task to work on
task-master next

# Update task status
task-master set-status --id=12 --status=done

# Generate detailed task files
task-master generate
```

For comprehensive Task Master integration, see [`README-task-master.md`](README-task-master.md).

## Contributing

### Development Workflow

1. **Check Current Tasks**: Use `task-master next` to see priority work
2. **Create Feature Branch**: Follow git workflow standards
3. **Implement Changes**: Follow code quality standards
4. **Run Quality Checks**: Ensure Black, Ruff, and basedpyright pass
5. **Write/Update Tests**: Maintain or improve test coverage
6. **Update Documentation**: Keep docs synchronized with changes
7. **Submit for Review**: Create PR with comprehensive description

### Contribution Guidelines

- **Follow [Coding Standards](.cursor/rules/coding-preferences.mdc)**: Type annotations, formatting,
  linting
- **Add Comprehensive Tests**: Unit and integration tests for new features
- **Update Documentation**: Keep README, guides, and docstrings current
- **Use Task Master**: Track work using the integrated task management system

For detailed contribution guidelines, see [`docs/guides/CONTRIBUTING.md`](docs/guides/CONTRIBUTING.md).

## Architecture & Design

### Design Principles

1. **Modularity**: Components with single responsibilities and loose coupling
2. **Configurability**: All parameters manageable via Hydra configurations
3. **Testability**: Comprehensive test coverage with clear interfaces
4. **Reproducibility**: Deterministic training with seed management
5. **Extensibility**: Easy to add new models, losses, and components

### Key Patterns

- **Factory Pattern**: Dynamic component creation via registries
- **Configuration Composition**: Hierarchical Hydra configuration system
- **Dependency Injection**: Testable components with injected dependencies
- **Abstract Base Classes**: Clear interfaces for extensibility

## License

MIT License. See [`LICENSE`](LICENSE) for details.

---

## Tips & Best Practices

- 🚀 **Use `run.py`** as the main entry point for training
- ⚙️ **Leverage Hydra configs** for all parameter management
- 🧪 **Run quality checks** before committing code changes
- 📊 **Monitor experiments** using TensorBoard and organized outputs
- 📚 **Check [`docs/reports/`](docs/reports/)** for latest project analysis
- 🔧 **Use Task Master** for structured development workflow
- 🎯 **Follow type annotations** for better code reliability
- 📈 **Maintain test coverage** when adding new features

## System Requirements

- **Python 3.12+** (required for modern type annotations and features)
- **CUDA-capable GPU** (recommended for training)
- **8GB+ RAM** (system memory)
- **6GB+ VRAM** (GPU memory for training)
- **conda/mamba** (for environment management)

All dependencies and tools are configured and tested for **Python 3.12**. Please ensure your
environment matches this version for full compatibility.

## Project Status

**All major roadmap tasks and subtasks are completed. The system is production-ready and open for**
**advanced contributions.**

## End-to-End Testing & Performance Benchmarking

- Advanced E2E testing infrastructure based on Docker and Selenium Grid with multi-browser support
  (Chrome, Firefox, Edge, mobile emulation).
- Automated orchestration of containers, artifact management, resource cleanup, and reporting.
- Automated performance tests with metrics for page load, memory/CPU/VRAM usage, and performance
  gates integrated into CI/CD.
- See details and scripts in `tests/e2e/` and documentation in `docs/guides/`.

## Documentation & Tutorials

- **User & Developer Guides:**
  - [Workflow Training Guide](docs/guides/WORKFLOW_TRAINING.md)
  - [Technical Architecture](docs/guides/TECHNICAL_ARCHITECTURE.md)
  - [Troubleshooting Guide](docs/guides/TROUBLESHOOTING.md)
  - [Tutorials](docs/tutorials/)
  - [Docker Testing Infrastructure](tests/docker/README-DOCKER-TESTING.md)

### CI/CD Automation

- **CI/CD Automation**: The pipeline runs tests, quality checks (black, ruff, basedpyright),
  coverage, vulnerability scanning (OSV-Scanner), and performance gates. Commits are only accepted
  if all checks pass.
