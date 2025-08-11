# Pavement Crack Segmentation Project

<!-- markdownlint-disable MD013 -->

> **Note:** This project was developed with the assistance of AI tools.

## Overview

Advanced deep learning system for semantic segmentation of cracks in asphalt pavement.
Features a modular, reproducible, and extensible codebase designed for both research and production environments.

### Key Features

- 🧠 **Modular Architecture**: Composable encoders, decoders, and training components
- 🔧 **Production Ready**: Type-safe, tested, and documented codebase
- 📊 **Research Focused**: Experiment tracking, reproducible configurations
- 🎯 **High Quality**: 66% test coverage, strict code quality standards
- ⚙️ **Configurable**: Hydra-based configuration system
- 📋 **Task Management**: Integrated Task Master AI workflow for development
- 🚀 **Modern Stack**: PyTorch 2.7 + CUDA 12.9 with optimized dependencies

## What's New in v0.2.0 (July 2025)

### Major Dependency Modernization

- **🎨 Visualization**: Graphviz → Matplotlib for architecture diagrams (simpler setup, cross-platform)
- **🤖 Computer Vision**: TorchVision → TIMM + Albumentations (more models, better transforms)
- **⚡ PyTorch Ecosystem**: Upgraded to PyTorch 2.7 with CUDA 12.9 (performance improvements)
- **📦 Environment**: Conda-first strategy with 60% faster setup times

### Enhanced Artifacts Management

- **🗂️ Organized Structure**: Comprehensive artifacts directory with experiments, shared resources,
  and production models
- **📊 Better Organization**: Clear separation between experiments, global reports, and production assets
- **🔄 Version Control**: Integrated versioning system for model and experiment tracking

### Performance & Optimization

- **RTX 3070 Ti Specific**: VRAM-optimized configurations for 8GB constraints
- **Environment Stability**: Resolved Windows compilation issues with streamlined dependencies
- **Faster Development**: Enhanced type checking with basedpyright and improved linting
- **Organized Outputs**: All experiment outputs now properly organized in artifacts/ structure

For complete migration notes, see [CHANGELOG.md](CHANGELOG.md#migration-notes).

## Quickstart

### Prerequisites

**System Dependencies Required:**

- Git, Conda/Miniconda
- Optional: CUDA Toolkit (for GPU acceleration)
- **Note**: Graphviz no longer required - architecture visualization now uses matplotlib

For detailed installation instructions, see System Dependencies (see
`docs/guides/developer-guides/development/legacy/SYSTEM_DEPENDENCIES.md`).

**Quick verification:**

```bash
# Verify all system dependencies
python scripts/utils/verify_system_dependencies.py
```

### 1. Environment Setup

```bash
# Create and activate conda environment with PyTorch 2.7
conda env create -f environment.yml
conda activate crackseg

# Install the module in editable mode for development
pip install -e . --no-deps

# Copy environment template and configure
cp .env.example .env  # Edit with your specific settings
```

**System Requirements:**

- **Python**: 3.12+ (required for modern type annotations)
- **PyTorch**: 2.7 with CUDA 12.9 support
- **GPU**: RTX 3070 Ti (8GB VRAM) or compatible
- **RAM**: 16GB+ recommended for training

**Migration from v0.1.0**: If upgrading from an existing installation, recreate the environment:

```bash
conda env remove -n crackseg
conda env create -f environment.yml
conda activate crackseg
pip install -e . --no-deps
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

# Train with modern Swin Transformer encoder (recommended)
python run.py model=architectures/swin_unet data.batch_size=4

# Advanced configuration for 8GB VRAM (RTX 3070 Ti optimized)
python run.py data.batch_size=4 \
              training.mixed_precision=true \
              training.gradient_accumulation_steps=4 \
              model=architectures/swin_unet
```

#### Using the Interactive GUI

For a user-friendly experience, launch the Streamlit-based GUI:

```bash
conda activate crackseg && streamlit run gui/app.py
```

The GUI provides:

- **Visual Configuration**: Browse and edit YAML configurations with real-time validation
- **Training Monitoring**: Live logs and metric charts during training
- **Results Visualization**: Model predictions and performance analysis
- **Architecture Viewer**: Interactive model architecture diagrams (matplotlib-based)

Refer to the [tutorials in `docs/tutorials/`](docs/tutorials/) for a detailed walkthrough.

### 4. Evaluate a Model

```bash
# Evaluate using the evaluation module
python -m src.evaluation \
    model.checkpoint_path=artifacts/experiments/{timestamp}-{config}/checkpoints/best_model.pth

# Or use the wrapper script
python src/evaluate.py
```

For detailed workflow and advanced configuration options, see **Training Workflow Guide** at
`docs/guides/operational-guides/workflows/legacy/WORKFLOW_TRAINING.md`.

## Project Structure

For a comprehensive view of the project organization, see [**Project Structure Guide**](.cursor/rules/project-structure.mdc).

### Artifacts Organization

The project uses a comprehensive artifacts management system:

```bash
artifacts/
├── experiments/          # Individual experiment outputs (timestamped)
├── shared/              # Shared resources and utilities
├── global/              # Global reports, visualizations, and analysis
├── production/          # Production-ready models and configurations
├── archive/             # Archived experiments and historical data
└── versioning/          # Version control and model registry
```

### Core Directories

- **`src/`** — Core application code (models, data, training, evaluation)
- **`configs/`** — Hydra YAML configurations (modular and composable)
- **`tests/`** — Comprehensive test suite (66% coverage, 866 tests)
- **`docs/`** — Documentation, guides, and organized reports
- **`artifacts/`** — Comprehensive experiment outputs, models, and analysis (git ignored)
- **`scripts/`** — Utility scripts and experimental code
- **`.taskmaster/`** — Task Master project management files

### Key Components

- **Model Architectures**: `src/model/architectures/` (U-Net, SwinUNet variants)
- **Data Pipeline**: `src/data/` (datasets, transforms, loaders)
- **Training System**: `src/training/` (losses, optimizers, schedulers)
- **Evaluation Suite**: `src/evaluation/` (metrics, visualization, reporting)
- **Configuration**: `configs/` (model, training, data configurations)
- **Artifacts Management**: `artifacts/` (experiments, shared resources, production models)

> **Note:** Scripts in `scripts/` are for experimentation and utilities only. Do not import them in
> core modules. Clean up temporary files like `__pycache__` regularly.

## Configuration System

The project uses **Hydra** for hierarchical configuration management with modern best practices:

### Configuration Structure

- **Model Architectures**: `configs/model/architectures/` (U-Net, SwinUNet, etc.)
- **Training Components**: `configs/training/` (losses, metrics, schedulers)
- **Data Processing**: `configs/data/` (dataloaders, transforms using albumentations)
- **Evaluation**: `configs/evaluation/` (metrics, thresholds)

### Usage Examples

```bash
# Use modern Swin Transformer architecture
python run.py model=architectures/swin_unet

# Modern image transforms (albumentations-based)
python run.py data.transforms=advanced_augmentation

# Combine multiple overrides for production training
python run.py model=architectures/swin_unet \
              training.loss=combined_focal_dice \
              data.batch_size=4 \
              training.mixed_precision=true \
              experiment_name="swin_production_v2"
```

All configurations are composable and can be overridden via CLI or configuration files.

## Modern Dependencies & Architecture

### Current Technology Stack (v0.2.0)

**Core ML Framework:**

- **PyTorch 2.7**: Latest stable with performance improvements
- **CUDA 12.9**: Optimized for RTX 3070 Ti and newer GPUs
- **TIMM**: Modern pre-trained models (replaces torchvision.models)
- **Albumentations**: Advanced image augmentations (replaces torchvision.transforms)

**Visualization & Analysis:**

- **Matplotlib**: Primary for architecture diagrams and training plots
- **TensorBoard**: Experiment tracking and metric visualization
- **Streamlit**: Interactive GUI for configuration and monitoring

**Development Quality:**

- **basedpyright**: Enhanced static type checking
- **Black**: Code formatting with Python 3.12+ support
- **Ruff**: Fast Python linting and error detection

### Architecture Highlights

**Encoder Options:**

- ResNet variants (via TIMM)
- Swin Transformer (recommended for crack segmentation)
- EfficientNet family
- Vision Transformer (ViT) variants

**Decoder Architectures:**

- U-Net style with skip connections
- FPN (Feature Pyramid Network)
- DeepLabV3+ variants

**Loss Functions:**

- Dice Loss (class imbalance handling)
- Focal Loss (hard negative mining)
- Combined losses with configurable weights
- Boundary-aware losses for edge preservation

## Code Quality & Development Standards

This project maintains strict code quality standards optimized for modern Python:

### Quality Tools & Gates

```bash
# All commands require conda environment activation
conda activate crackseg && black .              # Code formatting
conda activate crackseg && ruff . --fix         # Linting with auto-fix
conda activate crackseg && basedpyright .       # Static type checking
conda activate crackseg && pytest tests/ --cov=src  # Testing with coverage
```

### Quality Requirements

- **Type Annotations**: Required for all functions using Python 3.12+ features
- **Test Coverage**: Target >80% for new code, current 66% overall
- **Code Formatting**: Must pass Black and Ruff checks
- **Documentation**: English docstrings required for all public APIs

### Modern Python 3.12+ Features

```python
# Modern type annotations (no typing imports needed)
def process_batch(images: list[torch.Tensor]) -> dict[str, float]:
    results: dict[str, float] = {}
    return results

# Type aliases for clarity
type BatchDict = dict[str, torch.Tensor]
type MetricDict = dict[str, float]
```

Configuration files:

- **Centralized config**: `configs/linting/config.yaml`
- **Tool configuration**: `pyproject.toml`
- **Type checking**: `pyrightconfig.json`

For detailed code quality guidelines, see
[**Coding Standards**](.cursor/rules/coding-standards.mdc). See also
[CONTRIBUTING](CONTRIBUTING.md) and [CHANGELOG](CHANGELOG.md).

## Training & Evaluation Workflow

### Modern Training Process

1. **Environment Setup**: Conda environment with PyTorch 2.7 stack
2. **Architecture Selection**: Choose from TIMM-based encoders + custom decoders
3. **Data Processing**: Albumentations-based augmentation pipeline
4. **Training Execution**: Mixed precision training with gradient accumulation
5. **Monitoring**: TensorBoard integration with real-time metrics
6. **Model Analysis**: Matplotlib-based architecture visualization

### Experiment Tracking

Training outputs are organized by timestamp and configuration:

```bash
artifacts/experiments/
└── {timestamp}-{config_name}/
    ├── checkpoints/    # Model checkpoints (.pth files)
    ├── logs/          # TensorBoard training logs
    ├── metrics/       # Training/validation metrics (CSV/JSON)
    ├── visualizations/ # Architecture diagrams (matplotlib)
    └── results/       # Prediction outputs and analysis
```

### Performance Optimization (RTX 3070 Ti)

```bash
# Optimized training for 8GB VRAM
python run.py data.batch_size=4 \
              training.mixed_precision=true \
              training.gradient_accumulation_steps=4 \
              training.dataloader.num_workers=4
```

For complete workflow details, see **Training Workflow Guide** at
`docs/guides/operational-guides/workflows/legacy/WORKFLOW_TRAINING.md`.

## Testing Framework

Comprehensive test suite ensuring code reliability with modern practices:

### Test Organization

- **Unit Tests**: `tests/unit/` - Individual component testing
- **Integration Tests**: `tests/integration/` - Module interaction testing
- **End-to-End Tests**: `tests/integration/end_to_end/` - Complete pipeline testing

### Running Tests

```bash
# Run all tests with coverage (requires conda activation)
conda activate crackseg && pytest tests/ --cov=src --cov-report=html

# Run specific test categories
conda activate crackseg && pytest tests/unit/          # Unit tests only
conda activate crackseg && pytest tests/integration/   # Integration tests only

# Run tests with detailed output
conda activate crackseg && pytest tests/ -v --cov=src --cov-report=term-missing
```

### Test Coverage & Metrics

- **Current Coverage**: 66% (5,333/8,065 lines)
- **Test Count**: 866 implemented tests
- **Coverage Reports**: Available in `htmlcov/` after running with `--cov-report=html`
- **Quality Gates**: All tests must pass before deployment

For testing guidelines and details, see [`tests/README.md`](tests/README.md).

## Task Management Integration

This project integrates with **Task Master** for structured development:

### Key Features

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
- **`PYTHONPATH`**: Automatically configured by `run.py` (after module installation)

## Migration Guide from v0.1.0

### Breaking Changes

- **Python 3.12+**: Now required for modern type annotations
- **TorchVision Removal**: Replace with TIMM + Albumentations
- **Graphviz Optional**: Architecture visualization uses matplotlib by default

### Code Updates Required

```python
# Old (v0.1.0)
import torchvision.models as models
import torchvision.transforms as transforms

model = models.resnet50(pretrained=True)
transform = transforms.Compose([...])

# New (v0.2.0)
import timm
import albumentations as A

model = timm.create_model('resnet50', pretrained=True)
transform = A.Compose([...])
```

### Environment Updates

```bash
# Remove old environment
conda env remove -n crackseg

# Create new v0.2.0 environment
conda env create -f environment.yml
conda activate crackseg
```

For complete migration details, see [CHANGELOG](CHANGELOG.md#migration-notes).

## Contributing

### Development Workflow

1. **Check Current Tasks**: Use `task-master next` to see priority work
2. **Environment Setup**: Ensure conda environment is active
3. **Create Feature Branch**: Follow git workflow standards
4. **Implement Changes**: Follow modern code quality standards
5. **Run Quality Checks**: Ensure all quality gates pass
6. **Write/Update Tests**: Maintain or improve test coverage
7. **Update Documentation**: Keep docs synchronized with changes
8. **Submit for Review**: Create PR with comprehensive description

### Contribution Guidelines

- **Follow [Coding Standards](.cursor/rules/coding-standards.mdc)**: Python 3.12+ type annotations,
  formatting, linting
- **Add Comprehensive Tests**: Unit and integration tests for new features
- **Update Documentation**: Keep README, guides, and docstrings current
- **Use Task Master**: Track work using the integrated task management system

For detailed contribution guidelines, see [`docs/guides/CONTRIBUTING.md`](docs/guides/CONTRIBUTING.md).

## System Requirements

- **Python 3.12+** (required for modern type annotations and features)
- **CUDA-capable GPU** (recommended for training)
- **8GB+ RAM** (system memory)
- **8GB+ VRAM** (GPU memory for training, optimized for RTX 3070 Ti)
- **conda/mamba** (for environment management)

All dependencies and tools are configured and tested for **Python 3.12**. Please ensure your
environment matches this version for full compatibility.

## Documentation & Resources

### User & Developer Guides

- **Training Workflow Guide**: see
  `docs/guides/operational-guides/workflows/legacy/WORKFLOW_TRAINING.md` - Complete training process
- [**Technical Architecture**](docs/guides/TECHNICAL_ARCHITECTURE.md) - System design overview
- [**System Dependencies**](docs/guides/SYSTEM_DEPENDENCIES.md) - Installation and setup
- [**Architectural Decisions**](docs/guides/architectural_decisions.md) - ADR-001 and design rationale
- [**Troubleshooting Guide**](docs/guides/TROUBLESHOOTING.md) - Common issues and solutions
- [**Tutorials**](docs/tutorials/) - Step-by-step learning materials

### Reports & Analytics

All project reports, analysis, and documentation are organized in `docs/reports/`:

- **`testing/`** — Test coverage reports and improvement plans
- **`coverage/`** — Code coverage analysis and gap reports
- **`tasks/`** — Task completion summaries and complexity analysis
- **`models/`** — Model architecture analysis and import catalogs
- **`project/`** — Project-level plans and verification reports

Current metrics:

- 📊 **Test Coverage**: 66% global (target >80% for new code)
- 🧪 **Tests Implemented**: 866 tests across unit and integration suites
- 🏗️ **Architecture**: Modular design with factory patterns
- 🔧 **Code Quality**: 100% type coverage, Black/Ruff compliance

## Project Status

**Version 0.2.0** - Production Ready (July 2025)

All major roadmap tasks and subtasks are completed. The system features:

- ✅ **Modern Technology Stack**: PyTorch 2.7 + CUDA 12.9
- ✅ **Optimized Dependencies**: 60% faster environment setup
- ✅ **Enhanced Performance**: RTX 3070 Ti specific optimizations
- ✅ **Improved Stability**: Resolved cross-platform compatibility issues
- ✅ **Advanced Testing**: E2E testing infrastructure with Docker/Selenium Grid
- ✅ **Comprehensive Documentation**: Complete guides and architectural decisions

The system is open for advanced contributions and production deployment.

## License

MIT License. See [`LICENSE`](LICENSE) for details.

---

## Tips & Best Practices

- 🚀 **Use `run.py`** as the main entry point for training
- ⚡ **Leverage modern dependencies**: TIMM for models, Albumentations for transforms
- ⚙️ **Leverage Hydra configs** for all parameter management
- 🧪 **Always activate conda environment**: `conda activate crackseg &&` before any commands
- 📊 **Monitor experiments** using TensorBoard and organized outputs in `artifacts/`
- 📚 **Check [`docs/reports/`](docs/reports/)** for latest project analysis
- 🔧 **Use Task Master** for structured development workflow
- 🎯 **Follow Python 3.12+ type annotations** for better code reliability
- 📈 **Maintain test coverage** when adding new features
- 💡 **Architecture visualization**: Use `render_unet_architecture_diagram()` with matplotlib backend
- 🗂️ **Organized outputs**: All experiment results are saved in `artifacts/experiments/` with clear
  structure
