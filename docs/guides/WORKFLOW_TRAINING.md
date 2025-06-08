# Training Workflow Guide

This document provides a step-by-step guide to set up and run training for pavement crack segmentation models using our modular framework.

**For professional development**, consult our standards in [.cursor/rules/](../../.cursor/rules/) which complement this technical guide.

## Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Running Training](#running-training)
- [Model Evaluation](#model-evaluation)
- [Quality Standards](#quality-standards)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Integration with Professional Development](#integration-with-professional-development)

## Prerequisites

Before you start, make sure you have the following:

### 1. Environment and Tools

```bash
# Set up conda environment
conda env create -f environment.yml
conda activate torch

# Verify Python version (required: 3.12+)
python --version  # Should show Python 3.12.9

# Check quality tools (mandatory)
black --version    # Code formatting
ruff --version     # Linting and style checking
basedpyright --version  # Type checking (strict)
pytest --version   # Testing framework
```

### 2. Data Structure

Place your data in the appropriate structure:

```txt
data/
├── train/
│   ├── images/     # Training images (.jpg, .png)
│   └── masks/      # Binary masks (.png)
├── val/
│   ├── images/     # Validation images
│   └── masks/      # Validation masks
└── test/
    ├── images/     # Test images
    └── masks/      # Test masks
```

**Current Limits**: Training pipeline is configured with max samples for testing:

- Max train samples: 8
- Max validation samples: 4
- Max test samples: 4

### 3. Environment Variables

```bash
# Copy template and configure
cp .env.example .env
# Edit .env as needed (ANTHROPIC_API_KEY for Task Master integration)
```

### 4. Initial Verification

```bash
# Ensure code meets professional standards (mandatory before training)
black .                    # Auto-formatting (PEP 8 compliance)
ruff . --fix              # Linting and autofix (code quality)
basedpyright .            # Strict type checking (zero errors required)

# Run comprehensive test suite
pytest tests/ --cov=src --cov-report=term-missing
```

## Project Structure

The modular project structure allows flexibility in component selection. For detailed organizational information, see our [Project Structure Guide](../../.cursor/rules/project-structure.mdc).

### Configuration Components

- **Architectures**: `configs/model/architectures/` (U-Net variants, SwinUNet)
- **Encoders**: `configs/model/encoder/` (CNN, Swin Transformer)
- **Decoders**: `configs/model/decoder/` (CNN decoders)
- **Bottlenecks**: `configs/model/bottleneck/` (ASPP, ConvLSTM, default)
- **Loss Functions**: `configs/training/loss/` (BCE, Dice, Focal, Combined)
- **Metrics**: `configs/training/metric/` (IoU, F1, Precision, Recall)
- **LR Schedulers**: `configs/training/lr_scheduler/` (StepLR, CosineAnnealingLR, ReduceLROnPlateau)

All components use factory patterns and are registered for dynamic instantiation.

For loss function details, see [Loss Registry Guide](loss_registry_usage.md).

## Configuration

### Main Configuration

The main configuration file is `configs/config.yaml`, which includes:

```yaml
defaults:
  - data: default
  - model: default
  - training: default
  - evaluation: default
  - _self_
```

You can override any configuration directly from the command line using Hydra.

### Available Configurations

**Model Architectures**:

- `cnn_convlstm_unet` - U-Net with ConvLSTM bottleneck
- `swinv2_hybrid` - Hybrid Swin Transformer V2 architecture
- `unet_aspp` - U-Net with ASPP bottleneck
- `unet_cnn` - Standard CNN-based U-Net
- `unet_swin` - U-Net with Swin Transformer encoder
- `unet_swin_base` - Base Swin U-Net configuration
- `unet_swin_transfer` - Swin U-Net with transfer learning

**Loss Functions**:

- `bce` - Binary Cross Entropy
- `dice` - Dice Loss
- `focal` - Focal Loss
- `bce_dice` - Combined BCE + Dice
- `combined` - Multi-loss combination

### Configuration Examples

1. **Basic training with default U-Net**:

   ```bash
   python run.py
   ```

2. **Switch to SwinUNet architecture**:

   ```bash
   python run.py model=architectures/unet_swin \
                 data.batch_size=4  # Optimized for 8GB VRAM
   ```

3. **Use combined loss function**:

   ```bash
   python run.py training.loss=bce_dice
   ```

4. **Configuration optimized for 8GB VRAM (RTX 3070 Ti)**:

   ```bash
   python run.py data.batch_size=4 \
                 training.use_amp=true \
                 training.gradient_accumulation_steps=4
   ```

5. **Full research configuration with advanced features**:

   ```bash
   python run.py model=architectures/swinv2_hybrid \
                 data.batch_size=4 \
                 training.optimizer.lr=0.0005 \
                 training.loss=combined \
                 training.epochs=150 \
                 training.use_amp=true \
                 experiment.name="swinv2_hybrid_v1" \
                 random_seed=42
   ```

6. **Development mode with reduced dataset**:

   ```bash
   python run.py training.epochs=2 \
                 data.dataloader.max_train_samples=8 \
                 data.dataloader.max_val_samples=4 \
                 training.log_interval_batches=1
   ```

## Running Training

### Basic Training

To start training with the default configuration:

```bash
python run.py
```

This will use:

- Default U-Net architecture (CNN encoder + decoder)
- BCEDice loss function
- Adam optimizer with lr=0.001
- 2 epochs (development mode)
- Automatic mixed precision (AMP)

### Training with Reproducibility Standards

Following our [ML Research Standards](../../.cursor/rules/ml-research-standards.mdc):

```bash
# Reproducible training with fixed seed
python run.py random_seed=42 \
             experiment.name="baseline_reproducible"

# Monitor GPU memory usage during training
python run.py training.verbose=true \
             data.batch_size=4 \
             training.use_amp=true \
             training.log_interval_batches=10
```

### Advanced Training Configuration

```bash
# Production training with full epochs
python run.py training.epochs=100 \
             training.early_stopping.patience=10 \
             training.save_freq=10 \
             training.checkpoint_dir=outputs/checkpoints \
             experiment.name="production_v1"

# Research mode with extensive logging
python run.py training.epochs=50 \
             training.verbose=true \
             training.progress_bar=true \
             evaluation.save_predictions=true \
             evaluation.num_batches_visualize=3 \
             experiment.name="research_detailed"
```

### Monitoring

Training generates comprehensive outputs in:

```txt
outputs/experiments/
└── {timestamp}-{experiment_name}/
    ├── checkpoints/     # Model checkpoints (.pth.tar files)
    │   ├── model_best.pth.tar    # Best model (lowest val_loss)
    │   └── checkpoint_epoch_N.pth.tar  # Periodic saves
    ├── logs/           # Training logs and metrics
    ├── metrics/        # Structured metrics (CSV/JSON)
    └── results/        # Validation predictions and visualizations
```

To visualize training progress:

```bash
# View training logs in real-time
tail -f outputs/experiments/{timestamp}-{experiment_name}/logs/training.log

# Access Hydra configuration
cat outputs/experiments/{timestamp}-{experiment_name}/.hydra/config.yaml
```

## Model Evaluation

### Basic Evaluation

To evaluate a trained model:

```bash
python -m src.evaluation \
    model.checkpoint_path=outputs/experiments/{timestamp}-{config_name}/checkpoints/model_best.pth.tar \
    evaluation.save_predictions=true
```

### Comprehensive Evaluation with All Metrics

```bash
python -m src.evaluation \
    model.checkpoint_path=outputs/experiments/{timestamp}-{config_name}/checkpoints/model_best.pth.tar \
    evaluation.metrics.iou.threshold=0.5 \
    evaluation.metrics.f1.threshold=0.5 \
    evaluation.save_predictions=true \
    evaluation.visualize_samples=10 \
    evaluation.save_dir=eval_outputs/detailed_analysis
```

### Evaluation Output Structure

Evaluation generates comprehensive results:

```txt
eval_outputs/
├── metrics/
│   ├── iou_scores.json        # IoU per sample
│   ├── f1_scores.json         # F1 per sample
│   ├── precision_scores.json  # Precision per sample
│   └── recall_scores.json     # Recall per sample
├── predictions/
│   ├── sample_001_pred.png    # Prediction masks
│   └── sample_001_overlay.png # Overlays with ground truth
└── summary/
    ├── evaluation_report.json # Aggregate metrics
    └── confusion_matrix.png   # Confusion matrix visualization
```

## Quality Standards

### Pre-training Quality Checks (Mandatory)

Before each training run, verify code quality:

```bash
# Complete quality verification pipeline
black .                          # Auto-formatting (PEP 8)
ruff . --fix                     # Linting and autofix
basedpyright .                   # Strict type checking (zero errors)
pytest tests/ --cov=src --cov-report=html  # Full test suite with coverage
```

**Current Quality Metrics**:

- **Test Coverage**: 66% (5,333/8,065 lines)
- **Tests Implemented**: 866 tests across unit and integration suites
- **Type Coverage**: 100% (all code strictly typed)
- **Code Style**: Black + Ruff compliant

### During Development

Follow our [workflow standards](../../.cursor/rules/workflow-preferences.mdc):

- **Three-option analysis** for major technical decisions
- **Complete type annotations** for all ML components
- **Comprehensive documentation** of experiments and configurations
- **Robust testing** of model components before integration

### ML Research Standards

See [ML Research Standards](../../.cursor/rules/ml-research-standards.mdc):

- **Reproducibility**: Deterministic seeds, version-controlled configurations
- **VRAM Management**: Optimized for RTX 3070 Ti (8GB VRAM)
- **Experiment Tracking**: Structured outputs, comparative analysis
- **Performance Monitoring**: Memory usage, training efficiency metrics

## Performance Optimization

### Memory Management

For RTX 3070 Ti (8GB VRAM) optimization:

```bash
# Optimized configuration for 8GB VRAM
python run.py data.batch_size=4 \
             training.use_amp=true \
             training.gradient_accumulation_steps=4 \
             data.dataloader.pin_memory=true \
             data.dataloader.prefetch_factor=2
```

### Data Loading Optimization

```bash
# Improve data loading performance
python run.py data.num_workers=4 \
             data.dataloader.pin_memory=true \
             data.dataloader.prefetch_factor=2 \
             data.in_memory_cache=false  # For large datasets
```

### Training Acceleration

```bash
# Advanced performance optimizations
python run.py training.use_amp=true \
             training.gradient_accumulation_steps=8 \
             data.batch_size=2 \
             training.optimizer.lr=0.0005 \
             data.image_size=[256,256]  # Reduced resolution for speed
```

## Troubleshooting

### Common Issues and Solutions

1. **CUDA Out of Memory**:

   ```bash
   # Reduce memory usage
   python run.py data.batch_size=2 \
                data.image_size=[256,256] \
                training.use_amp=true \
                training.gradient_accumulation_steps=8
   ```

2. **Training Instability (Exploding Gradients)**:

   ```bash
   # Stabilize training
   python run.py training.optimizer.lr=0.0001 \
                training.gradient_clip_norm=1.0 \
                training.optimizer.weight_decay=0.0001
   ```

3. **Poor Convergence**:

   ```bash
   # Try different loss function and scheduler
   python run.py training.loss=combined \
                training.lr_scheduler=cosine \
                training.optimizer.lr=0.001
   ```

4. **Type Checking Errors**:

   ```bash
   # Debug type issues
   basedpyright src/ --stats      # Show type coverage
   basedpyright src/ --verbose    # Detailed error messages
   ```

5. **Data Loading Issues**:

   ```bash
   # Debug data pipeline
   python -c "from src.data import CrackSegmentationDataset; print('Data module OK')"
   python run.py data.dataloader.num_workers=0  # Single-threaded debugging
   ```

### Performance Debugging

```bash
# Comprehensive debugging workflow
black --check .                  # Check formatting compliance
ruff check . --verbose          # Detailed linting report
basedpyright --stats .          # Type checking statistics
pytest tests/ -v --tb=short    # Tests with concise tracebacks
python run.py training.epochs=1 training.verbose=true  # Quick training test
```

### Configuration Debugging

```bash
# Validate configuration
python run.py --help            # Show all available options
python run.py --cfg job         # Print resolved configuration
python -c "import hydra; print('Hydra version:', hydra.__version__)"
```

## Integration with Professional Development

This training workflow integrates seamlessly with our professional standards:

### Quality Standards Integration

- **[Code Standards](../../.cursor/rules/coding-preferences.mdc)**: Mandatory type hints, Black formatting, Ruff linting
- **[Testing Standards](../../.cursor/rules/testing-standards.mdc)**: ML-specific testing strategies and coverage requirements
- **[Git Standards](../../.cursor/rules/git-standards.mdc)**: Descriptive commits for experiments and model versions
- **[ML Research Standards](../../.cursor/rules/ml-research-standards.mdc)**: Reproducibility protocols and VRAM optimization

### Task Master Integration

This workflow integrates with [Task Master](../../README-task-master.md) for structured development:

```bash
# Check current training-related tasks
task-master list --status=pending | grep -i train

# Update task progress
task-master set-status --id=12.2 --status=done

# Document training results in tasks
task-master update-subtask --id=training_task_id \
    --prompt="Training completed with 95% IoU on validation set"
```

### Development Workflow

1. **Check Quality**: Run quality checks before training
2. **Configure Experiment**: Use Hydra for reproducible configurations
3. **Monitor Training**: Track progress through structured outputs
4. **Evaluate Results**: Use comprehensive evaluation suite
5. **Document Findings**: Update tasks and commit results
6. **Iterate**: Apply learnings to next experiment cycle

---

## Additional Resources

- **Technical Standards**: See [.cursor/rules/](../../.cursor/rules/) for comprehensive guidelines
- **Project Structure**: [Project Structure Guide](../../.cursor/rules/project-structure.mdc)
- **Contribution Guidelines**: [Contributing Guide](CONTRIBUTING.md)
- **Loss Functions**: [Loss Registry Usage Guide](loss_registry_usage.md)

## System Requirements

- **Python 3.12+** (required for modern type annotations)
- **CUDA-capable GPU** (RTX 3070 Ti 8GB recommended)
- **8GB+ System RAM**
- **conda/mamba** for environment management
- **basedpyright** for strict type checking

---

**Note**: This guide is maintained as a living document, updated with each project iteration to reflect current best practices and optimizations. Always verify commands against the latest project state before production use.
