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
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before you start, make sure you have the following:

### 1. Environment and Tools

```bash
# Set up conda environment
conda env create -f environment.yml
conda activate torch

# Check quality tools (mandatory)
black --version
ruff --version
basedpyright --version
```

### 2. Data

Place your data in the appropriate structure:

```txt
data/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### 3. Environment Variables

```bash
cp .env.example .env
# Edit .env as needed
```

### 4. Initial Verification

```bash
# Ensure code meets professional standards
black .
ruff . --fix
basedpyright .

# Run tests
pytest tests/ --cov=src
```

## Project Structure

The modular project structure allows flexibility in component selection. For organizational details, see our [contribution guide](CONTRIBUTING.md).

- **Architectures**: `configs/model/architectures/`
- **Encoders**: `configs/model/encoder/`
- **Decoders**: `configs/model/decoder/`
- **Loss Functions**: `configs/training/loss/` (see [Loss Registry Guide](loss_registry_usage.md))
- **Metrics**: `configs/training/metric/`
- **LR Schedulers**: `configs/training/lr_scheduler/`

All these components are combined in the main configuration.

## Configuration

### Main Configuration

The main configuration file is `configs/config.yaml`, which includes:

```yaml
defaults:
  - data: dataloader/default
  - model: architectures/unet
  - training: default
  - evaluation: default
  - _self_
```

You can override any configuration directly from the command line.

### Configuration Examples

1. **Basic training with U-Net**:

   ```bash
   python run.py model=architectures/unet
   ```

2. **Switch to SwinUNet architecture (Research)**:

   ```bash
   python run.py model=architectures/swin_unet \
                 model.encoder.embed_dim=96 \
                 data.batch_size=4  # Optimized for RTX 3070 Ti
   ```

3. **Change to combined loss function**:

   ```bash
   python run.py training.loss=dice_bce
   ```

4. **Configuration optimized for 8GB VRAM**:

   ```bash
   python run.py data.batch_size=4 \
                 training.mixed_precision=true \
                 training.gradient_accumulation_steps=4
   ```

5. **Full research configuration**:

   ```bash
   python run.py model=architectures/swin_unet \
                 model.encoder.embed_dim=96 \
                 data.batch_size=4 \
                 training.optimizer.lr=0.0005 \
                 training.loss=dice_focal \
                 training.epochs=150 \
                 training.mixed_precision=true \
                 experiment_name="swin_unet_v1"
   ```

## Running Training

### Basic Training

To start training with the default configuration:

```bash
python run.py
```

### Training with ML Standards

Following our [ML standards](../../.cursor/rules/ml-research-standards.mdc):

```bash
# Set seed for reproducibility
python run.py seed=42 \
             training.deterministic=true \
             experiment_name="baseline_experiment" \

# Monitor VRAM usage during training
python run.py training.log_gpu_memory=true \
             data.batch_size=4 \
             training.mixed_precision=true
```

### Monitoring

Training generates logs and metrics in:

```txt
outputs/
└── experiments/
    └── {timestamp}-{config_name}/
        ├── checkpoints/  # Saved models
        ├── logs/         # TensorBoard logs
        ├── metrics/      # Metrics CSV
        └── results/      # Validation predictions
```

To visualize training metrics:

```bash
tensorboard --logdir outputs/experiments/
```

## Model Evaluation

To evaluate a trained model:

```bash
python -m src.evaluation \
    model.checkpoint_path=outputs/experiments/{timestamp}-{config_name}/checkpoints/best_model.pth \
    evaluation.save_predictions=True
```

This will generate metrics and prediction images in the results directory.

### Evaluation with Standard Metrics

```bash
# Full evaluation with SOTA metrics
python -m src.evaluation \
    model.checkpoint_path=path/to/checkpoint.pth \
    evaluation.metrics=["iou", "f1", "precision", "recall"] \
    evaluation.save_confusion_matrix=True
```

## Quality Standards

### Pre-training

Before each training run, check code quality:

```bash
# Mandatory verification (based on coding-preferences.mdc)
black .                    # Auto-formatting
ruff . --fix              # Linting and autofix
basedpyright .            # Strict type checking
pytest tests/ --cov=src   # Tests with coverage
```

### During Development

Follow our [workflow rules](../../.cursor/rules/workflow-preferences.mdc):

- **Three-option analysis** for major technical decisions
- **Complete type hints** in all ML code
- **Comprehensive documentation** of experiments and results
- **Robust testing** of model components

### For ML Research

See [ML standards](../../.cursor/rules/ml-research-standards.mdc):

- **Reproducibility**: Deterministic seeds, documented configurations
- **VRAM management**: Optimized for RTX 3070 Ti (8GB)
- **Experiment tracking**: Detailed logs, SOTA comparisons

## Troubleshooting

### Common Issues

1. **CUDA Memory Error**: Reduce batch size or image resolution

   ```bash
   python run.py data.batch_size=2 \
                data.image_size=[384,384] \
                training.mixed_precision=true
   ```

2. **Exploding Gradients**: Adjust learning rate or enable gradient clipping

   ```bash
   python run.py training.optimizer.lr=0.0001 \
                training.clip_grad_norm=1.0
   ```

3. **Stagnant Metrics**: Try different loss functions or data augmentations

   ```bash
   python run.py training.loss=dice_focal \
                data.augmentation=strong
   ```

4. **Typing Errors**: Run basedpyright to identify issues

   ```bash
   basedpyright src/ --strict
   ```

### Performance Optimization

To improve training performance:

1. **Data Preloading**: Increase `num_workers` for the dataloader

   ```bash
   python run.py data.num_workers=4
   ```

2. **Mixed Precision**: Enable mixed precision training

   ```bash
   python run.py training.mixed_precision=true
   ```

3. **Gradient Accumulation**: For larger effective batch sizes

   ```bash
   python run.py training.gradient_accumulation_steps=4 \
                data.batch_size=2  # Effective batch: 2*4=8
   ```

### Code Debugging

If you encounter quality errors:

```bash
# Step-by-step debugging
black --check .           # Check formatting
ruff check .              # Check linting
basedpyright --stats .    # Typing check with stats
pytest tests/ -v --tb=short  # Tests with detailed output
```

## Integration with Professional Development

This training workflow integrates with our professional standards:

- **[Code Standards](../../.cursor/rules/coding-preferences.mdc)**: Mandatory technical quality
- **[Testing](../../.cursor/rules/testing-standards.mdc)**: ML testing strategies
- **[Git](../../.cursor/rules/git-standards.mdc)**: Descriptive commits for experiments
- **[ML Research](../../.cursor/rules/ml-research-standards.mdc)**: Reproducibility and optimization

---

For any additional questions:

- **Technical standards**: See `.cursor/rules/`
- **Specific issues**: Open an issue in the repository
- **Related guides**: See `docs/guides/`

**Note**: This guide is updated periodically with new features and optimizations, staying in sync with our professional standards.
