# Training Workflow Guide

This document provides a step-by-step guide to set up and run training for pavement
crack segmentation models using our modular framework.

**For professional development**, consult our standards in the `.cursor/rules/` directory,
which complement this technical guide.

## Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Running Training](#running-training)
- [Model Evaluation](#model-evaluation)
- [Quality Standards](#quality-standards)
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
python --version

# Check quality tools (mandatory)
black --version
ruff --version
basedpyright --version
pytest --version
```

### 2. Data Structure

Place your data in the appropriate structure as outlined in the project's main `README.md`.

### 3. Environment Variables

```bash
# Copy template and configure
cp .env.example .env
# Edit .env as needed
```

### 4. Initial Verification

```bash
# Ensure code meets professional standards (mandatory before training)
black .
ruff . --fix
basedpyright .
pytest tests/ --cov=src --cov-report=term-missing
```

## Project Structure

The modular project structure allows flexibility in component selection. For detailed
organizational information, refer to the project's development standards.

### Configuration Components

- **Architectures**: `configs/model/architectures/`
- **Encoders**: `configs/model/encoder/`
- **Decoders**: `configs/model/decoder/`
- **Bottlenecks**: `configs/model/bottleneck/`
- **Loss Functions**: `configs/training/loss/`
- **Metrics**: `configs/training/metric/`
- **LR Schedulers**: `configs/training/lr_scheduler/`

For loss function details, see [Loss Registry Guide](loss_registry_usage.md).

## Configuration

### Main Configuration

The main configuration file is `configs/config.yaml`. You can override any
configuration directly from the command line using Hydra.

### Configuration Examples

1. **Basic training with default U-Net**:

    ```bash
    python run.py
    ```

2. **Switch to SwinUNet architecture**:

    ```bash
    python run.py model=architectures/unet_swin data.batch_size=4
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

## Running Training

### Basic Training

To start training with the default configuration:

```bash
python run.py
```

### Training with Reproducibility Standards

To follow our ML research standards for reproducibility:

```bash
# Reproducible training with fixed seed
python run.py random_seed=42 \
              experiment.name="baseline_reproducible"

# Monitor GPU memory usage during training
python run.py training.verbose=true \
              data.batch_size=4 \
              training.use_amp=true
```

## Model Evaluation

After training, the model is evaluated on the test set.

### Running Evaluation

```bash
python -m src.evaluation.evaluate --checkpoint_path ... --config_path ...
```

### Evaluation Outputs

Evaluation results are saved in the experiment output directory.

## Quality Standards

All code and experiments must adhere to our strict quality standards. Refer to the
rules in the `.cursor/rules/` directory, which cover coding preferences, testing,
Git, and ML standards.

## Troubleshooting

### Common Issues

- **`CUDA out of memory`**: Reduce `data.batch_size`, enable `training.use_amp=true`,
  or use `training.gradient_accumulation_steps`.
- **`ModuleNotFoundError`**: Ensure you have activated the `torch` conda environment.
- **`basedpyright` errors**: Check for complete and correct type annotations.

### Getting Help

- For technical issues, consult our development guides and rule system.
- For bugs, open an issue in the repository with detailed logs.

## Integration with Professional Development

This training workflow is part of a larger professional development process.

### 📚 Essential Documentation

- **Task Management**: Refer to the project's Task Master guide.
- **Loss Registry**: See [Loss Registry Guide](loss_registry_usage.md).
- **Configuration Storage**: Review the specifications for configuration management.

### 🛠️ Key Principles

- **Evidence-Based**: All results must be supported by logs and metrics.
- **Reproducible**: Experiments must be repeatable.
- **Modular**: Components should be designed for reuse and independent testing.

---

**This guide ensures that all training activities align with our professional
standards for creating a state-of-the-art crack segmentation system.**
