# Training Configuration

This directory contains YAML configuration files for training parameters in the pavement crack segmentation project.

## Configuration Files

### Core Training Configuration

- **`default.yaml`**: Main training configuration including:
  - Trainer class and device settings
  - Optimizer configuration (Adam with lr=0.001)
  - Learning rate scheduler (StepLR)
  - Mixed precision training (AMP)
  - Checkpoint and early stopping settings
  - Loss function configuration
  - Training hyperparameters (epochs, learning rate, weight decay)

### Specialized Configurations

- **`trainer.yaml`**: Alternative trainer configuration with different settings
- **`loss/`**: Loss function configurations (BCE, Dice, Focal, Combined)
- **`lr_scheduler/`**: Learning rate scheduler configurations
- **`metric/`**: Training and validation metrics configurations
- **`logging/`**: Logging and checkpoint configurations

## Key Parameters

### Training Hyperparameters

- **Epochs**: 2 (development mode) / 100+ (production)
- **Learning Rate**: 0.001 (Adam optimizer)
- **Weight Decay**: 0.0001
- **Batch Size**: Inherited from `data.batch_size`
- **Mixed Precision**: Enabled (`use_amp: true`)

### Optimization

- **Optimizer**: Adam with configurable learning rate
- **Scheduler**: StepLR (step_size=10, gamma=0.5)
- **Gradient Accumulation**: 1 step (configurable)
- **Gradient Clipping**: Available via configuration

### Model Persistence

- **Checkpoint Directory**: `outputs/checkpoints`
- **Save Frequency**: Configurable (0 = only best model)
- **Best Model Saving**: Enabled, monitors `val_loss`
- **Early Stopping**: Enabled (patience=2, min_delta=0.001)

### Loss Functions

- **Default**: BCEDiceLoss (combined BCE + Dice)
- **Available**: BCE, Dice, Focal, Combined losses
- **Weights**: Configurable loss component weights

## Available Loss Functions

### Individual Losses

- **`bce.yaml`**: Binary Cross Entropy loss
- **`dice.yaml`**: Dice loss for segmentation
- **`focal.yaml`**: Focal loss for class imbalance

### Combined Losses

- **`bce_dice.yaml`**: Combined BCE + Dice loss (default)
- **`combined.yaml`**: Multi-component loss combination

## Learning Rate Schedulers

- **`step_lr.yaml`**: Step decay scheduler (default)
- **`cosine.yaml`**: Cosine annealing scheduler
- **`reduce_on_plateau.yaml`**: Adaptive scheduler based on metrics

## Usage Examples

### Basic training with defaults

```bash
python run.py
```

### Override training parameters

```bash
python run.py training.epochs=100 \
             training.optimizer.lr=0.0005 \
             training.use_amp=true
```

### Change loss function

```bash
python run.py training.loss=focal
```

### Modify learning rate scheduler

```bash
python run.py training.lr_scheduler=cosine
```

### Production training configuration

```bash
python run.py training.epochs=150 \
             training.early_stopping.patience=15 \
             training.save_freq=10 \
             training.optimizer.lr=0.001 \
             training.use_amp=true
```

### Memory-optimized training

```bash
python run.py training.gradient_accumulation_steps=4 \
             training.use_amp=true \
             data.batch_size=4
```

## Configuration Structure

```yaml
# Example structure from default.yaml
training:
  _target_: src.training.trainer.Trainer
  device: cuda:0
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.001
  lr_scheduler:
    _target_: torch.optim.lr_scheduler.StepLR
    step_size: 10
    gamma: 0.5
  use_amp: true
  loss:
    _target_: src.training.losses.BCEDiceLoss
    bce_weight: 0.5
    dice_weight: 0.5
  epochs: 2
  # ... additional parameters
```

## Integration

Training configurations integrate with:

- **Trainer**: `src/training/trainer.py`
- **Loss Functions**: `src/training/losses/`
- **Metrics**: `src/training/metrics.py`
- **Factory**: `src/training/factory.py`
- **Main Pipeline**: `src/main.py`

## Best Practices

1. **Development vs Production**: Use `epochs=2` for testing, `epochs=100+` for production
2. **Memory Management**: Enable AMP and adjust batch size for available VRAM
3. **Reproducibility**: Set deterministic seeds and document configurations
4. **Monitoring**: Enable verbose logging and progress bars for long training runs
5. **Checkpointing**: Save models regularly and monitor best model based on validation loss

## Hardware-Specific Recommendations

### RTX 3070 Ti (8GB VRAM)

```yaml
training:
  use_amp: true
  gradient_accumulation_steps: 4
data:
  batch_size: 4
```

### RTX 4090 (24GB VRAM)

```yaml
training:
  use_amp: true
  gradient_accumulation_steps: 1
data:
  batch_size: 16
```

### CPU Training

```yaml
training:
  device: cpu
  use_amp: false
data:
  batch_size: 2
  num_workers: 2
```

## Troubleshooting

- **CUDA OOM**: Reduce `batch_size`, enable `use_amp`, increase `gradient_accumulation_steps`
- **Slow convergence**: Try different loss functions or learning rate schedules
- **Unstable training**: Enable gradient clipping or reduce learning rate
- **Poor validation**: Adjust early stopping patience or validation frequency
