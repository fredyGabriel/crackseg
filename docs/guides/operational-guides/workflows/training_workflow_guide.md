# Training Workflow Guide

## Overview

This guide documents the complete training workflow for the CrackSeg project, from environment setup
to experiment execution and monitoring.

## 🚀 Quick Start Workflow

### 1. Environment Setup

```bash
# Activate conda environment
conda activate crackseg

# Install package in development mode
pip install -e . --no-deps

# Verify installation
python -c "import crackseg; print('✅ Success')"
```

### 2. Quality Gates Verification

```bash
# Run quality gates
black .
python -m ruff . --fix
basedpyright .
```

### 3. Dataset Preparation

**For Crack500 (360x360)**:

- Dataset available in `data/crack500/`
- No preprocessing required

**For PY-CrackDB (320x320)**:

```bash
# Process PY-CrackDB dataset
python scripts/data_processing/image_processing/process_py_crackdb_example.py
```

### 4. Experiment Execution

**Verified Experiments**:

```bash
# SwinV2 360x360 (Crack500)
python run.py --config-name=experiments/swinv2_hybrid/swinv2_360x360_corrected +training.epochs=2

# SwinV2 320x320 (PY-CrackDB)
python run.py --config-name=experiments/swinv2_hybrid/swinv2_320x320_py_crackdb +training.epochs=2
```

## 📊 Workflow Stages

### Stage 1: Pre-Execution

#### 1.1 Environment Verification

**Required Checks**:

- ✅ Conda environment activated
- ✅ Package installed in development mode
- ✅ Import verification successful
- ✅ Quality gates passing

#### 1.2 Dataset Validation

**Crack500 Dataset**:

```bash
# Verify dataset structure
ls data/crack500/
# Expected: images/ and masks/ directories
```

**PY-CrackDB Dataset**:

```bash
# Verify processed dataset
ls data/PY-CrackBD/
# Expected: 369 processed images and masks
```

#### 1.3 Configuration Validation

**Check Configuration Files**:

```bash
# Verify config exists
ls configs/experiments/swinv2_hybrid/
# Expected: swinv2_360x360_corrected.yaml, swinv2_320x320_py_crackdb.yaml
```

### Stage 2: Execution

#### 2.1 Training Execution

**Standard Command Format**:

```bash
python run.py --config-name=experiments/swinv2_hybrid/[config_name] +training.epochs=[epochs]
```

**Example Executions**:

```bash
# Quick verification (2 epochs)
python run.py --config-name=experiments/swinv2_hybrid/swinv2_360x360_corrected +training.epochs=2

# Full training (50 epochs)
python run.py --config-name=experiments/swinv2_hybrid/swinv2_320x320_py_crackdb +training.epochs=50
```

#### 2.2 Monitoring During Training

**TensorBoard Monitoring**:

```bash
# Start TensorBoard (in separate terminal)
tensorboard --logdir artifacts/experiments/tensorboard
```

**Expected Output Pattern**:

```bash
Epoch 1: val_loss: X.XXXX | val_iou: X.XXXX | val_dice: X.XXXX
Epoch 2: val_loss: X.XXXX | val_iou: X.XXXX | val_dice: X.XXXX
```

### Stage 3: Post-Execution

#### 3.1 Results Verification

**Check Generated Artifacts**:

```bash
# Verify experiment outputs
ls artifacts/experiments/
# Expected: experiment directories with logs, checkpoints, config.yaml
```

**Expected Directory Structure**:

```txt
artifacts/experiments/
├── swinv2_360x360_corrected/
│   ├── logs/
│   ├── checkpoints/
│   └── config.yaml
└── py_crackdb_swinv2/
    ├── logs/
    ├── checkpoints/
    └── config.yaml
```

#### 3.2 Performance Analysis

**Key Metrics to Monitor**:

- **Loss**: Should decrease over epochs
- **IoU**: Should increase over epochs
- **Dice**: Should increase over epochs
- **Memory Usage**: Should remain stable

## 🔧 Advanced Workflow

### Custom Experiment Creation

#### 1. Configuration Adaptation

**Base Template**: `configs/experiments/swinv2_hybrid/swinv2_360x360_corrected.yaml`

**Adaptation Process**:

```yaml
# Modify these parameters for new experiments
experiment:
  name: my_custom_experiment

model:
  encoder_cfg:
    img_size: 320  # Change image size
    target_img_size: 320

data:
  data_root: data/my_dataset/  # Change dataset path
  root_dir: data/my_dataset/
```

#### 2. Validation Testing

**Quick Validation**:

```bash
# Test with 2 epochs
python run.py --config-name=experiments/swinv2_hybrid/my_custom_experiment +training.epochs=2
```

**Expected Success Indicators**:

- ✅ No configuration errors
- ✅ Training starts successfully
- ✅ Metrics are logged
- ✅ Checkpoints are saved

### Troubleshooting Workflow

#### Common Issues and Solutions

**1. Configuration Not Found**:

```bash
# Problem: Key 'model' is not in struct
# Solution: Use the fixed run.py with nested config handling
python run.py --config-name=experiments/swinv2_hybrid/swinv2_360x360_corrected +training.epochs=2
```

**2. Dataset Not Found**:

```bash
# Problem: Dataset directory not found
# Solution: Verify dataset path in config
ls data/PY-CrackBD/  # Should exist
```

**3. Memory Issues**:

```bash
# Problem: CUDA out of memory
# Solution: Reduce batch size in config
# Or use CPU training
python run.py --config-name=experiments/swinv2_hybrid/swinv2_360x360_corrected training.device=cpu
```

**4. Import Errors**:

```bash
# Problem: No module named 'crackseg'
# Solution: Reinstall package
pip install -e . --no-deps
python -c "import crackseg; print('✅ Success')"
```

## 📈 Performance Optimization

### Memory Management

**RTX 3070 Ti Optimization**:

- **Batch Size**: 12 (optimal for 8GB VRAM)
- **Mixed Precision**: Enabled by default
- **Gradient Accumulation**: 2 steps
- **Memory Monitoring**: Continuous VRAM tracking

### Training Efficiency

**Recommended Settings**:

```yaml
training:
  batch_size: 12
  gradient_accumulation_steps: 2
  use_amp: true  # Mixed precision
  device: auto
```

### Quality Assurance

**Pre-Execution Checklist**:

- ✅ Environment activated
- ✅ Package installed
- ✅ Quality gates pass
- ✅ Dataset verified
- ✅ Configuration validated

**Post-Execution Checklist**:

- ✅ Training completed
- ✅ Metrics logged
- ✅ Checkpoints saved
- ✅ No errors in logs

## 🔄 Continuous Improvement

### Workflow Optimization

**Regular Updates**:

1. **Monitor performance metrics**
2. **Update configurations based on results**
3. **Optimize hyperparameters**
4. **Document successful patterns**

### Best Practices

**Configuration Management**:

- Use version control for configurations
- Document configuration changes
- Test configurations before full training

**Experiment Tracking**:

- Log all hyperparameters
- Save configuration files
- Track performance metrics
- Document unexpected results

## 📚 References

- **Successful Experiments**: `successful_experiments_guide.md`
- **Configuration Guide**: `../technical-specs/configuration_guide.md`
- **Troubleshooting**: `../user-guides/troubleshooting.md`
- **Installation**: `../user-guides/getting-started/installation.md`

---

**Last Updated**: December 2024
**Status**: Active - Verified with successful experiments
**Next Steps**: Optimize for production deployment
