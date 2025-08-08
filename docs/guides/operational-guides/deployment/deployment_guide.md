# Deployment Guide

## Overview

This guide covers the deployment process for the CrackSeg project, from development environment
setup to production deployment considerations.

## 🚀 Development Environment Deployment

### Prerequisites

**System Requirements**:

- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.12+
- **GPU**: RTX 3070 Ti (8GB VRAM) or equivalent
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ for datasets and models

### Environment Setup

#### 1. Conda Environment Creation

```bash
# Create conda environment
conda create -n crackseg python=3.12

# Activate environment
conda activate crackseg
```

#### 2. Package Installation

```bash
# Install in development mode
pip install -e . --no-deps

# Verify installation
python -c "import crackseg; print('✅ Success')"
```

#### 3. Quality Gates Verification

```bash
# Run quality gates
black .
python -m ruff . --fix
basedpyright .
```

### Dataset Deployment

#### Crack500 Dataset

**Location**: `data/crack500/`
**Status**: ✅ **Ready for use**

```bash
# Verify dataset structure
ls data/crack500/
# Expected: images/ and masks/ directories
```

#### PY-CrackDB Dataset

**Location**: `data/PY-CrackBD/`
**Status**: ✅ **Processed and ready**

```bash
# Process dataset (if not already done)
python scripts/data_processing/image_processing/process_py_crackdb_example.py

# Verify processed dataset
ls data/PY-CrackBD/
# Expected: 369 processed images and masks
```

## 🔧 Configuration Deployment

### Verified Configurations

#### 1. SwinV2 360x360 (Crack500)

**File**: `configs/experiments/swinv2_hybrid/swinv2_360x360_corrected.yaml`
**Status**: ✅ **Production ready**

**Deployment Command**:

```bash
python run.py --config-name=experiments/swinv2_hybrid/swinv2_360x360_corrected +training.epochs=2
```

#### 2. SwinV2 320x320 (PY-CrackDB)

**File**: `configs/experiments/swinv2_hybrid/swinv2_320x320_py_crackdb.yaml`
**Status**: ✅ **Recently verified**

**Deployment Command**:

```bash
python run.py --config-name=experiments/swinv2_hybrid/swinv2_320x320_py_crackdb +training.epochs=2
```

### Configuration Validation

**Pre-Deployment Checklist**:

- ✅ Configuration file exists
- ✅ Dataset paths are correct
- ✅ Model parameters are valid
- ✅ Training parameters are appropriate

**Validation Command**:

```bash
# Test configuration loading
python -c "
import hydra
from omegaconf import OmegaConf
cfg = hydra.compose(config_name='experiments/swinv2_hybrid/swinv2_360x360_corrected')
print('✅ Configuration loaded successfully')
"
```

## 📊 Monitoring Deployment

### TensorBoard Setup

**Start TensorBoard**:

```bash
# Start monitoring
tensorboard --logdir artifacts/experiments/tensorboard

# Access at: http://localhost:6006
```

### Logging Configuration

**Log Locations**:

- **Training Logs**: `artifacts/experiments/*/logs/`
- **Checkpoints**: `artifacts/experiments/*/checkpoints/`
- **Configurations**: `artifacts/experiments/*/config.yaml`

### Performance Monitoring

**Key Metrics to Track**:

- **GPU Memory**: Monitor VRAM usage
- **Training Loss**: Should decrease over time
- **Validation Metrics**: IoU, Dice, Precision, Recall
- **System Resources**: CPU, RAM usage

## 🔍 Troubleshooting Deployment

### Common Deployment Issues

#### 1. Environment Issues

**Problem**: Import errors

```bash
# Solution: Reinstall package
pip install -e . --no-deps
python -c "import crackseg; print('✅ Success')"
```

#### 2. Configuration Issues

**Problem**: Key 'model' is not in struct

```bash
# Solution: Use fixed run.py with nested config handling
python run.py --config-name=experiments/swinv2_hybrid/swinv2_360x360_corrected +training.epochs=2
```

#### 3. Dataset Issues

**Problem**: Dataset not found

```bash
# Solution: Verify dataset paths
ls data/crack500/  # Should exist
ls data/PY-CrackBD/  # Should exist
```

#### 4. Memory Issues

**Problem**: CUDA out of memory

```bash
# Solution: Reduce batch size or use CPU
python run.py --config-name=experiments/swinv2_hybrid/swinv2_360x360_corrected training.device=cpu
```

### Deployment Verification

**Success Indicators**:

- ✅ Training starts without errors
- ✅ Metrics are logged correctly
- ✅ Checkpoints are saved
- ✅ TensorBoard shows progress
- ✅ No memory leaks

## 🚀 Production Deployment Considerations

### Scalability

**Current Limitations**:

- **Single GPU**: RTX 3070 Ti (8GB VRAM)
- **Batch Size**: Maximum 12 for current models
- **Dataset Size**: Limited by available memory

**Optimization Strategies**:

- **Gradient Accumulation**: Effective batch size of 24
- **Mixed Precision**: FP16 training for memory efficiency
- **Model Optimization**: Pruning and quantization

### Security

**Best Practices**:

- **Environment Isolation**: Use conda environments
- **Configuration Security**: Don't commit sensitive data
- **Access Control**: Limit access to production systems
- **Logging**: Monitor for suspicious activity

### Backup and Recovery

**Critical Files to Backup**:

- **Configurations**: `configs/experiments/swinv2_hybrid/`
- **Datasets**: `data/crack500/`, `data/PY-CrackBD/`
- **Checkpoints**: `artifacts/experiments/*/checkpoints/`
- **Logs**: `artifacts/experiments/*/logs/`

**Recovery Procedures**:

1. **Restore environment**: `conda env create -f environment.yml`
2. **Reinstall package**: `pip install -e . --no-deps`
3. **Restore datasets**: Copy from backup
4. **Restore checkpoints**: Copy from backup

## 📈 Performance Optimization

### Hardware Optimization

**RTX 3070 Ti Settings**:

```yaml
training:
  batch_size: 12  # Optimal for 8GB VRAM
  gradient_accumulation_steps: 2
  use_amp: true  # Mixed precision
  device: auto
```

### Memory Management

**Monitoring Commands**:

```bash
# Monitor GPU memory
nvidia-smi

# Monitor system resources
htop  # Linux/macOS
taskmgr  # Windows
```

### Training Optimization

**Recommended Settings**:

- **Learning Rate**: 5.0e-05 (optimized for current models)
- **Optimizer**: AdamW with weight decay
- **Scheduler**: CosineAnnealingWarmRestarts
- **Loss**: BCEDiceLoss (balanced for crack segmentation)

## 🔄 Continuous Deployment

### Automated Testing

**Pre-Deployment Tests**:

```bash
# Quality gates
black . && python -m ruff . --fix && basedpyright .

# Unit tests
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/
```

### Deployment Pipeline

**Recommended Workflow**:

1. **Development**: Test in local environment
2. **Staging**: Test with subset of data
3. **Production**: Full deployment with monitoring

### Version Control

**Configuration Versioning**:

- **Track changes**: Use Git for configuration files
- **Tag releases**: Tag stable configurations
- **Document changes**: Update changelog

## 📚 References

- **Successful Experiments**: `../successful_experiments_guide.md`
- **Training Workflow**: `../workflows/training_workflow_guide.md`
- **Configuration Guide**: `../../technical-specs/configuration_guide.md`
- **Troubleshooting**: `../../user-guides/troubleshooting.md`

---

**Last Updated**: December 2024
**Status**: Active - Verified with successful deployments
**Next Steps**: Optimize for multi-GPU and distributed training
