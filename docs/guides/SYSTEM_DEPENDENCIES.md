# System Dependencies Guide

## Overview

This guide covers the system-level dependencies required to run the CrackSeg project, including
hardware requirements, operating system considerations, and installation instructions.

## Required Dependencies

### Core Requirements

- **Python 3.12+**: Via Conda/Miniconda
- **Git**: For version control and repository management
- **Compatible Graphics Driver**: NVIDIA drivers recommended for CUDA acceleration

### ~~Removed Dependencies~~

- ~~**Graphviz**: No longer required~~ ✅ **Replaced with matplotlib** (ADR-001)
  - **Reason**: Complex compilation issues on Windows with PyTorch 2.7
  - **Alternative**: Built-in matplotlib-based architecture visualization
  - **Benefits**: Simpler setup, better cross-platform compatibility

### 1. ~~Graphviz~~ → **Matplotlib Architecture Visualization**

**✅ NEW**: Matplotlib-based visualization (default)

```bash
# Already included in conda environment
conda activate crackseg
python -c "import matplotlib.pyplot as plt; print('✅ Matplotlib available')"
```

**⚠️ LEGACY**: Graphviz (optional, for advanced users)

```bash
# Windows - Advanced users only
choco install graphviz
# OR download from https://graphviz.org/download/
# Add to PATH: C:\Program Files\Graphviz\bin

# Linux - Advanced users only
sudo apt install graphviz graphviz-dev

# macOS - Advanced users only
brew install graphviz
```

**Test architecture visualization:**

```bash
python -c "
from src.crackseg.model.common.utils import render_unet_architecture_diagram
print('✅ Architecture visualization ready (matplotlib backend)')
"
```

---

## Cross-Platform Installation Guide

### 1. Basic System Requirements

**Core Requirements:**

- **Git**: Version control (any recent version)
- **Conda/Miniconda**: Python environment management
- **Compatible GPU drivers**: NVIDIA drivers for CUDA support (optional)

**Platform-Specific Installation:**

```bash
# Windows (using Chocolatey)
choco install git miniconda3

# Linux (Ubuntu/Debian)
sudo apt update && sudo apt install -y git curl
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# macOS (using Homebrew)
brew install git miniconda
```

### 2. CrackSeg Environment Setup

**Create and activate the environment:**

```bash
# Clone repository
git clone https://github.com/crackseg/crackseg.git
cd crackseg

# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate crackseg

# Verify installation
python -c "import torch, matplotlib; print('✅ Core dependencies ready')"
```

### 3. Optional: GPU Support

**For NVIDIA GPU acceleration:**

```bash
# Check GPU compatibility
nvidia-smi

# CUDA is included in environment.yml - verify it works
conda activate crackseg && python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

---

## Verification & Testing

### 1. Complete System Verification

**Run comprehensive verification:**

```bash
conda activate crackseg && python -c "
import torch
import matplotlib.pyplot as plt
import streamlit as st
import hydra
import numpy as np

print('✅ Core dependencies verified')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')

try:
    from src.crackseg.model.common.utils import render_unet_architecture_diagram
    print('✅ Architecture visualization ready')
except ImportError as e:
    print(f'⚠️ Project modules: {e}')
"
```

### 2. Common Issues

**Environment not found:**

```bash
# List available environments
conda env list

# Recreate if corrupted
conda env remove -n crackseg
conda env create -f environment.yml
```

**GPU not detected:**

```bash
# Check NVIDIA driver
nvidia-smi

# Verify PyTorch CUDA
conda activate crackseg && python -c "
import torch
print(f'CUDA devices: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
"
```

---

## Hardware Requirements

### Minimum Configuration

- **CPU**: 4+ cores
- **RAM**: 8GB
- **GPU**: Any CUDA-compatible (optional for CPU-only development)
- **Storage**: 10GB free space

### Recommended Configuration

- **CPU**: 8+ cores for efficient data loading
- **RAM**: 16GB+ for large datasets
- **GPU**: RTX 3070 Ti or better (8GB+ VRAM)
- **Storage**: NVMe SSD for faster I/O

---

## 📋 Setup Checklist

- [ ] **Miniconda/Conda installed**
- [ ] **Git available**
- [ ] **Repository cloned**
- [ ] **Environment created from environment.yml**
- [ ] **Environment activation works**
- [ ] **Core packages import successfully**
- [ ] **GPU detected (if available)**
- [ ] **Architecture visualization functional**

---

## 🔗 Resources

- **[PyTorch Installation](https://pytorch.org/get-started/locally/)**: Official PyTorch setup guide
- **[Conda Documentation](https://docs.conda.io/en/latest/)**: Environment management guide
- **[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)**: GPU acceleration setup
- **[Project ADR-001](architectural_decisions.md#adr-001)**: Graphviz migration details

---

## 📝 Updated January 2025

- **Simplified dependencies**: Removed graphviz requirement
- **Matplotlib-first**: Architecture visualization via matplotlib
- **PyTorch 2.7**: Latest stable with CUDA 12.9 support
- **Cross-platform**: Tested on Windows, Linux, macOS
