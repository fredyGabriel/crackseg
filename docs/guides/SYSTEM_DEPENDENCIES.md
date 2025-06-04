# System Dependencies - CrackSeg Project

This document lists all system (non-Python) dependencies required for the CrackSeg pavement crack segmentation project.

## 📋 System Dependencies Overview

### Required Dependencies

- **Graphviz**: For model architecture visualization
- **Git**: For version control
- **Conda/Miniconda**: For Python environment management

### Optional Dependencies

- **CUDA Toolkit**: For GPU acceleration (recommended for training)
- **FFmpeg**: For video processing (future functionality)

---

## 🖥️ Windows (Primary Environment)

### 1. Graphviz

**Purpose**: Model architecture visualization and flowchart diagrams

**Installation**:

```bash
# Option 1: Conda (Recommended)
conda install -c conda-forge graphviz

# Option 2: Chocolatey
choco install graphviz

# Option 3: Manual
# Download from https://graphviz.org/download/
# Add to PATH: C:\Program Files\Graphviz\bin
```

**Verification**:

```bash
dot -V
python -c "import graphviz; print('✅ Graphviz working')"
```

### 2. Git

**Purpose**: Version control and repository cloning

**Installation**:

```bash
# Option 1: Chocolatey
choco install git

# Option 2: Manual
# Download from https://git-scm.com/download/win
```

**Verification**:

```bash
git --version
```

### 3. Conda/Miniconda

**Purpose**: Python environment management and scientific dependencies

**Installation**:

```bash
# Option 1: Miniconda (Recommended - lighter)
# Download from https://docs.conda.io/en/latest/miniconda.html

# Option 2: Anaconda (Complete)
# Download from https://www.anaconda.com/products/distribution
```

**Verification**:

```bash
conda --version
```

### 4. CUDA Toolkit (Optional - GPU)

**Purpose**: GPU acceleration for model training

**Installation**:

```bash
# Check GPU compatibility
nvidia-smi

# Install CUDA Toolkit 12.1 (compatible with PyTorch 2.5+)
# Download from https://developer.nvidia.com/cuda-downloads
```

**Verification**:

```bash
nvcc --version
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🐧 Linux (Ubuntu/Debian)

### Install All Dependencies

```bash
# Update repositories
sudo apt update

# Basic dependencies
sudo apt install -y \
    git \
    graphviz \
    graphviz-dev \
    build-essential \
    curl \
    wget

# Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# CUDA (if you have NVIDIA GPU)
# Follow official NVIDIA instructions for your distribution
```

---

## 🍎 macOS

### Installation with Homebrew

```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install git
brew install graphviz
brew install --cask miniconda

# Verify installations
git --version
dot -V
conda --version
```

---

## 🔧 Post-Installation Configuration

### 1. Environment Variables

Add to shell configuration file (`.bashrc`, `.zshrc`, etc.):

```bash
# Windows (PowerShell Profile)
# Add to $PROFILE

# Linux/macOS
export PATH="/path/to/graphviz/bin:$PATH"
export GRAPHVIZ_ROOT="/path/to/graphviz"
```

### 2. Complete System Verification

Run the verification script:

```bash
# From project directory
python scripts/verify_system_dependencies.py
```

---

## 🐳 Docker (Alternative)

To avoid manual installations, Docker can be used:

```dockerfile
# Dockerfile includes all system dependencies
FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y \
    graphviz \
    graphviz-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Rest of configuration is in project Dockerfile
```

---

## 🚨 Common Troubleshooting

### Graphviz Not Found

```bash
# Windows
# Verify that C:\Program Files\Graphviz\bin is in PATH

# Linux
sudo apt install graphviz-dev

# macOS
brew install graphviz
```

### CUDA Not Detected

```bash
# Check NVIDIA drivers
nvidia-smi

# Check compatible CUDA version
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Conda Not Working

```bash
# Reinitialize conda
conda init
# Restart terminal

# Or use full path
/path/to/miniconda3/bin/conda --version
```

---

## 📋 Verification Checklist

- [ ] **Git installed and working**
- [ ] **Conda/Miniconda installed**
- [ ] **Graphviz installed and in PATH**
- [ ] **Python 3.12+ available in conda**
- [ ] **CUDA Toolkit installed (if using GPU)**
- [ ] **Environment variables configured**
- [ ] **Verification script executed successfully**

---

## 🔗 References

- [Graphviz Downloads](https://graphviz.org/download/)
- [Git Downloads](https://git-scm.com/downloads)
- [Miniconda Downloads](https://docs.conda.io/en/latest/miniconda.html)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

---

## 📝 Version Notes

- **Last updated**: January 2025
- **Recommended CUDA version**: 12.1+
- **Recommended Python version**: 3.12+
- **Minimum Graphviz version**: 2.40+
