#!/bin/bash
# Simple Installation Check - CrackSeg Project
# Replaces complex test_clean_installation.py with essential checks

set -e

echo "🔍 CrackSeg Installation Check"
echo "=============================="

# Activate conda environment
if ! conda info --envs | grep -q "crackseg"; then
    echo "❌ Conda environment 'crackseg' not found"
    echo "💡 Run: conda env create -f environment.yml"
    exit 1
fi

echo "✅ Conda environment found"

# Test core imports
echo "📦 Testing core imports..."
conda activate crackseg && python -c "
import src.crackseg
import torch
import hydra
import timm
import albumentations
import streamlit
print('✅ Core imports successful')
"

# Test PyTorch CUDA support
echo "🚀 Testing PyTorch CUDA support..."
conda activate crackseg && python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
"

# Test modern computer vision libraries
echo "🖼️ Testing computer vision libraries..."
conda activate crackseg && python -c "
import timm
import albumentations
print(f'timm version: {timm.__version__}')
print(f'albumentations version: {albumentations.__version__}')
print('✅ Computer vision libraries available')
"

# Run essential tests
echo "🧪 Running essential tests..."
conda activate crackseg && pytest tests/unit/ --maxfail=5 --tb=short

echo "✅ Installation validation completed"
echo "🎯 System ready for development"