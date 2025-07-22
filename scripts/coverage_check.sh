#!/bin/bash
# Simple Coverage Check Script - CrackSeg Project
# Replaces complex validate_coverage.py with efficient pytest-native approach

set -e

echo "🧪 CrackSeg Coverage Check"
echo "========================="

# Activate conda environment
if ! conda info --envs | grep -q "crackseg"; then
    echo "❌ Conda environment 'crackseg' not found"
    exit 1
fi

# Check coverage with pytest native tools
echo "📊 Running coverage analysis..."
conda activate crackseg && pytest tests/ \
    --cov=src \
    --cov=gui \
    --cov-report=html \
    --cov-report=term-missing \
    --cov-fail-under=80

echo "✅ Coverage check completed"
echo "📋 HTML report available at: htmlcov/index.html"