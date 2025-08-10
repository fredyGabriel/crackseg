# Clean Environment Installation Guide

This guide provides step-by-step instructions for installing the CrackSeg project from scratch in a
clean environment.

## Prerequisites

Before starting, ensure you have the following system dependencies installed:

### Required System Dependencies

- **Git**: Version control system
- **Conda/Miniconda**: Package and environment management
- **Python 3.12**: Specified Python version (managed via Conda)

### Optional but Recommended

- **CUDA Toolkit**: For GPU acceleration (if using NVIDIA GPU)
- **Graphviz**: For visualization capabilities

For detailed installation instructions for these prerequisites, see `docs/guides/developer-guides/architecture/legacy/architectural_decisions.md#adr-001`.

## Step-by-Step Installation

### 1. Clone the Repository

```bash
# Clone the project repository
git clone https://github.com/fredyGabriel/crackseg.git
cd crackseg
```

### 2. Create Conda Environment

```bash
# Create the conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate crackseg
```

**Expected Output:**

- Environment creation should complete without errors
- You should see package installation progress
- Environment should activate successfully

### 3. Install Additional Dependencies

```bash
# Install pip-only dependencies
pip install -r requirements.txt
```

**Note:** Some packages (like Streamlit components) are only available via pip and are automatically
installed from requirements.txt.

### 4. Verify Installation

#### Run System Dependencies Check

```bash
python scripts/verify_system_dependencies.py
```

#### Run Python Compatibility Check

```bash
python scripts/verify_python_compatibility.py
```

#### Run Complete Installation Test

```bash
python scripts/test_clean_installation.py
```

### 5. Verify Project Structure

Ensure all key directories and files are present:

```txt
crackseg/
├── src/                     # Main source code
├── tests/                   # Test suites
├── configs/                 # Configuration files
├── scripts/                 # Utility scripts
├── docs/                    # Documentation
├── environment.yml          # Conda environment specification
├── requirements.txt         # Pip requirements
├── pyproject.toml          # Project configuration
└── README.md               # Project overview
```

### 6. Test Development Tools

```bash
# Check code formatting
black --check .

# Check linting
ruff check .

# Check type annotations
basedpyright .

# Run tests
pytest tests/ --cov=src --cov-report=term-missing
```

## Verification Checklist

Use this checklist to ensure your installation is complete and functional:

### ✅ System Prerequisites

- [ ] Git installed and accessible
- [ ] Conda/Miniconda installed
- [ ] Python 3.12 available through Conda

### ✅ Environment Setup

- [ ] Conda environment 'crackseg' created successfully
- [ ] Environment activated without errors
- [ ] All conda dependencies installed

### ✅ Dependencies

- [ ] Core ML libraries (PyTorch, NumPy, OpenCV) working
- [ ] Configuration management (Hydra, OmegaConf) available
- [ ] Development tools (Black, Ruff, basedpyright) functional
- [ ] Testing framework (pytest) operational

### ✅ Project Functionality

- [ ] Project modules import correctly
- [ ] CUDA functionality working (if GPU available)
- [ ] Verification scripts run successfully
- [ ] Quality gates pass

### ✅ Development Workflow

- [ ] Code formatting with Black works
- [ ] Linting with Ruff produces no errors
- [ ] Type checking with basedpyright passes
- [ ] Basic tests can be executed

## Common Issues and Solutions

### Issue: Conda Environment Creation Fails

**Symptoms:**

- Error during `conda env create -f environment.yml`
- Missing package conflicts

**Solutions:**

1. Update conda: `conda update conda`
2. Clear conda cache: `conda clean --all`
3. Try creating with specific channel: `conda env create -f environment.yml -c conda-forge`

### Issue: Streamlit Installation Problems

**Symptoms:**

- Streamlit components fail to install
- Import errors with streamlit modules

**Solutions:**

1. Ensure you're in the crackseg environment: `conda activate crackseg`
2. Install via pip: `pip install streamlit streamlit-option-menu streamlit-ace`
3. If conflicts persist, install specific versions from requirements.txt

### Issue: CUDA Not Detected

**Symptoms:**

- `torch.cuda.is_available()` returns False
- GPU not being utilized

**Solutions:**

1. Verify GPU drivers are installed
2. Check CUDA toolkit installation
3. Ensure PyTorch CUDA version matches your CUDA installation
4. Reinstall PyTorch with correct CUDA version:

   ```bash
   conda install pytorch pytorch-cuda=12.9 -c pytorch -c nvidia
   ```

### Issue: Import Errors

**Symptoms:**

- Cannot import project modules
- ModuleNotFoundError for crackseg modules

**Solutions:**

1. Ensure you're in the project root directory
2. Install the module in editable mode:

   ```bash
   conda activate crackseg && pip install -e . --no-deps
   ```

3. Verify the installation:

   ```bash
   conda activate crackseg && python -c "import crackseg; print('✅ Module installed successfully')"
   ```

4. If issues persist, check that all dependencies are installed:

   ```bash
   conda activate crackseg && conda list | grep crackseg
   ```

### Issue: Type Checking Errors

**Symptoms:**

- basedpyright reports errors
- Type annotations not recognized

**Solutions:**

1. Ensure basedpyright is installed: `pip install basedpyright`
2. Check pyrightconfig.json exists and is properly configured
3. Verify Python version is 3.12: `python --version`

## Performance Optimization

### For Development

- Use `conda-libmamba-solver` for faster dependency resolution:

  ```bash
  conda install -n base conda-libmamba-solver
  conda config --set solver libmamba
  ```

### For GPU Workloads

- Verify GPU memory is sufficient for your models
- Monitor GPU utilization: `nvidia-smi`
- Adjust batch sizes based on available GPU memory

## Next Steps

After successful installation:

1. **Read the Project Documentation**: Review README.md and docs/ directory
2. **Run Example Scripts**: Test basic functionality with provided examples
3. **Configure for Your Environment**: Adjust configurations in configs/ directory
4. **Start Development**: Follow the development workflow in the main README

## Troubleshooting

If you encounter issues not covered here:

1. **Check the Logs**: Review error messages carefully
2. **Verify Prerequisites**: Ensure all system dependencies are correctly installed
3. **Environment Isolation**: Make sure no conflicting packages from other environments
4. **Update Documentation**: If you solve a new issue, consider contributing to this guide

## Automated Installation Verification

For a comprehensive check of your installation, run:

```bash
python scripts/test_clean_installation.py
```

This script will:

- Verify all prerequisites are available
- Check project structure completeness
- Test conda environment setup
- Validate all dependencies
- Confirm project imports work
- Test development tools
- Run quality gates
- Execute basic functionality tests

The script provides a complete report of your installation status and highlights any issues that
need attention.

---

**Installation Support**: If you continue to experience issues, please check the project's issue
tracker or contact the development team.
