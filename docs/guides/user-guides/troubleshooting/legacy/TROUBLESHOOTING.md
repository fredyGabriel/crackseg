# Troubleshooting Guide

This guide provides solutions to common problems encountered while working with the
CrackSeg project. It is organized by category to help you quickly diagnose and resolve issues.

## Contents

- [1. Environment and Installation Issues](#1-environment-and-installation-issues)
- [2. GPU and Memory Errors](#2-gpu-and-memory-errors)
- [3. Configuration (Hydra) Errors](#3-configuration-hydra-errors)
- [4. Data Loading and Processing Errors](#4-data-loading-and-processing-errors)
- [5. Training and Evaluation Issues](#5-training-and-evaluation-issues)
- [6. Streamlit GUI Issues](#6-streamlit-gui-issues)

---

## 1. Environment and Installation Issues

### 1.1. `conda: command not found` or `'conda' is not recognized`

- **Problem**: The system cannot find the Conda executable.
- **Cause**: Conda is not installed or its path is not included in your system's
    `PATH` environment variable.
- **Solution**:
    1. Ensure you have installed Anaconda or Miniconda.
    2. During installation, select the option "Add Anaconda to my PATH environment variable".
    3. If already installed, find your Conda installation's `Scripts` directory
        (e.g., `C:\Users\YourUser\miniconda3\Scripts`) and add it to your system's `PATH`.

### 1.2. `CondaSSLError`, `CondaHTTPError`

- **Problem**: Conda cannot download packages from repositories.
- **Cause**: This is typically due to network connectivity issues, firewalls, or SSL
    certificate problems.
- **Solution**:
    1. Check your internet connection.
    2. If you are behind a corporate proxy, configure Conda's proxy settings in your
        `.condarc` file.
    3. Try disabling SSL verification (use with caution):

        ```bash
        conda config --set ssl_verify false
        ```

    4. Re-enable it once the issue is resolved:

        ```bash
        conda config --set ssl_verify true
        ```

### 1.3. `ModuleNotFoundError: No module named 'some_package'`

- **Problem**: A required Python package is not installed in the active environment.
- **Cause**: You have not activated the correct Conda environment, or the environment
    is missing dependencies.
- **Solution**:
    1. **Activate the environment**: Always run `conda activate crackseg` before
        executing any scripts.
    2. **Verify installation**: Check if the package is listed in `environment.yml`.
    3. **Update environment**: Run `conda env update --file environment.yml --prune`
        to install missing packages and remove unused ones.

---

## 2. GPU and Memory Errors

### 2.1. `CUDA out of memory`

- **Problem**: The GPU does not have enough VRAM to accommodate the model and the
    data batch. This is common on GPUs with limited memory like the RTX 3070 Ti (8GB).
- **Cause**: The batch size is too large, the model is too complex, or the input
    image resolution is too high.
- **Solution (choose one or more)**:
    1. **Reduce Batch Size**: In your Hydra configuration
        (e.g., `configs/training/default.yaml`), decrease `data.batch_size`.
        Try `8`, `4`, or even `2`.
    2. **Reduce Image Resolution**: In `configs/data/default.yaml`, decrease
        the image `height` and `width` in the `transforms` section.
    3. **Enable Mixed Precision**: Set `training.mixed_precision=True` in your
        configuration. This can reduce memory usage significantly.
    4. **Gradient Accumulation**: This is a more advanced technique that involves
        modifying the training loop to accumulate gradients over several smaller
        batches before performing an optimizer step.

### 2.2. `Could not load dynamic library 'cudart64_...dll'`

- **Problem**: PyTorch cannot find the NVIDIA CUDA Toolkit libraries.
- **Cause**: The CUDA Toolkit is not installed correctly, or the version is
    incompatible with your PyTorch build or NVIDIA driver.
- **Solution**:
    1. **Verify Driver**: Ensure you have the latest NVIDIA drivers for your GPU.
    2. **Verify CUDA Version**: Check the CUDA version PyTorch was built with:
        `python -c "import torch; print(torch.version.cuda)"`.
    3. **Install Correct CUDA Toolkit**: Download and install the matching version
        from the NVIDIA website. Ensure it's added to your system's `PATH`.

---

## 3. Configuration (Hydra) Errors

### 3.1. `Could not find 'some_key' in 'full_key'`

- **Problem**: Hydra cannot resolve a key in the configuration.
- **Cause**: A typo, an incorrect path in a YAML file, or a missing default value.
- **Solution**:
    1. Carefully check the key mentioned in the error message for typos.
    2. Ensure that the configuration file where the key is defined is correctly
        included in `configs/base.yaml` or your experiment file.
    3. Use the command line override `--config-dir` and `--config-name` to debug
        which configuration files are being loaded.

---

## 4. Data Loading and Processing Errors

### 4.1. `FileNotFoundError: Cannot find image/mask at path...`

- **Problem**: The `Dataset` class cannot locate the data files.
- **Cause**: The data paths in your configuration (`data.data_dir`) are incorrect,
    or the dataset is not structured as expected (e.g., missing `images` or `masks`
    subdirectories).
- **Solution**:
    1. Verify that `data.data_dir` in your configuration points to the correct
        root directory of your dataset.
    2. Ensure the dataset directory contains the `train/`, `val/`, and `test/`
        subdirectories, each with `images/` and `masks/` folders inside.

---

## 5. Training and Evaluation Issues

### 5.1. Loss is `NaN` or `inf`

- **Problem**: The training loss becomes Not-a-Number (NaN) or infinity.
- **Cause**: "Exploding gradients" due to a high learning rate, or numerical
    instability in a loss function or model layer.
- **Solution**:
    1. **Lower Learning Rate**: Decrease `training.optimizer.lr` in your
        configuration. Try reducing it by a factor of 10.
    2. **Gradient Clipping**: In the training script, add
        `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
        before `optimizer.step()`.
    3. **Check Data**: Ensure your data is normalized correctly and does not
        contain corrupted labels.

---

## 6. Streamlit GUI Issues

### 6.1. GUI is slow or unresponsive

- **Problem**: The GUI freezes or takes a long time to update.
- **Cause**: A long-running process (like model loading or a training step) is
    blocking the main thread.
- **Solution**:
    1. The GUI is designed to run training in a separate thread. If it's still
        unresponsive, it might be due to heavy data processing for visualization.
    2. Restart the Streamlit server.
    3. Check the terminal where you launched Streamlit for any error messages.

### 6.2. `Process has already been started`

- **Problem**: You try to start training, but the GUI reports that a process is
    already running.
- **Cause**: The state management did not correctly register that a previous
    training run finished or was terminated.
- **Solution**:
    1. Click the "Stop" or "Kill" button in the GUI if it is available.
    2. If the button is not available or does not work, the safest solution is to
        stop the Streamlit server (Ctrl+C in the terminal) and restart it.
