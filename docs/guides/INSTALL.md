# Installation Guide

This guide provides instructions for setting up the environment for the CrackSeg Professional GUI.

## Prerequisites

- **Conda**: You must have a working installation of Anaconda or Miniconda.
- **Git**: Required for cloning the repository.

## Installation Steps

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/fgrv/crackseg.git
    cd crackseg
    ```

2. **Create Conda Environment**:
    Use the provided `environment.yml` file to create the conda environment. This ensures all
    dependencies are installed with the correct versions.

    ```bash
    conda env create -f environment.yml
    ```

3. **Activate the Environment**:
    Before running the application, you must activate the conda environment.

    ```bash
    conda activate crackseg
    ```

4. **Install in Editable Mode**:
    Install the project in editable mode to ensure that the Python interpreter can find the source
    code in the `scripts` directory.

    ```bash
    pip install -e .
    ```

## Verify Installation

To verify that the installation was successful, you can run the application's help command.

```bash
streamlit run gui/app.py -- --help
```

This should display the help message for the application without any errors. You are now ready to
use the CrackSeg Professional GUI.
