# Tutorial 1: Basic Training Workflow

**TODO: This tutorial needs verification and testing with the current GUI implementation.**

This tutorial guides you through running a basic training experiment using a
pre-defined configuration. You can use either the interactive GUI or direct CLI commands.

## Prerequisites

- You have successfully installed the project and its dependencies. See
  [CLEAN_INSTALLATION.md](../../guides/workflows/CLEAN_INSTALLATION.md).
- You have activated the `crackseg` conda environment.
- **CRITICAL**: You have installed the `crackseg` package in development mode.

### Installing the Package (Required)

Before running any training, you must install the `crackseg` package:

#### Option A: Using Conda Environment (Recommended)

```bash
# From the project root directory
conda activate crackseg
pip install -e . --no-deps
```

This installs the package without dependencies, using the conda packages already installed.

#### Option B: Using Pip (Alternative)

```bash
# From the project root directory
conda activate crackseg
pip install -e .
```

This installs the package with all pip dependencies.

**Note**: If you encounter OpenCV installation issues, use Option A (conda) as OpenCV is already
installed via conda.

This makes the `crackseg` package available for imports and ensures all components work correctly.

### Verifying Installation

Verify that the package is installed correctly:

```bash
conda activate crackseg
python -c "import crackseg; print('✅ CrackSeg package imported successfully')"
```

## Option A: Using the GUI (Recommended for Beginners)

### Step 1: Launch the GUI

From the project root directory, run the following command in your terminal:

```bash
conda activate crackseg
streamlit run gui/app.py
```

This will open the CrackSeg application in your web browser, displaying the new
**Home** page.

### Step 2: Navigate to the Configuration Page

On the Home page, use the **"Config"** option in the sidebar navigation (left side of the screen).
This will take you to the **Experiment Configuration** page.

### Step 3: Load a Configuration

1. On the configuration page, the **"Model Configuration"** section is
    expanded by default.
2. Under the "Browse Project Files" tab, you'll see a file browser.
3. Navigate the `configs/` directory. For a basic training, click on
    `base.yaml`.
4. The configuration will be loaded instantly, and you'll see a success
    message: "✅ Configuration loaded". The editor below will also be
    populated with the file's content.

### Step 4: Set the Run Directory

1. Scroll down and expand the **"Output & Run Directory"** section.
2. Enter a path for your experiment's output in the text box, for example:
    `outputs/basic_training_run`.
3. The directory will be automatically set when you enter the path. This directory
    will be created if it doesn't exist and will store all your training artifacts
    (logs, checkpoints, etc.).

### Step 5: Start the Training

1. Once the configuration and run directory are set, you'll see a green "System Ready for Training"
    status in the Setup Status section at the bottom of the page.
2. Navigate to the **Train** page using the sidebar or the **"Start New Training"** button
    on the Home page.
3. On the Train page, click the **"Start Training"** button to begin the process.
4. You can now monitor the training in real-time:
    - The **Live Log Viewer** shows the direct output from the training
      script.
    - The **Training Metrics** chart visualizes key metrics like loss as
      they are generated.

### Step 6: View Results

Once the training is complete:

1. You can navigate to the **Results** page using the **"View Latest Results"**
    button available on the Home page.
2. Here you can view the final metrics and analyze the model's predictions on
    validation images.

## Option B: Using CLI Commands (Advanced Users)

### Step 1: Verify Configuration

First, verify that the base configuration exists and is valid:

```bash
conda activate crackseg
ls configs/base.yaml
```

### Step 2: Run Basic Training

Execute the training directly from the command line:

```bash
conda activate crackseg
python run.py --config-name basic_verification
```

### Step 3: Monitor Training

The training will start and show progress in the terminal. You can monitor:

- Training loss and metrics
- Validation performance
- Checkpoint saving

### Step 4: View Results

Results will be saved in the default output directory. You can view them using:

```bash
conda activate crackseg
ls artifacts/outputs/
```

## Troubleshooting

### Common Issues

1. **Import Error**: No module named 'crackseg'

    - Solution: Run `pip install -e . --no-deps` from the project root (for conda environments)
    - Alternative: Run `pip install -e .` (for pip environments)

2. **Configuration Error**: Could not find 'hydra/default'

    - Solution: This is a known issue. Try using a different configuration or check the configs directory

3. **GUI Not Starting**

    - Solution: Ensure Streamlit is installed: `pip install streamlit`
    - Alternative: Use CLI commands instead

### Quality Gates

After making any changes, run the quality gates:

```bash
conda activate crackseg
black .
python -m ruff . --fix
basedpyright .
```

## What's Next?

In the next tutorial, you will learn how to use the advanced editor to create
and save your own custom experiment configurations.
