# Tutorial 1: Basic Training Workflow

This tutorial guides you through running a basic training experiment using a
pre-defined configuration with the new interactive GUI.

## Prerequisites

- You have successfully installed the project and its dependencies. See
  [CLEAN_INSTALLATION.md](../../guides/CLEAN_INSTALLATION.md).
- You have activated the `crackseg` conda environment.

## Step 1: Launch the GUI

From the project root directory, run the following command in your terminal:

```bash
streamlit run scripts/gui/app.py
```

This will open the CrackSeg application in your web browser, displaying the new
**Home** page.

## Step 2: Navigate to the Configuration Page

On the Home page, under "Quick Actions", click the **"Configure Experiment"**
button. This will take you to the **Experiment Configuration** page.

## Step 3: Load a Configuration

1. On the configuration page, the **"Model Configuration"** section is
    expanded by default.
2. Under the "Browse Project Files" tab, you'll see a file browser.
3. Navigate the `configs/` directory. For a basic training, click on
    `base.yaml`.
4. The configuration will be loaded instantly, and you'll see a success
    message: "✅ Configuration loaded". The editor below will also be
    populated with the file's content.

## Step 4: Set the Run Directory

1. Scroll down and expand the **"Output & Run Directory"** section.
2. Enter a path for your experiment's output in the text box, for example:
    `outputs/basic_training_run`.
3. Click the **"Set Directory"** button. This directory will be created if it
    doesn't exist and will store all your training artifacts (logs,
    checkpoints, etc.).

## Step 5: Start the Training

1. Once the configuration and run directory are set, a "Next Steps" section
    appears at the bottom.
2. Click the **"🚀 Start Training"** button.
3. You will be redirected to the **Train** page.
4. Click the **"Start Training"** button on this page to begin the process.
5. You can now monitor the training in real-time:
    - The **Live Log Viewer** shows the direct output from the training
      script.
    - The **Training Metrics** chart visualizes key metrics like loss as
      they are generated.

## Step 6: View Results

Once the training is complete:

1. You can navigate to the **Results** page using the **"📊 View Results"**
    button available on the Home page.
2. Here you can view the final metrics and analyze the model's predictions on
    validation images.

## What's Next?

In the next tutorial, you will learn how to use the advanced editor to create
and save your own custom experiment configurations.
