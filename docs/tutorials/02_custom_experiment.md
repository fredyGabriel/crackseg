# Tutorial 2: Creating a Custom Experiment

This tutorial explains how to use the GUI's advanced features to create,
modify, and save a custom configuration for your experiment.

## Prerequisites

- You have completed Tutorial 1 and understand the basic training workflow.

## Step 1: Load a Base Configuration

1. Launch the GUI (`streamlit run scripts/gui/app.py`).
2. Navigate to the **Experiment Configuration** page.
3. Load the `configs/base.yaml` file using the file browser, just as you did
    in the previous tutorial.

## Step 2: Modify the Configuration in the Editor

1. Once `base.yaml` is loaded, scroll down to the
    **"Editor & Real-time Validation"** expander.
2. The content of the configuration is displayed in an advanced code editor.
3. Let's modify the learning rate. Find the `optimizer` section and change
    the `lr` value from `0.0001` to `0.0005`.

    ```yaml
    # ...
    optimizer:
      _target_: torch.optim.AdamW
      lr: 0.0001 # Change this to 0.0005
    # ...
    ```

4. Notice the **Real-time Validation** panel next to the editor. It provides
    instant feedback. If you introduce a syntax error (e.g., incorrect
    indentation), it will immediately flag it as a `YAML Error`. If you
    reference a component that doesn't exist, it will raise a
    `Hydra Instantiation Error`.

## Step 3: Save the Custom Configuration

1. After modifying the content, scroll to the "Save Changes" section below
    the editor.
2. Click the **"💾 Save Configuration As..."** button. A save dialog will
    appear.
3. In the dialog, provide a **File Name** for your new configuration, for
    example: `custom_lr_experiment`. The `.yaml` extension is added
    automatically.
4. The dialog shows where the file will be saved (typically in the
    `generated_configs/` directory).
5. Click **"Save"**. The file is saved, and the dialog closes. You should see a
    notification confirming the save.

## Step 4: Run the Custom Experiment

1. Your new configuration is **not** automatically loaded. You must now load
    the file you just created.
2. Go back to the **"Model Configuration"** section at the top of the page.
3. Use the file browser to navigate into the `generated_configs/` directory
    and select your `custom_lr_experiment.yaml` file.
4. Set a new, descriptive **Run Directory**, such as
    `outputs/custom_lr_run`.
5. Navigate to the **Train** page and start the training.
6. Monitor the experiment. How does the increased learning rate affect the
    training dynamics compared to your first run?

## What's Next?

You now know how to tailor experiments using the GUI. The next tutorial covers
the more advanced topic of extending the project with your own custom Python
components, like a new model architecture or loss function.
