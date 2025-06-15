# Usage Guide

This guide explains how to use the CrackSeg Professional GUI to perform inference and visualize results.

## Running the Application

To start the application, ensure your conda environment is activated and run the following command
from the project root:

```bash
streamlit run scripts/gui/app.py
```

The application will open in your default web browser.

## Main Features

The GUI is organized into several pages, accessible from the sidebar.

### 1. Configuration

- **Load Configuration**: Load a model and training configuration from a `.yaml` file.
- **Select Checkpoint**: Choose a specific model checkpoint (`.pth.tar`) for inference.
- **Adjust Parameters**: Modify inference parameters such as image processing settings.

### 2. Inference

- **Upload Images**: Upload one or more images for crack segmentation.
- **Run Inference**: Process the images using the selected model.
- **View Results**: The predicted segmentation masks will be displayed alongside the original images.

### 3. Results Gallery

- **Browse History**: View a gallery of past inference results.
- **Inspect Details**: Click on a result to see the original image, the mask, and an overlay.
- **Export Results**: Export individual or all results to a local directory.

## Example Workflow

1. **Start the GUI**:

    ```bash
    streamlit run scripts/gui/app.py
    ```

2. **Navigate to Configuration**: Select the model configuration and a trained checkpoint.
3. **Navigate to Inference**: Upload a set of pavement images.
4. **Click "Run Inference"**: Wait for the model to process the images.
5. **Analyze Results**: View the generated masks on the page.
6. **Go to Gallery**: Browse and export the results you wish to save.
