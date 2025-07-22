# CrackSeg Professional GUI - User Guide

This guide provides a comprehensive walkthrough of the CrackSeg Professional Graphical User
Interface (GUI). It is designed to help users navigate the application, configure experiments,
run training, and analyze results effectively.

## 1. Getting Started

### 1.1. Launching the Application

Before launching, ensure your Conda environment is activated. Run the following command from
the project's root directory:

```bash
conda activate crackseg && streamlit run gui/app.py
```

The application will open in your default web browser, presenting the main interface.

> **[Screenshot: The initial loading screen of the CrackSeg application, showing the title and
> a welcome message.]**

---

## 2. Navigating the Interface

The GUI is structured around a main sidebar for navigation and a central content area where
each page's functionality is displayed.

### 2.1. Sidebar Navigation

The sidebar on the left provides access to all major pages of the application.

> **[Screenshot: A close-up of the sidebar, highlighting the navigation links: 'Config',
> 'Architecture', 'Train', and 'Results'.]**

The navigation breadcrumbs at the top of the page show your current location within the app.

> **[Screenshot: The top section of a page, showing the breadcrumb trail, e.g.,
> 'Navigation: 🏠 > 🔧 Config'.]**

---

## 3. Configuration (`Config` Page)

This is the starting point for any workflow. Here, you define the model, data, and training
parameters.

### 3.1. Loading Configuration

- **Load Config File**: Click the "Upload a YAML file" button to load a pre-defined Hydra
  configuration. This populates all the necessary fields for a reproducible experiment.
- **Run Directory**: Specify or create a directory where all outputs (checkpoints, logs,
  results) for this run will be saved.

> **[Screenshot: The 'Config' page with the file uploader and run directory input field
> clearly visible. An arrow points to the "Upload" button.]**

### 3.2. Auto-Save and Drafts

The application automatically saves your configuration changes in your browser's local
storage as a draft. If you accidentally close the tab, your changes can be recovered.

---

## 4. Model Architecture (`Architecture` Page)

This page provides a visual representation of the currently loaded model architecture.

- **Visualization**: An interactive graph shows the layers and connections within the model.
  This is useful for verifying that your configuration has been loaded correctly.

> **[Screenshot: The 'Architecture' page displaying a Graphviz visualization of a U-Net model.
> Key components like 'Encoder', 'Bottleneck', and 'Decoder' are visible.]**

---

## 5. Training (`Train` Page)

This is where you launch, monitor, and manage the model training process.

### 5.1. Device Selection

Before starting, select the hardware for training:

- **GPU (CUDA)**: If a compatible NVIDIA GPU is detected, this option will be available for
  accelerated training.
- **CPU**: If no GPU is available, training will run on the CPU.

> **[Screenshot: The 'Device Selector' component on the 'Train' page, showing options for
> 'cuda' and 'cpu'.]**

### 5.2. Launching and Monitoring

- **Start Training**: Click this button to begin the training process.
- **Progress Monitoring**: Real-time updates are provided through:
  - A main progress bar for the total training epochs.
  - Live metrics for loss and accuracy.
  - An integrated TensorBoard panel for detailed analysis.

> **[Screenshot: The 'Train' page during a training session. The progress bar is partially
> filled, metric charts are updating, and the TensorBoard component is active.]**

### 5.3. Confirmation Dialogs

Critical actions, such as starting or stopping a training run, will trigger a confirmation
dialog to prevent accidental clicks.

> **[Screenshot: A confirmation dialog pop-up asking "Are you sure you want to start the
> training session?". The 'Confirm' and 'Cancel' buttons are visible.]**

---

## 6. Results (`Results` Page)

After training, this page is your hub for analyzing model performance and visualizing
predictions.

### 6.1. Results Gallery

- **Image Predictions**: View a gallery of original images, their ground truth masks, and the
  model's predicted masks side-by-side. This allows for qualitative assessment of the model's
  performance.

> **[Screenshot: The 'Results Gallery' showing a triplet of images (original, ground truth,
> prediction) for a single test sample. A crack is clearly segmented in the prediction.]**

### 6.2. Metrics and Analysis

- **Quantitative Metrics**: Review detailed performance metrics like IoU, Dice Score,
  Precision, and Recall.
- **TensorBoard Integration**: Dive deeper into the training history, analyze learning curves,
  and inspect model graphs using the embedded TensorBoard interface.

> **[Screenshot: The 'TensorBoard' tab on the 'Results' page, displaying loss curves and
> IoU metrics over training epochs.]**
