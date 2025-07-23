# CrackSeg Professional GUI - User Guide

This guide provides a comprehensive walkthrough of the CrackSeg Professional Graphical User
Interface (GUI). It is designed to help users navigate the application, configure experiments,
run training, and analyze results effectively.

> **⚠️ Important Note**: This GUI guide is based on the current implementation. For verified
> workflows, we recommend following the CLI tutorials in `docs/tutorials/cli/` which have been
> thoroughly tested and validated.

## Prerequisites

Before using the GUI, ensure you have:

1. **Environment Setup**: Follow the [CLEAN_INSTALLATION.md](CLEAN_INSTALLATION.md) guide
2. **Package Installation**: `pip install -e . --no-deps` (for conda environments)
3. **Verification**: `python -c "import crackseg; print('✅ Success')"`
4. **Streamlit**: `pip install streamlit` (if not already installed)

## 1. Getting Started

### 1.1. Launching the Application

Before launching, ensure your Conda environment is activated. Run the following command from
the project's root directory:

```bash
conda activate crackseg
streamlit run gui/app.py
```

The application will open in your default web browser, presenting the main interface with the
**Home Dashboard**.

> **[Screenshot: The initial loading screen of the CrackSeg application, showing the title and
> a welcome message.]**

---

## 2. Navigating the Interface

The GUI is structured around a main sidebar for navigation and a central content area where
each page's functionality is displayed.

### 2.1. Sidebar Navigation

The sidebar on the left provides access to all major pages of the application:

- **🏠 Home**: Main dashboard with quick actions and statistics
- **🔧 Config**: Experiment configuration and file management
- **🏗️ Architecture**: Model architecture visualization
- **⚙️ Advanced Config**: Advanced configuration options
- **🚀 Train**: Training execution and monitoring
- **📊 Results**: Results analysis and visualization

> **[Screenshot: A close-up of the sidebar, highlighting the navigation links: 'Home', 'Config',
> 'Architecture', 'Advanced Config', 'Train', and 'Results'.]**

The navigation breadcrumbs at the top of the page show your current location within the app.

> **[Screenshot: The top section of a page, showing the breadcrumb trail, e.g.,
> 'Navigation: 🏠 > 🔧 Config'.]**

---

## 3. Home Dashboard (`Home` Page)

The Home page serves as the central dashboard and starting point for all workflows.

### 3.1. Project Overview

- **Welcome Message**: Introduction to the CrackSeg project
- **Quick Actions**: Direct access to core workflows:
  - **Start New Training**: Navigate to the Train page
  - **View Latest Results**: Navigate to the Results page
  - **Configure Architecture**: Navigate to the Config page

### 3.2. Dataset Statistics

The dashboard displays key statistics about your dataset:

- **Total Images**: Combined count of all images
- **Training Images**: Number of training samples
- **Validation Images**: Number of validation samples
- **Test Images**: Number of test samples

> **[Screenshot: The Home dashboard showing quick action buttons and dataset statistics
> in a clean, organized layout.]**

---

## 4. Configuration (`Config` Page)

This is the starting point for any workflow. Here, you define the model, data, and training
parameters.

### 4.1. Loading Configuration

- **Browse Project Files**: Use the file browser to navigate and select configuration files
  from the `configs/` directory
- **Load Configuration**: Click on a YAML file to load it into the editor
- **Real-time Validation**: The system validates YAML syntax and Hydra instantiation in real-time
- **Run Directory**: Specify or create a directory where all outputs (checkpoints, logs,
  results) for this run will be saved

### 4.2. Configuration Editor

- **Advanced Editor**: Full-featured code editor with syntax highlighting
- **Real-time Validation**: Instant feedback on YAML syntax and component availability
- **Save Configuration**: Save modified configurations to `generated_configs/` directory
- **Auto-Save**: Automatic draft saving in browser local storage

> **[Screenshot: The 'Config' page with the file browser, configuration editor, and validation
> panel clearly visible.]**

### 4.3. Recommended Configurations

For best results, use these verified configurations:

- **`basic_verification.yaml`**: Recommended for testing and basic training
- **`base.yaml`**: Full configuration with all components
- **Custom configurations**: Create your own in `generated_configs/`

---

## 5. Advanced Configuration (`Advanced Config` Page)

This page provides advanced configuration options and fine-tuning capabilities.

### 5.1. Advanced Parameters

- **Model Parameters**: Fine-tune model architecture settings
- **Training Parameters**: Advanced training configurations
- **Data Parameters**: Dataset and dataloader settings
- **Optimization Parameters**: Optimizer and scheduler settings

### 5.2. Configuration Management

- **Import/Export**: Save and load configuration states
- **Validation**: Advanced validation of complex configurations
- **Templates**: Pre-defined configuration templates

---

## 6. Model Architecture (`Architecture` Page)

This page provides a visual representation of the currently loaded model architecture.

### 6.1. Architecture Visualization

- **Interactive Graph**: Visual representation of model layers and connections
- **Component Details**: Detailed information about each model component
- **Configuration Verification**: Verify that your configuration has been loaded correctly

> **[Screenshot: The 'Architecture' page displaying a Graphviz visualization of a U-Net model.
> Key components like 'Encoder', 'Bottleneck', and 'Decoder' are visible.]**

---

## 7. Training (`Train` Page)

This is where you launch, monitor, and manage the model training process.

### 7.1. Device Selection

Before starting, select the hardware for training:

- **GPU (CUDA)**: If a compatible NVIDIA GPU is detected, this option will be available for
  accelerated training
- **CPU**: If no GPU is available, training will run on the CPU

> **[Screenshot: The 'Device Selector' component on the 'Train' page, showing options for
> 'cuda' and 'cpu'.]**

### 7.2. Launching and Monitoring

- **Start Training**: Click this button to begin the training process
- **Progress Monitoring**: Real-time updates are provided through:
  - A main progress bar for the total training epochs
  - Live metrics for loss and accuracy
  - An integrated TensorBoard panel for detailed analysis
  - Real-time log streaming

### 7.3. Training Management

- **Stop Training**: Safely stop ongoing training sessions
- **Confirmation Dialogs**: Critical actions trigger confirmation dialogs
- **Status Monitoring**: Real-time status updates and error handling

> **[Screenshot: The 'Train' page during a training session. The progress bar is partially
> filled, metric charts are updating, and the TensorBoard component is active.]**

---

## 8. Results (`Results` Page)

After training, this page is your hub for analyzing model performance and visualizing
predictions.

### 8.1. Results Gallery

- **Image Predictions**: View a gallery of original images, their ground truth masks, and the
  model's predicted masks side-by-side
- **Qualitative Assessment**: Visual comparison for model performance evaluation
- **Export Options**: Save and export results for further analysis

> **[Screenshot: The 'Results Gallery' showing a triplet of images (original, ground truth,
> prediction) for a single test sample. A crack is clearly segmented in the prediction.]**

### 8.2. Metrics and Analysis

- **Quantitative Metrics**: Review detailed performance metrics like IoU, Dice Score,
  Precision, and Recall
- **TensorBoard Integration**: Dive deeper into the training history, analyze learning curves,
  and inspect model graphs using the embedded TensorBoard interface
- **Performance Analysis**: Comprehensive analysis tools for model evaluation

> **[Screenshot: The 'TensorBoard' tab on the 'Results' page, displaying loss curves and
> IoU metrics over training epochs.]**

---

## 9. Best Practices

### 9.1. Configuration Management

- **Use Verified Configurations**: Start with `basic_verification.yaml` for testing
- **Save Custom Configurations**: Use the save feature to preserve your experiments
- **Validate Before Training**: Always check the validation panel before starting training

### 9.2. Training Workflow

- **Start Small**: Begin with small experiments to verify setup
- **Monitor Resources**: Watch GPU memory usage and adjust batch sizes accordingly
- **Save Checkpoints**: Regular checkpointing prevents loss of progress

### 9.3. Results Analysis

- **Compare Multiple Runs**: Use the results page to compare different experiments
- **Export Data**: Save important results for external analysis
- **Document Experiments**: Keep notes on configuration changes and results

---

## 10. Troubleshooting

### 10.1. Common Issues

- **GUI Not Starting**: Ensure Streamlit is installed and conda environment is activated
- **Configuration Errors**: Check YAML syntax and component availability
- **Training Failures**: Verify package installation with `python -c "import crackseg"`
- **Memory Issues**: Reduce batch size or use CPU training

### 10.2. Getting Help

- **CLI Alternative**: Use the verified CLI tutorials for reliable workflows
- **Documentation**: Check the project documentation and guides
- **Quality Gates**: Run `black .`, `python -m ruff . --fix`, `basedpyright .` to verify code

---

## 11. Integration with CLI Workflows

The GUI complements the CLI workflows. For production use and automation:

- **Use CLI for Scripting**: Automated experiments and batch processing
- **Use GUI for Exploration**: Interactive configuration and visualization
- **Hybrid Approach**: Configure in GUI, execute via CLI

### 11.1. Recommended Workflow

1. **Start with CLI Tutorials**: Follow the verified CLI tutorials first
2. **Use GUI for Exploration**: Experiment with configurations interactively
3. **Return to CLI for Production**: Use CLI for final experiments and automation

---

**This guide provides a comprehensive overview of the CrackSeg GUI. For verified workflows
and production use, we recommend following the CLI tutorials in `docs/tutorials/cli/`.**
