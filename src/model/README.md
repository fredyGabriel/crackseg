# src/model

This directory contains the core model architectures and related utilities for the pavement crack segmentation project. The code is organized to maximize modularity, reusability, and clarity, following the project's coding standards.

## Structure

- **bottleneck/**: Building blocks for bottleneck layers used in encoder/decoder architectures.
- **decoder/**: Decoder modules for segmentation models.
- **encoder/**: Encoder modules for segmentation models.
- **utils.py**: Utility functions for model analysis, parameter counting, receptive field estimation, memory usage, and architecture visualization (Graphviz).
- **unet.py**: Main UNet and BaseUNet class definitions, using modular blocks and utilities.

## Key Components

### 1. BaseUNet & UNet Classes (`unet.py`)
- Implements the core UNet architecture.
- Uses modular encoder, decoder, and bottleneck blocks.
- Delegates parameter counting, receptive field estimation, and architecture visualization to utility functions in `utils.py`.
- Designed for easy extension and integration with Hydra-based configuration.

### 2. Utilities (`utils.py`)
- **Parameter Counting:** Functions to count total and trainable parameters of a model.
- **Receptive Field Estimation:** Estimate the receptive field for each layer/block.
- **Memory Usage Estimation:** Approximate memory requirements for a given input size.
- **Layer Hierarchy Extraction:** Retrieve a structured summary of the model's layers.
- **Architecture Visualization:** Generate a Graphviz diagram of the model architecture.

All utility functions are documented with concise docstrings and are tested independently.

## Usage

- **Importing Models:**
  ```python
  from src.model.unet import UNet
  ```
- **Using Utilities:**
  ```python
  from src.model.utils import count_parameters, render_unet_architecture_diagram
  ```

- **Visualization Example:**
  ```python
  model = UNet(...)
  render_unet_architecture_diagram(model, output_path="unet_architecture.png")
  ```

## Testing

- All major components and utilities are covered by unit tests in `tests/model/`.
- Utilities are tested with mocks to ensure independence from specific model implementations.

## Extending

- To add new architectures, create a new file or submodule following the modular pattern.
- Place reusable blocks in the appropriate subdirectory (`encoder/`, `decoder/`, `bottleneck/`).
- Add new utilities to `utils.py` if they are of general use.
- Update or add tests as needed.

## Dependencies

- All dependencies are managed via Conda (`environment.yml`) and pip (`requirements.txt`).
- Visualization requires `graphviz` (ensure both Python package and system binaries are installed).

## Conventions

- Follow PEP8 and project coding guidelines.
- Keep docstrings minimal and in English.
- Do not hardcode configuration; use Hydra for all parameters.
- Do not introduce new files or directories without explicit confirmation and documentation updates.

---

For more details, see the [project-structure.mdc](../../.cursor/guides/project-structure.mdc) and [development-guide.mdc](../../.cursor/guides/development-guide.mdc). 