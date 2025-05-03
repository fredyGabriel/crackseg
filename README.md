# Pavement Crack Segmentation Project

## Project Overview

This project aims to achieve state-of-the-art performance in pavement crack segmentation using modular U-Net-based deep learning architectures. It is designed for researchers, engineers, and practitioners interested in automated road inspection and infrastructure monitoring.

## Project Structure

- `src/` — Main source code (models, data pipeline, training, utils)
- `tests/` — Unit and integration tests
- `configs/` — Hydra YAML configuration files
- `scripts/` — Utility and setup scripts
- `tasks/` — TaskMaster task files

## Basic Usage

After setting up the environment and configuration files, you can run the main training pipeline (example):

```bash
python src/main.py
```

For more details on configuration and advanced usage, see the documentation in the `docs/` folder (if available) or the comments in the configuration files.

## Training Flow

The training process is now fully modular and managed by the `Trainer` class. The main script (`src/main.py`) delegates all training logic to this class, ensuring a clean separation of concerns and easier maintenance.

**Key features of the training flow:**
- **Trainer-based orchestration:** All training, validation, and checkpointing logic is handled by `src/training/trainer.py`.
- **Checkpointing:** The best and last model states are automatically saved during training. You can resume training from any checkpoint by setting the appropriate configuration in Hydra (`resume_from_checkpoint`).
- **Early stopping:** Training can be stopped early based on validation metrics, as configured in the Hydra YAML files (`early_stopping` section).
- **Hydra configuration:** All parameters (epochs, optimizer, scheduler, checkpointing, early stopping, etc.) are managed via YAML files in `configs/training/`.
- **No duplicate logic:** All legacy training code has been removed from `main.py`.

**To run training:**
```bash
python src/main.py
```

**To resume from a checkpoint:**
- Edit your Hydra config (e.g., `configs/training/trainer.yaml`) and set the path in `training.checkpoints.resume_from_checkpoint`.

**For more details:**
- See `src/training/trainer.py` for the orchestration logic.
- See `configs/training/trainer.yaml` for all configurable options.
- See the `tests/training/` folder for integration and unit tests covering the training flow.

## Evaluation Flow

The final evaluation is no longer performed in `main.py`. To evaluate your trained model on the test set, use the dedicated evaluation script:

```bash
python src/evaluate.py
```

- This script loads the best or last checkpoint and computes metrics on the test set.
- Configuration (paths, metrics, etc.) is managed via Hydra YAML files, just like training.
- See `src/evaluate.py` and `configs/evaluation/` for details.

**Why this change?**
- This separation ensures a clean, modular workflow and avoids mixing training and evaluation logic in the same script.
- It also makes it easier to run evaluation independently, automate experiments, and maintain the codebase.

## How to Contribute

- Please read the guidelines in `CONTRIBUTING.md` before submitting a pull request.
- Follow the coding style and modularity guidelines (see `coding-preferences.mdc`).
- Add or update tests for your changes.
- Update documentation as needed.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Conda Environment

This project uses a Conda environment named `torch`.

**To activate it:**
```bash
conda activate torch
```

**To install additional dependencies:**
```bash
conda install <package>
```

**To reproduce the environment:**
```bash
conda env create -f environment.yml
```

## Environment Variables

This project uses environment variables for sensitive configuration.  
See the example file: `.env.example`

- Copy `.env.example` to `.env` and fill in the required values.
- Never commit your real `.env` file to the repository.

Main variables:
- `ANTHROPIC_API_KEY`: API key for Anthropic Claude (Task Master)
- `DEBUG`: Enable/disable debug mode (`true` or `false`)

## Dependency Management

### Update Verification

The project includes a script to verify available updates for main dependencies:

```bash
python scripts/check_updates.py
```

This script:
- Verifies current versions in environment.yml
- Compares with latest versions available on conda-forge and PyPI
- Shows an update report

### Updating Dependencies

To update dependencies:

1. Run the verification script
2. Update versions in environment.yml as needed
3. Apply the updates:
   ```bash
   conda env update -f environment.yml --prune
   ```
4. Verify compatibility by running tests:
   ```bash
   pytest
   ```

### Update Considerations

- Keep PyTorch and torchvision versions compatible
- Verify CUDA compatibility if using GPU
- Document significant changes in CHANGELOG.md
- Perform thorough testing after updating critical dependencies

--- 