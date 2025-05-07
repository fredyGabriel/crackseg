# Pavement Crack Segmentation Project

> **Note:** This project was developed with the assistance of artificial intelligence tools (AI-assisted code generation and documentation).

## Project Overview

This project explores and develops state-of-the-art (SOTA) deep learning models for semantic segmentation of cracks in asphalt pavement, with a focus on modular U-Net-based architectures. The main goal is to identify the best-performing configuration for crack detection using public and custom datasets, leveraging a flexible, reproducible, and well-documented codebase. The approach includes experimenting with CNNs, Swin Transformer V2, hybrid models, and attention mechanisms, all orchestrated via Hydra configuration and Conda environments. The project is intended for researchers, engineers, and practitioners seeking robust, automated pavement crack analysis.

## Project Final Status

- The project is fully functional, robust, and production-ready.
- All unit and integration tests pass successfully; obsolete or redundant tests have been removed after refactoring.
- The codebase is modular, clean, and follows best practices for maintainability and reproducibility.
- Configuration, training, evaluation, and experiment management are fully separated and documented.
- The environment is managed via Conda and all dependencies are versioned.

## Project Structure

- `src/` — Main source code (models, data pipeline, training, utils)
- `tests/` — Unit and integration tests (see `tests/README.md` for details)
- `configs/` — Hydra YAML configuration files for all modules
- `scripts/` — Utilities, experiment scripts, and examples (not imported by core code)
- `outputs/` — Experiment results, logs, and checkpoints
- `tasks/` — TaskMaster task files

> Note: Scripts in `scripts/` are for experimentation and utilities only. Do not import them in core modules. Clean up temporary files like `__pycache__` regularly.

## Training Workflow

The training process is fully modular and managed by the `Trainer` class. The main script (`src/main.py`) delegates all training logic to this class, ensuring clear separation of concerns and easy maintenance.

**Key features:**
- **Trainer orchestration:** All training, validation, and checkpointing logic is in `src/training/trainer.py`.
- **Checkpointing:** Best and last model states are saved automatically. Resume training from any checkpoint via Hydra config (`resume_from_checkpoint`).
- **Early stopping:** Configurable via Hydra YAML (`early_stopping`).
- **Hydra configuration:** All parameters (epochs, optimizer, scheduler, checkpointing, early stopping, etc.) are managed via YAML in `configs/training/`.
- **No duplicate logic:** All legacy training code has been removed from `main.py`.

**To train:**
```bash
python src/main.py
```

**To resume from a checkpoint:**
- Edit your Hydra config (e.g., `configs/training/trainer.yaml`) and set the path in `training.checkpoints.resume_from_checkpoint`.

**For details:**
- See `src/training/trainer.py` for orchestration logic.
- See `configs/training/trainer.yaml` for all configurable options.
- See `tests/training/` for integration and unit tests of the training flow.

## Evaluation Workflow

Final evaluation is performed using a dedicated script:

```bash
python src/evaluate.py
```

- This script loads the best or last checkpoint and computes metrics on the test set.
- Configuration (paths, metrics, etc.) is managed via Hydra YAML, just like training.
- See `src/evaluate.py` and `configs/evaluation/` for details.

**Why this separation?**
- Ensures a clean, modular workflow and avoids mixing training and evaluation logic.
- Facilitates experiment automation and code maintenance.

## Testing

- All tests are located in the `tests/` directory and are organized into unit and integration tests.
- The suite covers all main flows and edge cases for data, model, training, and evaluation.
- See `tests/README.md` for details on running and organizing tests.

## How to Contribute

- Please read the guidelines in `CONTRIBUTING.md` before submitting a pull request.
- Follow the coding style and modularity guidelines (see `coding-preferences.mdc`).
- Add or update tests for your changes.
- Update documentation as needed.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Conda Environment

This project uses a Conda environment named `torch`.

**To activate:**
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

Sensitive configuration is managed via environment variables. See `.env.example` for a template.

- Copy `.env.example` to `.env` and fill in the required values.
- Never commit your real `.env` file to the repository.

Main variables:
- `ANTHROPIC_API_KEY`: API key for Anthropic Claude (Task Master)
- `DEBUG`: Enable/disable debug mode (`true` or `false`)

## Dependency Management

### Update Verification

A script is included to check for updates to main dependencies:

```bash
python scripts/utils/check_updates.py
```

This script:
- Checks current versions in `environment.yml`
- Compares with latest versions on conda-forge and PyPI
- Shows an update report

### Updating Dependencies

1. Run the verification script
2. Update versions in `environment.yml` as needed
3. Apply updates:
   ```bash
   conda env update -f environment.yml --prune
   ```
4. Verify compatibility by running the tests:
   ```bash
   pytest
   ```

### Update Considerations

- Maintain compatible versions of PyTorch and torchvision
- Check CUDA compatibility if using GPU
- Document significant changes in `CHANGELOG.md`
- Run thorough tests after updating critical dependencies

## How to Train, Monitor, and Visualize Results

### 1. Training

To start a training run with the default configuration:
```bash
python src/main.py
```
- All training parameters (model, optimizer, scheduler, etc.) are managed via Hydra YAML files in `configs/`.
- To resume from a checkpoint, set the path in your Hydra config (e.g., `configs/training/trainer.yaml` → `training.checkpoints.resume_from_checkpoint`).

### 2. Monitoring Training
- **Logs:** Training and validation progress, losses, and metrics are logged to the console and to log files in `outputs/` and `outputs/experiments/<experiment_id>/logs/`.
- **Checkpoints:** Model checkpoints (best and last) are saved in `outputs/experiments/<experiment_id>/checkpoints/`.
- **Hydra Output:** Hydra creates a timestamped folder for each run in `outputs/experiments/`.

### 3. Visualizing and Accessing Results
- **Numerical Results:**
  - Metrics (IoU, F1, Precision, Recall) are saved in `outputs/experiments/<experiment_id>/experiment_info.json` and/or `metrics/` if present.
  - Training/evaluation logs are in `outputs/experiments/<experiment_id>/logs/`.
- **Predictions:**
  - Segmentation masks predicted by the model are saved in `outputs/experiments/<experiment_id>/results/predictions/{test,validation}/`.
- **Visualizations:**
  - Visual comparison images (input, ground truth, prediction) are saved in `outputs/experiments/<experiment_id>/results/visualizations/`.

### 4. Evaluation
To evaluate a trained model:
```bash
python src/evaluate.py
```
- This computes metrics on the test set and saves results as above.

> For more details, see the sections below and the configuration files in `configs/`.

--- 