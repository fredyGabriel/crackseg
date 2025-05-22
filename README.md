# Pavement Crack Segmentation Project

> **Note:** This project was developed with the assistance of AI tools.

## Overview

Deep learning for semantic segmentation of cracks in asphalt pavement.
Modular, reproducible, and extensible codebase for research and production.

## Quickstart

1. **Install environment**

   ```bash
   conda env create -f environment.yml
   conda activate torch
   cp .env.example .env  # Edit as needed
   ```

2. **Train a model**

   ```bash
   python run.py
   ```

3. **Evaluate**

   ```bash
   python src/evaluate.py
   ```

- For a detailed workflow and configuration options, see [WORKFLOW_TRAINING.md](docs/guides/WORKFLOW_TRAINING.md).

## Project Structure

- `src/` — Core code (models, data, training)
- `configs/` — Hydra YAML configs (modular, see below)
- `outputs/` — Results, logs, checkpoints
- `tests/` — Unit/integration tests
- `scripts/` — Utilities and experiment scripts
- `tasks/` — TaskMaster task files

> Note: Scripts in `scripts/` are for experimentation and utilities only. Do not import them in core modules. Clean up temporary files like `__pycache__` regularly.

## Configuration

- All model, data, and training options are set via YAML files in `configs/`.
- **Architectures:** see `configs/model/architectures/`
- **Losses, metrics, schedulers:** see `configs/training/`
- **Data splits, batch size:** see `configs/data/`
- Combine components by editing `configs/config.yaml` or via CLI overrides.

## Code Quality & Linting

El proyecto utiliza las siguientes herramientas para mantener la calidad del código:

- **Black**: Formateo automático de código.
- **Ruff**: Linter y verificación de estilo.
- **Mypy**: Verificación de tipos estáticos.

El hook de pre-commit ejecuta automáticamente estas herramientas sobre los archivos Python staged.

Asegúrate de que estas herramientas estén instaladas en tu entorno (ya incluidas en `environment.yml`).

Configuración:

- Configuración centralizada: `configs/linting/config.yaml`
- Configuración de herramientas: `pyproject.toml`

Para instalar y activar el hook de pre-commit (opcional):

```bash
conda install -c conda-forge pre-commit
pre-commit install
```

For a detailed plan on code quality improvements, see [docs/LINTING_PLAN.md](docs/LINTING_PLAN.md).

For a comprehensive report on linting optimization, see [docs/reports/LINTING_REPORT.md](docs/reports/LINTING_REPORT.md).

## Training & Workflow

- See [WORKFLOW_TRAINING.md](docs/guides/WORKFLOW_TRAINING.md) for a step-by-step guide, including:
  - Environment setup
  - Data configuration
  - Model/component selection
  - Training and evaluation
  - Example minimal config

## Testing

- Run all tests:

  ```bash
  pytest
  ```

- See `tests/README.md` for details.

## Environment Variables

- Copy `.env.example` to `.env` and fill required values.
- Main variables:
  - `ANTHROPIC_API_KEY`: API key for Anthropic Claude (Task Master)
  - `DEBUG`: Enable/disable debug mode (`true` or `false`)

## Dependency Management

- Check for updates:

  ```bash
  python scripts/utils/check_updates.py
  ```

- Update environment:

  ```bash
  conda env update -f environment.yml --prune
  ```

## Contributing

- See [CONTRIBUTING.md](docs/guides/CONTRIBUTING.md) and follow coding guidelines.
- Add/update tests for your changes.
- Update documentation as needed.

## License

MIT License. See `LICENSE`.

---

**Tips:**

- Use `run.py` as the main entry point for training.
- All configuration is modular and can be overridden via Hydra.
- For advanced usage, see the documentation in each `configs/` subfolder.
