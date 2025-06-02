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
- `docs/reports/` — **Organized project reports and analysis** (see [Reports](#reports))

> Note: Scripts in `scripts/` are for experimentation and utilities only. Do not import them in core modules. Clean up temporary files like `__pycache__` regularly.

## Reports

All project reports, analysis, and documentation are now organized in `docs/reports/` with the following structure:

- **`testing/`** — Test coverage reports, improvement plans, and testing priorities
- **`coverage/`** — Code coverage analysis and gap reports
- **`tasks/`** — Task completion summaries and complexity analysis
- **`models/`** — Model architecture analysis and import catalogs
- **`project/`** — Project-level plans and verification reports
- **`archive/`** — Historical reports and deprecated documentation

📊 **Current Metrics:** 66% test coverage (up from 25%), 866 tests implemented

For a complete index and navigation guide, see [`docs/reports/README.md`](docs/reports/README.md).

### Report Organization Tool

Use the automated report organizer to maintain structure:

```bash
# Check current organization
python scripts/utils/organize_reports.py --report

# Organize scattered reports (dry run)
python scripts/utils/organize_reports.py --dry-run

# Actually organize reports
python scripts/utils/organize_reports.py
```

## Configuration

- All model, data, and training options are set via YAML files in `configs/`.
- **Architectures:** see `configs/model/architectures/`
- **Losses, metrics, schedulers:** see `configs/training/`
- **Data splits, batch size:** see `configs/data/`
- Combine components by editing `configs/config.yaml` or via CLI overrides.

## Code Quality & Linting

The project uses the following tools to maintain code quality:

- **Black**: Automatic code formatting.
- **Ruff**: Linter and style checking.
- **Basedpyright**: Static type checking (replaces mypy in the main workflow).

The pre-commit hook automatically runs these tools on staged Python files.

Make sure these tools are installed in your environment (already included in `environment.yml`).

Configuration:

- Centralized configuration: `configs/linting/config.yaml`
- Tool configuration: `pyproject.toml`

To install and activate the pre-commit hook (optional):

```bash
conda install -c conda-forge pre-commit
pre-commit install
```

The pre-commit workflow runs:

```bash
black <files>
ruff check <files>
basedpyright <files>
```

If you want to continue using mypy for additional analysis, you can run it manually:

```bash
mypy .
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
- **Current Coverage:** 66% (5,333/8,065 lines) — see [`docs/reports/coverage/`](docs/reports/coverage/) for detailed analysis

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
- Check [`docs/reports/`](docs/reports/) for the latest project analysis and metrics.

## Requirements

This project requires **Python 3.12** or higher. All dependencies and tools are configured and tested for Python 3.12. Please ensure your environment matches this version for full compatibility.
