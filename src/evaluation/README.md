# Evaluation Module

This directory contains all core logic and helpers for model evaluation, result aggregation, and ensemble analysis.

## Purpose
- Centralizes the evaluation workflow, including metrics computation, result loading, ensemble methods, and experiment setup.
- Promotes modularity and separation of concerns for maintainable and extensible evaluation code.

## File Overview
- `core.py`: Core evaluation logic (metrics computation, evaluation loops, etc.).
- `ensemble.py`: Ensemble methods for combining predictions or models.
- `loading.py`: Functions for loading predictions, results, or evaluation data.
- `results.py`: Helpers for aggregating, formatting, or saving evaluation results.
- `data.py`: Data manipulation utilities specific to evaluation.
- `setup.py`: Experiment or evaluation pipeline setup/configuration.
- `__main__.py`: CLI or main script entry point for running evaluation workflows.
- `__init__.py`: Module initialization.

## Conventions
- All configuration is loaded via Hydra/OmegaConf YAML files when applicable.
- No hardcoded evaluation parameters: use config files or CLI arguments.
- All new metrics or ensemble methods should be stateless and registered in their respective modules.
- Use helpers/utilities from `src/utils/` for logging, configuration, and shared logic.

## Extending
- To add a new metric or evaluation method: implement the function in `core.py` or a new module and register it.
- To add new ensemble logic: implement in `ensemble.py` and document usage.
- To add new data loaders or result handlers: extend `loading.py` or `results.py` as needed.

## Related
- See the main project README for high-level usage and configuration patterns.
- See `src/utils/` for shared utilities (logging, configuration, etc). 