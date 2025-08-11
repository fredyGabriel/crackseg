# Public API and Layering Rules

This document defines the project-wide import boundaries and public interfaces to keep the codebase
modular, testable, and easy to evolve.

## Layering Boundaries

- Model layer must not import from Training layer.
  - Forbidden: `from crackseg.training ...` inside `crackseg.model`.
- Model and Training may import from Utils, but only through public shims when provided.
- Evaluation may import from Model and Utils; avoid importing from Training.
- GUI and Scripts should consume only public APIs from packages; avoid deep private imports.

## Public Shims (Stable Entry Points)

- Storage (configs + checkpoints):
  - Use: `from crackseg.utils.storage import ...`
  - Avoid: `crackseg.utils.checkpointing.*`, `crackseg.utils.config.standardized_storage`

- Reporting (results, experiment data, validation reports):
  - Use: `from crackseg.utils.reporting import ...`
  - Avoid: `crackseg.evaluation.utils.results`, `crackseg.utils.experiment_saver`, `crackseg.utils.deployment.validation.reporting.core`

- Model Factory:
  - Use: `from crackseg.model.factory import create_unet, validate_config`
  - Deep symbols are re-exported via shims to keep imports stable.

## Package Exports

- All packages expose their public surface via `__all__` in their `__init__.py` or shims.
- New modules must add exports explicitly when intended for public use.

## CI Guardrail

A simple static checker (`layering_rules_check.py`) scans imports in `src/crackseg/` and fails on:

- Model → Training imports
- Use of internal storage/reporting modules bypassing shims

Run locally:

```bash
python scripts/utils/quality/guardrails/layering_rules_check.py
```
