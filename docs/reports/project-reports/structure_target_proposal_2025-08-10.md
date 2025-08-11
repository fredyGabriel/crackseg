# Target Structure & Classification Proposal - 2025-08-10

This proposal defines adjustments to maintain consistency with project structure rules and recent
refactors. It focuses on classification and documentation, not moving code immediately.

## Principles

- Preserve public APIs; avoid breaking imports
- Mirror `src/` structure in `tests/`
- Keep scripts under `scripts/` with `<action>_<target>.py` naming
- Prefer focused modules (≤300 LOC, hard max 400)

## Proposed Adjustments (No-Op/Low-Risk)

1. Documentation stubs in dense dirs
   - Add/update `README.md` in:
     - `src/crackseg/utils/deployment/core/`
     - `src/crackseg/reporting/utils/`
     - `src/crackseg/evaluation/visualization/utils/`
2. Scripts classification
   - Consolidate report generators under `scripts/reports/`
   - Keep maintenance tools under `scripts/maintenance/`
3. Tests alignment (deferred PR)
   - Mirror new `schemas/`, `deployment/core/`, `reporting/utils/`
   - Add smoke tests for visualization/deployment strategies

## Optional Future Moves (Considered, not executed here)

- `scripts/utils/quality/guardrails/` → keep as-is; already aligned to rules
- `src/crackseg/utils/training_data.py` → consider move to `src/crackseg/utils/data/` for thematic grouping

## Guardrails

- Line limit guardrail remains active
- Quality gates must pass for any structural changes

— End —
