<!-- markdownlint-disable-file -->
# Deletion Candidates (initial validation)

Sources: `file_inventory`, dependency graph, duplicate/unused scan, link checker

Guardrails status:
- Link checker (docs + infrastructure): 0 errors, 0 warnings
- Dependency graph (src/*): candidates below do not affect src import graph except explicitly noted
- Line limit: unaffected by asset/docs deletions

## A. Duplicated assets (safe after relinking)

1) Duplicate logos
- Keep: `gui/assets/images/logos/primary-logo.png`
- Remove: `docs/designs/logo.png`
- Preconditions: update any doc references to point to the kept path (link checker clean run confirms)
- Risk: low; Rollback: restore file

2) Legacy duplicated plots (same content/different timestamp)
- Paths:
  - `docs/reports/experiment-reports/plots/legacy/training_curves_20250724_081112.png`
  - `docs/reports/experiment-reports/plots/legacy/training_curves_20250724_081136.png`
- Action: keep one canonical; update references; delete the other
- Risk: low; Rollback: restore file

## B. Obvious backups / generated artifacts

- Remove: `src/crackseg/training/trainer.py.backup` (backup file not used by code/tests)
- Risk: low; Checks: grep usage = none, imports = none

## C. Potentially unnecessary __init__.py (needs careful check)

- Candidates (from duplicate scan):
  - `tests/unit/__init__.py`
  - `tests/unit/gui/__init__.py`
  - Review needed: `gui/__init__.py`, `gui/services/__init__.py` (could affect streamlit app packaging/imports)
- Plan: remove only test package `__init__.py` after verifying test discovery unaffected; defer GUI `__init__.py` to dedicated ticket

## D. Legacy reports/catalogs

- Candidate: `docs/reports/model-reports/analysis/legacy/model_imports_catalog.json` (stale, superseded by current reports)
- Action: move to `docs/reports/model-reports/analysis/legacy/` or delete if duplicated; verify no links

## E. Language compliance cleanup (non-blocking deletions)

- Consolidate Spanish-only legacy docs by archiving under `docs/legacy/` or adding banner; avoid deletion if referenced

---

Next steps:
- Update `docs/reports/cleanup/deletion_plan.md` with these items and owners
- Batch 1 (low risk): A(1), A(2), B, tests `__init__.py`
- Validate with link checker and quick grep before deleting; commit per-batch with rollback notes
