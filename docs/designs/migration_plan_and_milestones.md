<!-- markdownlint-disable-file -->
# Migration Plan and Milestones — CrackSeg

This plan orchestrates the incremental migration from current structure to the Target Structure v1 while preserving behavior, minimizing risk, and maintaining green quality gates.

## Scope
- Code moves only (no feature changes)
- Registry consolidation (single canonical registry)
- Import path updates
- Documentation refresh

## Phase 1 — Training Pipeline Consolidation (No behavior change)
Goals:
- Move `src/training_pipeline/` helpers into `src/crackseg/training/` keeping public function signatures
- Update imports in `run.py` and `src/main.py`

Tasks:
- Create new modules under `src/crackseg/training/`:
  - `data_loading.py`, `model_creation.py`, `training_setup.py`, `checkpoint_manager.py`
- Move code from `src/training_pipeline/` and adapt absolute imports
- Add compatibility re-exports (temporary) if needed

Verification:
- `ruff`, `black`, `basedpyright` pass
- `pytest -q` green for unit/integration
- `run.py --config-name=basic_verification` executes

Rollback:
- Revert move commits; retain originals under `src/training_pipeline/`

## Phase 2 — Registry Consolidation
Goals:
- Use a single registry at `src/crackseg/model/factory/registry.py`
- Remove `src/crackseg/model/architectures/registry.py`

Tasks:
- Search usages of `architectures/registry.py` and switch to factory registry
- Add tests ensuring registration for encoders/decoders/bottlenecks/architectures

Verification:
- Unit tests for registry pass; no duplicate registrations
- Dependency graph shows no new cycles

Rollback:
- Restore removed registry file; revert import changes

## Phase 3 — Public API Hardening & Layering Rules
Goals:
- Define minimal public interfaces via package `__init__.py`
- Enforce import directions (data→model→training→evaluation→reporting→gui)

Tasks:
- Add explicit `__all__` in key packages
- Optional: configure lint to detect reversed imports

Verification:
- Dependency report shows no cross-layer violations

Rollback:
- Relax `__init__` exports and revert lint rules

## Phase 4 — Documentation & Reports Update
Goals:
- Align docs and diagrams with final structure

Tasks:
- Update `docs/guides/developer-guides` and `technical-specs`
- Regenerate `docs/reports/project_tree.md` and dependency report

Verification:
- Docs build without broken links (MkDocs if applicable)

Rollback:
- Revert docs-only commits

## Milestones & Acceptance Criteria
- M1 (Phase 1 complete): Training runs via `run.py` using new paths; tests & gates green
- M2 (Phase 2 complete): Single registry in use; tests validating registration
- M3 (Phase 3 complete): No layering violations; stable public API
- M4 (Phase 4 complete): Updated docs, tree, and analysis reports

## Risks & Mitigations
- Import breakage during moves → Move with re-exports, run CI per phase
- Hydra path drift → Keep `run.py` canonical and test configs from `configs/experiments/`
- Hidden dependencies in legacy files → Use dependency report to locate and fix

## Testing Strategy
- Unit tests mirroring moved modules
- Integration tests for training loop and config loading
- Smoke runs for recommended experiment configs

## Change Management
- Use chore/refactor branches per phase
- Conventional commits and PRs with checklists
- CI gates: `ruff`, `black`, `basedpyright`, `pytest`


