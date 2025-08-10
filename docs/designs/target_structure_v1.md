<!-- markdownlint-disable-file -->
# Target Structure v1 — CrackSeg

This document proposes a clean, modular, and production-ready target structure for CrackSeg, aligned with project standards (coding-standards, ml-pytorch-standards, development-workflow) and constraints (English-only files, quality gates, proper placement, and consolidation to avoid redundancy).

## Objectives
- Improve maintainability and discoverability
- Reduce coupling, clarify layering, and isolate responsibilities
- Align Hydra configuration paths and registries with code structure
- Prepare for incremental migration with minimal risk

## Core Principles
- Single responsibility per module (200–300 lines preferred, <400 max)
- Clear layers: data → model → training → evaluation → reporting → presentation (GUI)
- Absolute imports within `src/crackseg/...`
- One registry system for model components (avoid duplication)
- Config-first: all runtime parameters in `configs/` (Hydra)

## Current Gaps (high-level)
- Two registry patterns under `src/crackseg/model/architectures/registry.py` and `src/crackseg/model/factory/registry.py` (consolidate into one)
- `src/training_pipeline/` sits outside the `crackseg/` package (relocate or merge into `crackseg/training/`)
- Mixed configuration helpers exist across `src/crackseg/utils/config/` and GUI-specific config utils (`gui/utils/config/`); define boundaries and avoid cross-layer imports
- Legacy or backup files inside `src/` (e.g., `*.backup`) should be moved to `artifacts/archive/` or removed per deletion plan

## Target Structure (proposed)
```
crackseg/                          # Project root
├── src/
│   └── crackseg/
│       ├── data/
│       │   ├── datasets/
│       │   ├── loaders/
│       │   ├── transforms/
│       │   ├── utils/
│       │   └── validation/
│       ├── model/
│       │   ├── architectures/     # High-level models (UNet variants, etc.)
│       │   ├── encoder/
│       │   ├── decoder/
│       │   ├── bottleneck/
│       │   ├── components/        # Attention blocks, ConvLSTM, ASPP, etc.
│       │   ├── common/
│       │   ├── factory/
│       │   │   ├── registry.py    # Single, canonical registry implementation
│       │   │   └── factory.py     # Unified factory using the registry
│       │   └── config/
│       ├── training/
│       │   ├── losses/
│       │   ├── optimizers/
│       │   ├── components/
│       │   ├── metrics.py
│       │   ├── trainer.py
│       │   └── batch_processing.py
│       ├── evaluation/
│       │   ├── cli/
│       │   ├── core/
│       │   ├── metrics/
│       │   └── visualization/
│       ├── reporting/
│       │   ├── performance/
│       │   ├── recommendations/
│       │   └── templates/
│       ├── utils/
│       │   ├── artifact_manager/
│       │   ├── checkpointing/
│       │   ├── config/            # Hydra helpers, validation, schema
│       │   ├── deployment/
│       │   ├── experiment/
│       │   ├── integrity/
│       │   ├── logging/
│       │   ├── monitoring/
│       │   ├── traceability/
│       │   └── visualization/
│       └── main.py                # Training entry (imported by run.py)
├── gui/                            # Presentation layer (Streamlit)
├── configs/                        # Hydra config tree (already structured)
├── tests/                          # Mirrors src/ structure
└── run.py                          # Hydra entrypoint wrapper
```

Notes:
- `src/training_pipeline/` merges into `src/crackseg/training/` as helpers where appropriate
- Remove `src/crackseg/model/architectures/registry.py` in favor of `model/factory/registry.py`
- Keep GUI config utilities under `gui/utils/config/` and avoid importing GUI from `src/crackseg/*`

## Mapping: Current → Target
- `src/training_pipeline/*` → `src/crackseg/training/{data_loading,model_creation,training_setup,checkpoint_manager}.py` (consolidated where sensible)
- `src/crackseg/model/architectures/registry.py` → deprecated; use `src/crackseg/model/factory/registry.py`
- Legacy backups `*.backup` inside `src/` → `artifacts/archive/` (or removal if superseded)
- Keep `src/main.py` but ensure `run.py` is the canonical launcher; `main.py` stays as orchestrator

## Import & Layering Rules
- Lower layers must not import from upper layers:
  - data → model → training → evaluation → reporting → gui
- `utils/config` is infra-level; allow usage by data/model/training/evaluation/reporting; disallow usage by `gui` to reach back into core (GUI can call public APIs only)
- One registry system exposed via `crackseg.model.factory.registry`

## Hydra Alignment
- Keep configs in `configs/` with composition; confirm `run.py` uses `@hydra_main(config_path="configs")`
- Ensure `src/crackseg/utils/config/init.py` keeps CWD-relative resolution stable
- Validate all experiments under `configs/experiments/` execute via `run.py`

## Incremental Migration Plan
Phase 1 (No behavior change):
- Move `src/training_pipeline/*` into `src/crackseg/training/` keeping the same public functions
- Replace imports where necessary; run tests and quality gates
- Delete old paths after CI passes

Phase 2 (Registry consolidation):
- Remove `model/architectures/registry.py`; update any references to `model/factory/registry.py`
- Add unit tests covering registry usage across encoders/decoders/bottlenecks

Phase 3 (Public API hardening):
- Define minimal public imports in package `__init__.py` files
- Enforce layering with import lints (optional: custom ruff rules or static checks)

Phase 4 (Docs & Examples):
- Update developer guides and architecture docs reflecting the final layout
- Regenerate project tree and dependency report

## Risks & Rollback
- Broken imports during moves → Mitigation: move-first with compatibility shims; verify with `pytest` and type checks
- Hydra path resolution drift → Mitigation: keep `run.py` as the canonical entry; add an integration test for experiment configs
- Registry mismatch → Mitigation: add tests to ensure all components register correctly after consolidation
Rollback Strategy: revert the moves per phase if CI fails; phases are independent

## Acceptance Criteria
- All imports resolved; no circular dependencies introduced (verified by dependency report)
- Tests green; quality gates pass (`ruff`, `black`, `basedpyright`)
- `run.py` executes recommended experiment configs successfully
- Project tree and dependency report updated and committed


