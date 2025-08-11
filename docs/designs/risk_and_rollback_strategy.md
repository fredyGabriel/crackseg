<!-- markdownlint-disable-file -->
# Risk and Rollback Strategy — CrackSeg Refactor

This document outlines risks, mitigations, and rollback actions for the structural refactor phases described in Target Structure v1 and Migration Plan.

## Key Risks
- Import breakage during file moves
- Hydra config path drift (config discovery, overrides)
- Registry inconsistency (duplicate or missing registrations)
- Latent circular dependencies introduced by new imports
- Hidden legacy dependencies (backup/legacy modules used indirectly)

## Mitigations
- Phase-by-phase PRs with CI gates (ruff, black, basedpyright, pytest)
- Temporary re-exports for compatibility during moves
- Dependency report regeneration after each phase
- Smoke runs of recommended experiment configs via `run.py`
- Unit tests for registries and config initialization

## Rollback Strategy
- Each phase is isolated; revert the phase branch if CI fails
- Restore previous module paths (git revert) and remove re-exports last
- Keep migration commits atomic to simplify revert

## Smoke Tests (Per Phase)
- `python run.py --config-name=basic_verification`
- `python run.py --config-path=configs --config-name=experiments/swinv2_hybrid/swinv2_360x360_corrected`

## Acceptance Gate
- All quality gates green
- Dependency and Hydra–Registry reports regenerated without regressions
- No new cycles; imports resolve from `run.py`


