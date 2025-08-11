<!-- markdownlint-disable-file -->
# CI Guardrails — Oversized Files & Smoke Tests

This document describes CI checks to keep refactors safe:

## Checks
- Line limit guardrail: fail if any file > 400 lines; warn if > 300 (report artifact)
- Hydra smoke: `python run.py --config-name=basic_verification`
- Dependency & alignment: regenerate `dependency_graph.md` and `hydra_registry_alignment.md` after P1 changes

## Artifacts
- `docs/reports/analysis-reports/architecture/line_limit_guardrail.md`
- `docs/reports/analysis-reports/architecture/dependency_graph.md`
- `docs/reports/analysis-reports/architecture/hydra_registry_alignment.md`


