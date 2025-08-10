<!-- markdownlint-disable-file -->
# Deletion plan and impact assessment

This plan consolidates findings from recent audits to propose safe removals or consolidations. All actions follow c1–c4 and will be executed in small PRs with clear rollback.

Sources
- Duplicates/unused: docs/reports/project-reports/technical/duplicates_and_unused_report.md
- Inventory: docs/reports/file_inventory.md
- Artifacts/binaries: docs/reports/project-reports/technical/artifacts_and_binaries_audit.md
- Language: docs/reports/project-reports/documentation/language_compliance_report.md

## Summary of proposed actions

- Duplicates: deduplicate assets (logos, legacy plots) and redundant init files
- Unused modules: validate high-probability unused under `evaluation/visualization/legacy/`, `reporting/`, and deployment helpers
- Artifacts: ensure `.gitignore` coverage; no >50 MB files detected
- Language: translate/merge legacy Spanish docs; archive strictly Spanish variants

## Detailed proposals

### A. Potential duplicate groups (size+hash)

Decision model: keep canonical, remove duplicates; update references; add CI check.

- Group 1
  - `docs/designs/logo.png`
  - `gui/assets/images/logos/primary-logo.png`
  - Action: keep `gui/assets/images/logos/primary-logo.png`; remove `docs/designs/logo.png`; update docs to reference canonical asset
  - Risk: low; Rollback: restore deleted file if referenced in older docs

- Group 2
  - `docs/reports/experiment-reports/plots/legacy/training_curves_20250724_081112.png`
  - `docs/reports/experiment-reports/plots/legacy/training_curves_20250724_081136.png`
  - Action: deduplicate by content; keep one with clearer naming and update links
  - Risk: low; Rollback: keep both if provenance matters

- Group 3
  - `tests/unit/__init__.py`
  - `tests/unit/gui/__init__.py`
  - Review only (defer deletion): `gui/__init__.py`, `gui/services/__init__.py`
  - Action: remove unnecessary test `__init__.py` after verifying test discovery; defer GUI package files to separate ticket
  - Risk: medium (import paths); Rollback: restore removed `__init__.py`

- Group 4
  - `scripts/__init__.py`
  - `tests/__init__.py`
  - Action: treat similarly to Group 3; remove unnecessary `__init__.py` if not needed for namespace
  - Risk: medium; Rollback: restore file if discovery/imports break

### B. Unused Python modules under `src/` (n=89)

Decision model: confirm unused via grep/tests; delete or integrate. Prioritize legacy visualization/reporting and deployment helpers.

- High-probability unused categories (examples)
  - `src/crackseg/evaluation/visualization/legacy/*`
  - `src/crackseg/reporting/*` and `src/crackseg/utils/deployment/*` subsets
  - Actions: create per-module tickets; run quick import/usage checks; remove if truly dead; otherwise move to `docs/legacy-snippets/` or mark deprecated
  - Risk: medium (hidden imports in configs/tests); Rollback: revert deletion; add deprecation period

### C. Backup and generated files

- Remove `src/crackseg/training/trainer.py.backup` (not referenced by imports/tests)
- Risk: low; Rollback: restore if needed

### C. Artifacts and binaries

- Status: no files >50 MB detected; artifacts largely covered by `.gitignore`
- Action: `.gitignore` already updated with missing entries; re-run audit in CI weekly
- Risk: low; Rollback: n/a

### D. Language compliance

- Actions
  - Translate or consolidate Spanish docs flagged in `language_compliance_report.md`
  - Archive Spanish-only legacy docs under `docs/legacy/` if not needed
  - Ensure README/active guides are English-only
- Risk: low; Rollback: keep originals in `docs/legacy/`

## Impact and verification

- Affected areas: docs links, GUI assets, package discovery (`__init__.py`), optional visualization/reporting modules
- Verification
  - Run full quality gates and smoke tests
  - Import graph check for removed `__init__.py`
  - Broken-link checker across docs
  - Grep for removed paths before merge

## Rollback strategy

- Each change in separate PR with clear scope
- Tag PRs with `cleanup` and link to this plan
- If issues arise, revert PR entirely (single-commit PRs preferred)

## Next steps (tickets)

1. Dedupe logos (Group 1)
2. Dedupe legacy plots (Group 2)
3. Prune unnecessary test `__init__.py` (Group 3)
4. Remove `trainer.py.backup`
5. Validate y remover top-20 unused modules (batch 1)
6. Translate/consolidate top-10 Spanish docs
7. Add CI checks: duplicate guard and link checker (done for links)


