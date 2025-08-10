<!-- markdownlint-disable-file -->
# Documentation Consolidation Map

Purpose: Track grouping and canonical paths while consolidating legacy docs.

Canonical Buckets and Pointers

1) Developer Guides (canonical)
- Canonical: `docs/guides/developer-guides/`
- Legacy pointers:
  - `development/legacy/CONTRIBUTING.md` → Root `CONTRIBUTING.md`
  - `development/legacy/gui_development_guidelines.md` → keep in legacy, cross-link from dev guides index
  - `quality/legacy/quality_gates_guide.md` → referenced from README Quality Gates

2) Operational Workflows (canonical)
- Canonical: `docs/guides/operational-guides/`
- Legacy pointers:
  - `workflows/legacy/CLEAN_INSTALLATION.md`
  - `workflows/legacy/WORKFLOW_TRAINING.md`
  - `cicd/legacy/ci_cd_testing_integration.md`

3) Technical Specs (canonical)
- Canonical: `docs/guides/technical-specs/specifications/`
- Legacy pointers:
  - `legacy/checkpoint_format_specification.md`
  - `legacy/configuration_storage_specification.md`

4) User Usage Guides (canonical)
- Canonical: `docs/guides/user-guides/usage/`
- Legacy pointers:
  - `legacy/USAGE.md`
  - `legacy/experiment_tracker/`

Actions Completed
- Link path fixes across README, tutorials, infra.
- Link checker integrated and validated (0 issues).

Pending (non-blocking)
- Evaluate merging duplicated READMEs within legacy folders into single index per bucket.
- Add short landing indexes per bucket with “legacy” pointers.


