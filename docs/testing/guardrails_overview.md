# Guardrails Overview

This document summarizes the project guardrails used to keep the codebase clean and consistent
during the cleanup phases. All guardrails are runnable locally and integrated into CI.

## Line Limits (Per-File)

- Script: `scripts/utils/quality/guardrails/line_limit_check.py`
- Report: `docs/reports/analysis-reports/architecture/line_limit_guardrail.md`
- Rules: Preferred ≤ 300 lines; Hard max 400 lines (fails on any file > 400).

Usage:

```bash
python scripts/utils/quality/guardrails/line_limit_check.py
```

## Oversized Modules (Architecture View)

- Script: `scripts/utils/analysis/scan_oversized_modules.py`
- Report: `docs/reports/analysis-reports/architecture/oversized_modules_report.md`
- Purpose: Highlights modules exceeding preferred limits to prioritize refactors.

Usage:

```bash
python scripts/utils/analysis/scan_oversized_modules.py
```

## Link Checker

- Script: `scripts/utils/quality/guardrails/link_checker.py`
- Uses: Mapping registry to catch outdated or broken internal links.
- Output: Console summary; can be consumed by the CI consistency checker.

Usage:

```bash
python scripts/utils/quality/guardrails/link_checker.py --directories docs scripts
```

## Duplicate Code Guardrail

- Scanner: `scripts/reports/duplicate_scan.py` (token-normalized hashing of functions/classes under
  `src/`, `scripts`, `gui`).
- Guardrail: `scripts/utils/quality/guardrails/duplicate_guardrail.py`
- Reports:
  - Current scan: `docs/reports/project-reports/duplicate_scan_report.json`
  - Status: `docs/reports/project-reports/duplicate_guardrail_status.md`
  - Baseline: `docs/reports/project-reports/duplicate_scan_baseline.json`

Usage:

```bash
# Update baseline after intentional refactors
python scripts/utils/quality/guardrails/duplicate_guardrail.py --update-baseline

# Fail if any new duplicate group is introduced
python scripts/utils/quality/guardrails/duplicate_guardrail.py --max-delta 0
```

## Consolidated Summary

- Script: `scripts/utils/quality/guardrails/guardrails_summary.py`
- Generates:
  - `docs/reports/analysis-reports/architecture/guardrails_validation_summary.json`
  - `docs/reports/analysis-reports/architecture/guardrails_validation_summary.md`

Usage:

```bash
python scripts/utils/quality/guardrails/guardrails_summary.py
```

## CI Entry Point

- Script: `scripts/utils/quality/guardrails/ci_consistency_checker.py`
- Integrates: link checker, import policy, stale report scan, mapping registry validation, and
  duplicate guardrail.
- Flags:
  - `--skip-links`, `--skip-imports`, `--skip-stale`, `--skip-duplicates`
  - `--fail-on-warnings` to treat warnings as failures

Usage:

```bash
python scripts/utils/quality/guardrails/ci_consistency_checker.py
```

## Notes

- Only update the duplicate baseline when duplicates are intentionally accepted or refactors
  legitimately change grouping.
- Guardrails are designed to be fast, dependency-light, and Windows-friendly.
