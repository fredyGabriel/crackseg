# Duplicate Scan Priorities (Task 4.2)

Source: `duplicate_scan_report.md/json` (35 groups)

## Prioritization Criteria
- Impact: breadth of usage across scripts/src
- Risk: likelihood of behavior change
- Effort: size/complexity to consolidate

## High Priority (low risk, broad impact)
1) setup_logging (7 hits, scripts/*)
- Action: centralize in `scripts/utils/common/logging_utils.py` (DONE in Wave 1)
- Next: follow-up scan to ensure all scripts import the shared helper

2) Empty plot helpers (3 hits, viz legacy)
- Action: `create_empty_plot` → `evaluation/visualization/utils/plot_utils.py` (DONE in Wave 2)
- Next: sweep remaining legacy viz files for stragglers

## Medium Priority
3) Target image size getter `_get_target_size` (3 hits, eval/viz)
- Action: `get_target_size_from_config` → `utils/data/image_size.py` (DONE)
- Next: check any residual bespoke implementations

4) feature_info shape metadata (4 hits, model encoders)
- Action: keep `feature_info_utils` as single source; confirm all encoders use
  `create_feature_info_entry`/`build_feature_info_from_channels` (CONFIRMED)
- Next: add unit smoke to spot regressions when adding new encoders (deferred to tests PR)

## Lower Priority / Investigate
5) Plot layout/grid/overlay helpers (multiple occurrences in viz)
- Action: route via `plot_utils` (`compute_grid_layout`, `reshape_axes_to_2d`,
  `overlay_mask`, `build_error_map_and_legend`) (PARTIAL)
- Next: greps to enforce usage and remove bespoke copies

6) Doc/report string templates and link checks (scripts/docs)
- Action: unify small regex/link utilities where safe; keep file-specific behavior
- Next: whitelist intentional variants in report

## Acknowledged (Benign / Keep as-is)
- Compatibility shims and re-exports to preserve imports
- Minimal duplication in legacy modules retained until deprecation window closes

## Execution Plan
- Wave 1 (DONE): logging
- Wave 2 (DONE): empty plot + target size; viz utils extended
- Wave 3 (IN PROGRESS): model `feature_info` audit (no functional changes)
- Wave 4 (NEXT): viz consolidation sweep and script regex/link helpers

## Guardrails
- Preserve public APIs and import paths (use re-exports when needed)
- File size limits: preferred ≤300 LOC, hard max 400
- Quality gates: ruff/black/basedpyright green per wave
- No behavioral changes; observational equivalence

## Metrics
- Goal: reduce duplicate groups ≥50% after Waves 1–2 (baseline: 35)
- CI: include duplicate scan in reports for visibility (optional)
