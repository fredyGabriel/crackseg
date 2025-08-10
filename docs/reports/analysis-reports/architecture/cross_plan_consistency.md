<!-- markdownlint-disable-file -->
# Cross-Plan Consistency & Automation

## Risks
- Stale paths in docs/tickets after moves
- Hydra component breaks after registry/structure changes
- Analysis reports not regenerated post-changes
- Broken links in docs
- Checkpoint/class path drift

## Solutions
- Mapping registry (`scripts/utils/maintenance/mapping_registry.py`) drives import path updates and re-exports
- Regeneration script (`scripts/utils/maintenance/regenerate_analysis_reports.py`) run post-phase
- CI guardrails: line limits, Hydra smoke, dependency/alignment reports
- Link checker integrated (scripts/utils/quality/guardrails/link_checker.py) — current status: 0 issues in docs/ and infrastructure/

## Workflow
1. Execute phase/ticket
2. Update mapping entries for any moved public symbol
3. Run regeneration script and review reports
4. Ensure guardrails pass; fix issues
5. Commit mappings + regenerated artifacts


