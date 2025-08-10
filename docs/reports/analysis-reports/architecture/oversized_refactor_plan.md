<!-- markdownlint-disable-file -->
# Oversized Modules Refactor Plan

Prioritized proposals for splitting and refactoring oversized modules. Thresholds: preferred ≤300 lines, hard max 400.

## Success criteria

Module | Target lines | Stable public API | Mandatory tests | Checkpoint compatibility
:-- | --: | :--: | :-- | :--:
`crackseg.model.decoder.cnn_decoder` | ≤ 280 | Yes | unit + integration | Verified (`strict` and `strict=False`)
`crackseg.model.core.unet` | ≤ 300 | Yes | unit + integration | N/A
`crackseg.model.architectures.swinv2_cnn_aspp_unet` | ≤ 300 | Yes | unit + config-validation | Verified via Hydra

## Priority 1 (Critical, structural core)
- `crackseg.model.decoder.cnn_decoder` (974)
  - Split into:
    - `blocks.py`: basic conv/up blocks, attention blocks if any
    - `decoder_head.py`: final projection layers and logits head
    - `skip_fusion.py`: skip connection fusion (concats, alignments)
    - `init_utils.py`: initialization, weight loading helpers
  - Interfaces: small internal protocols for block inputs/outputs
  - Risks: state dict compatibility
    - Mitigation: keep class names/paths via re-exports for one release; add `load_state_dict` tests (strict and `strict=False`)

- `crackseg.model.core.unet` (794)
  - Split into: `encoder_bridge.py`, `decoder_bridge.py`, `unet_core.py` (assembly), `interfaces.py` (feature info, skip contracts)
  - Risks: coupling to trainers
    - Mitigation: adapters for current trainer expectations; typed interfaces

- `crackseg.model.architectures.swinv2_cnn_aspp_unet` (662)
  - Split into: `swin_adapter.py`, `aspp_bottleneck.py`, `hybrid_unet.py`
  - Risks: Hydra instantiation paths
    - Mitigation: update configs, run config-validation tests; verify with `run.py` smoke

## Priority 2 (Critical, config & factory)
- `crackseg.model.config.instantiation` (631)
  - Split into: `parsers.py`, `validators.py`, `builders.py`

- `crackseg.model.factory.factory` (561)
  - Split into: `component_factory.py`, `architecture_factory.py`, ensure single registry usage per 8.x

## Priority 3 (Supporting, utils/reporting)
- `crackseg.utils.config.standardized_storage` (581)
  - Split into: `io.py`, `validation.py`, `migration.py`

- `crackseg.reporting.comparison.engine` (652); `reporting.figures.publication_figure_generator` (504)
  - Split by concern: data prep vs rendering vs exporters

## Quick wins (high impact / low risk)
- Extract constructors from `crackseg.model.factory.factory` to reduce size and improve testability
- Separate validators from `crackseg.model.config.instantiation`
- Isolate rendering from data prep in reporting modules

## Milestones
1. Decoder split (P1) → tests green
   - Depends on: none
   - Stop condition: fail if dependency graph introduces new cycles or checkpoint tests fail
2. UNet core split (P1) → tests + dependency report
   - Depends on: Decoder split completed
   - Stop condition: fail if trainer integration tests break
3. Hybrid architecture split (P1) → Hydra configs updated
   - Depends on: UNet core
   - Stop condition: fail if Hydra smoke tests fail
4. Config instantiation/factory splits (P2)
5. Reporting/utils splits (P3)

## Tests & Gates
- Unit tests per new module; integration tests for training run
- Regenerate dependency graph; ensure no new cycles
- Hydra smoke tests:
  - `python run.py --config-name=basic_verification`
  - Recommended experiment config from `configs/experiments/...`
- Checkpoint compatibility tests (when applicable)
- VRAM smoke on RTX 3070 Ti (small batch) for P1 modules
- CI guardrail: fail job if any touched file >400 lines; warn if >300 (current status: link checker green; next cycle addresses top offenders by tickets)
- Gates: ruff, black, basedpyright, pytest

## Rollback
Revert per milestone; keep re-exports until next release

- Re-exports with deprecation notice for one release window
- `state_dict` compatibility tests (strict and `strict=False`)
- Mapping (old → new) imports (sample):

Old import | New import
:-- | :--
`crackseg.model.decoder.cnn_decoder.Decoder` | `crackseg.model.decoder.decoder_head.Decoder`
`crackseg.model.core.unet.UNet` | `crackseg.model.core.unet_core.UNet`

## Alignment & acceptance checks
- After each P1 milestone: update and review
  - `docs/reports/analysis-reports/architecture/dependency_graph.md`
  - `docs/reports/analysis-reports/architecture/hydra_registry_alignment.md`

## Ownership & estimates (optional)
Milestone | Owner | ETA
:-- | :-- | :--
Decoder split | TBD | TBD
UNet core split | TBD | TBD
Hybrid architecture split | TBD | TBD

## Non-goals
- GUI features are not blocked unless training breaks
- Reporting refactors should not delay P1 milestones

