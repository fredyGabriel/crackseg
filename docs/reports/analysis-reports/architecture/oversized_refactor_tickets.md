<!-- markdownlint-disable-file -->
# Refactor Tickets — Oversized Modules

References:
- Prioritization: `docs/reports/analysis-reports/architecture/oversized_prioritization_matrix.md`
- Plan: `docs/reports/analysis-reports/architecture/oversized_refactor_plan.md`
- Risks/Rollback: `docs/designs/risk_and_rollback_strategy.md`

## Ticket Summary (Table)

ID | Module | Priority | Scope | Acceptance (capsule) | Dependencies | Risks | Owner | Due
:-- | :-- | :--: | :-- | :-- | :-- | :-- | :-- | :--
T-P1-DEC-01 | `crackseg.model.decoder.cnn_decoder` | P1 | Split into `blocks.py`, `decoder_head.py`, `skip_fusion.py`, `init_utils.py` | ≤280 lines each; tests green; no new cycles; checkpoints load | None | state_dict compat | TBD | TBD
T-P1-UNET-02 | `crackseg.model.core.unet` | P1 | Split into `encoder_bridge.py`, `decoder_bridge.py`, `unet_core.py`, `interfaces.py` | ≤300; unit+int tests; no new cycles | T-P1-DEC-01 | trainer coupling | TBD | TBD
T-P1-HYB-03 | `crackseg.model.architectures.swinv2_cnn_aspp_unet` | P1 | Split into `swin_adapter.py`, `aspp_bottleneck.py`, `hybrid_unet.py` | ≤300; hydra smoke ok; config-validation | T-P1-UNET-02 | hydra paths | TBD | TBD
T-P2-CONF-04 | `crackseg.model.config.instantiation` | P2 | Extract `parsers.py`, `validators.py`, `builders.py` | ≤300; unit tests; no cycles | T-P1-* | coupling | TBD | TBD
T-P2-FACT-05 | `crackseg.model.factory.factory` | P2 | Extract `component_factory.py`, `architecture_factory.py` | ≤300; unit tests; single registry | T-P1-* | registry drift | TBD | TBD
T-P3-UTIL-06 | `crackseg.utils.config.standardized_storage` | P3 | Extract `io.py`, `validation.py`, `migration.py` | ≤300; unit tests | none | broad usage | TBD | TBD
T-P3-REPT-07 | `crackseg.reporting.comparison.engine` & figures | P3 | Separate data prep / rendering / exporters | ≤300; unit tests | none | template coupling | TBD | TBD

---

## T-P1-DEC-01 — Decoder split
- Module: `crackseg.model.decoder.cnn_decoder` (974)
- Priority: P1
- Scope:
  - Create `blocks.py`, `decoder_head.py`, `skip_fusion.py`, `init_utils.py`
  - Move code by concern; add small internal interfaces where useful
  - Provide transitional re-exports to keep import paths
- Acceptance:
  - Each file ≤280 lines; unit + integration tests green
  - Dependency graph shows no new cycles
  - Checkpoints load with `strict=True` or via compatibility with `strict=False`
  - Hydra smoke: `python run.py --config-name=basic_verification`
- Risks & Mitigations: state_dict compatibility → re-exports + tests

## T-P1-UNET-02 — UNet core split
- Module: `crackseg.model.core.unet` (794)
- Priority: P1 (depends on DEC-01)
- Scope: split into `encoder_bridge.py`, `decoder_bridge.py`, `unet_core.py`, `interfaces.py`
- Acceptance: ≤300 lines per file; unit+integration tests; no cycles; Hydra smoke passes
- Risks: trainer coupling → adapters + typed interfaces

## T-P1-HYB-03 — SwinV2 hybrid split
- Module: `crackseg.model.architectures.swinv2_cnn_aspp_unet` (662)
- Priority: P1 (depends on UNET-02)
- Scope: split into `swin_adapter.py`, `aspp_bottleneck.py`, `hybrid_unet.py`
- Acceptance: ≤300; config-validation; experiment smoke passes
- Risks: Hydra paths → update configs + run config tests

## T-P2-CONF-04 — Config instantiation split
- Module: `crackseg.model.config.instantiation` (631)
- Scope: extract `parsers.py`, `validators.py`, `builders.py`
- Acceptance: ≤300; unit tests; hydra-registry alignment unchanged

## T-P2-FACT-05 — Factory split
- Module: `crackseg.model.factory.factory` (561)
- Scope: extract `component_factory.py`, `architecture_factory.py`; ensure single registry
- Acceptance: ≤300; unit tests; no duplicate registration

## T-P3-UTIL-06 — Standardized storage split
- Module: `crackseg.utils.config.standardized_storage` (581)
- Scope: extract `io.py`, `validation.py`, `migration.py`
- Acceptance: ≤300; unit tests; maintain API

## T-P3-REPT-07 — Reporting split
- Modules: `crackseg.reporting.comparison.engine` (652) & figures (504)
- Scope: split by concern (data prep, rendering, exporters)
- Acceptance: ≤300; unit tests; templates render unchanged


