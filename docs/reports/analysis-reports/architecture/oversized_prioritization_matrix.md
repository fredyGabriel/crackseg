<!-- markdownlint-disable-file -->
# Oversized Modules Prioritization Matrix

Impact vs Effort matrix to schedule refactors. Impact considers code centrality (fan-in/out), size, and training-path criticality. Effort is an estimate based on coupling and config exposure.

Module | Lines | Impact (H/M/L) | Effort (H/M/L) | Priority (P1/P2/P3) | Notes
:-- | --: | :--: | :--: | :--: | :--
`crackseg.model.decoder.cnn_decoder` | 974 | H | M | P1 | Decoder core; high fan-in; checkpoint sensitive
`crackseg.model.core.unet` | 794 | H | M | P1 | Afecta training; depende de decoder split
`crackseg.model.architectures.swinv2_cnn_aspp_unet` | 662 | H | M | P1 | Hydra exposure; validar configs
`crackseg.model.config.instantiation` | 631 | M | M | P2 | Validadores/parsers con coupling
`crackseg.model.factory.factory` | 561 | M | L | P2 | Quick win: extraer constructores
`crackseg.utils.config.standardized_storage` | 581 | M | M | P3 | Utilidad transversal; plan gradual
`crackseg.reporting.comparison.engine` | 652 | M | M | P3 | Partir en data prep/render/export
`crackseg.reporting.figures.publication_figure_generator` | 504 | M | L | P3 | Quick win

Scheduling guidance:
- Execute P1 in order: decoder → unet core → hybrid architecture
- Then P2 modules, starting with factory (lower effort)
- Defer P3 unless bloquea P1/P2


