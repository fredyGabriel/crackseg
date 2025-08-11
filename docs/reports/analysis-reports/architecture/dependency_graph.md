<!-- markdownlint-disable-file -->
# Dependency graph report

Nodes: 411 | Edges: 509 | Cycles: 3

## Top fan-out (dependencies)

Module | Outgoing
:-- | --:
`crackseg.training.components.setup` | 18
`crackseg.model.components.registry_support` | 15
`crackseg.model.architectures.swinv2_cnn_aspp_unet_core` | 12
`crackseg.utils.reporting` | 11
`crackseg.utils.storage` | 11
`crackseg.model.encoder.swin.core` | 10
`crackseg.training.components.training_loop` | 10
`crackseg.model.decoder.decoder_head` | 9
`crackseg.model.encoder.swin` | 9
`crackseg.evaluation.cli.components` | 8
`crackseg.evaluation.cli.runner` | 8
`crackseg.model.encoder.cnn_encoder` | 8
`crackseg.utils.experiment.experiment` | 8
`crackseg.utils.factory.factory` | 8
`crackseg.evaluation.utils.loading` | 7
`crackseg.model.encoder.swin_v2_adapter` | 7
`crackseg.model.encoder` | 7
`crackseg.model.config.instantiation.components` | 7
`crackseg.utils.experiment.tracker` | 7
`crackseg.data.factory.dataset_creator` | 6

## Top fan-in (dependents)

Module | Incoming
:-- | --:
`crackseg` | 100
`crackseg.utils` | 46
`crackseg.model` | 31
`crackseg.utils.logging` | 26
`crackseg.training` | 18
`crackseg.model.factory` | 15
`crackseg.model.base` | 14
`crackseg.data` | 12
`crackseg.model.base.abstract` | 12
`crackseg.model.factory.registry_setup` | 12
`crackseg.model.encoder` | 9
`crackseg.training.losses` | 9
`crackseg.training.losses.loss_registry_setup` | 8
`crackseg.utils.experiment` | 8
`crackseg.data.loaders` | 6
`crackseg.utils.logging.base` | 6
`crackseg.utils.storage` | 5
`crackseg.utils.logging.training` | 5
`crackseg.utils.core` | 5
`crackseg.model.encoder.swin` | 4

## Cycles (SCCs > 1)

- Cycle 1 (7 modules):
  - `crackseg.model.encoder`
  - `crackseg.model.encoder.cnn_encoder`
  - `crackseg.model.encoder.swin`
  - `crackseg.model.encoder.swin.core`
  - `crackseg.model.encoder.swin.initialization`
  - `crackseg.model.encoder.swin_transformer_encoder`
  - `crackseg.model.encoder.swin_v2_adapter`
- Cycle 2 (2 modules):
  - `crackseg.utils.checkpointing.helpers`
  - `crackseg.utils.storage`
- Cycle 3 (2 modules):
  - `crackseg.data`
  - `crackseg.data.loaders`
