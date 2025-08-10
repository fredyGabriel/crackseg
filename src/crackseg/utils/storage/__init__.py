"""Public storage shim aggregating config and checkpointing APIs.

Import from here for a stable public interface:

from crackseg.utils.storage import (
    StandardizedConfigStorage,
    generate_environment_metadata,
    validate_configuration_completeness,
    enrich_configuration_with_environment,
    create_configuration_backup,
    migrate_legacy_configuration,
    save_checkpoint,
    load_checkpoint,
    load_checkpoint_dict,
    validate_checkpoint_completeness,
)
"""

from __future__ import annotations

from crackseg.utils.checkpointing.config import CheckpointSaveConfig
from crackseg.utils.checkpointing.helpers import (
    CheckpointConfig,
    CheckpointContext,
    handle_epoch_checkpointing,
)
from crackseg.utils.checkpointing.load import (
    load_checkpoint,
    load_checkpoint_dict,
    validate_checkpoint_completeness,
)
from crackseg.utils.checkpointing.save import save_checkpoint
from crackseg.utils.checkpointing.setup import setup_checkpointing
from crackseg.utils.config.standardized_storage import (
    StandardizedConfigStorage,
    create_configuration_backup,
    enrich_configuration_with_environment,
    generate_environment_metadata,
    migrate_legacy_configuration,
    validate_configuration_completeness,
)

__all__ = [
    # Config storage
    "StandardizedConfigStorage",
    "generate_environment_metadata",
    "validate_configuration_completeness",
    "enrich_configuration_with_environment",
    "create_configuration_backup",
    "migrate_legacy_configuration",
    # Checkpointing
    "save_checkpoint",
    "load_checkpoint",
    "load_checkpoint_dict",
    "validate_checkpoint_completeness",
    "CheckpointSaveConfig",
    "CheckpointConfig",
    "CheckpointContext",
    "handle_epoch_checkpointing",
    "setup_checkpointing",
]
