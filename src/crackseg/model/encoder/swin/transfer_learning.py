"""Transfer learning utilities for Swin Transformer V2 Encoder.

This module provides comprehensive transfer learning capabilities including
layer freezing, gradual unfreezing, and differential learning rate
configuration for fine-tuning Swin Transformer models.
"""

import logging
import re
from typing import Any

import torch

logger = logging.getLogger(__name__)


class SwinTransferLearning:
    """Handles transfer learning operations for Swin Transformer models."""

    @staticmethod
    def apply_layer_freezing(
        model: torch.nn.Module, freeze_layers: bool | str | list[str]
    ) -> None:
        """Apply layer freezing based on the freeze_layers configuration.

        This is used for transfer learning to control which parts of the model
        are trainable.

        Args:
            model: The model instance to apply freezing to.
            freeze_layers: Layer freezing specification.
        """
        freeze_patterns = []

        # Convert string input to list of patterns
        if isinstance(freeze_layers, bool):
            if freeze_layers:
                # Default behavior: freeze all except the last block
                if hasattr(model, "stages"):
                    # Handle stages attribute with proper type checking
                    stages_attr = model.stages
                    if hasattr(stages_attr, "__len__"):
                        num_stages = len(stages_attr)
                        freeze_patterns = [
                            "patch_embed",
                            *(f"stages.{i}" for i in range(num_stages - 1)),
                        ]
                    else:
                        logger.warning(
                            "Could not determine stages length. "
                            "Using basic freezing."
                        )
                        freeze_patterns = [
                            "patch_embed",
                            "blocks.0",
                            "blocks.1",
                        ]
                else:
                    logger.warning(
                        "Could not determine stages in model. "
                        "Using basic freezing."
                    )
                    freeze_patterns = ["patch_embed", "blocks.0", "blocks.1"]
            else:
                # No freezing
                return
        elif isinstance(freeze_layers, str):
            if freeze_layers.lower() == "all":
                freeze_patterns = [".*"]  # Freeze all layers
            else:
                # Split comma-separated string
                freeze_patterns = [p.strip() for p in freeze_layers.split(",")]
        else:
            freeze_patterns = freeze_layers

        # Apply freezing
        frozen_params = 0
        total_params = 0

        for name, param in model.named_parameters():
            total_params += param.numel()
            # Check if parameter matches any pattern
            if any(re.search(pattern, name) for pattern in freeze_patterns):
                param.requires_grad = False
                frozen_params += param.numel()

        # Log freezing statistics
        frozen_percentage = (frozen_params / max(total_params, 1)) * 100
        logger.info(
            f"Transfer learning configuration applied: "
            f"Froze {frozen_params:,} parameters "
            f"({frozen_percentage:.1f}% of model)"
        )

    @staticmethod
    def get_optimizer_param_groups(
        model: torch.nn.Module,
        finetune_lr_scale: dict[str, float],
        base_lr: float = 0.001,
    ) -> list[dict[str, Any]]:
        """Return parameter groups with differential learning rates for
        fine-tuning.

        This enables techniques like discriminative learning rates, where
        different parts of the model are trained with different learning
        rates.

        Args:
            model: The model instance.
            finetune_lr_scale: Learning rate scaling factors by layer pattern.
            base_lr: Base learning rate to scale other LRs from.

        Returns:
            List of parameter groups with custom learning rates.
        """
        if not finetune_lr_scale:
            # If no scaling is specified, return a single parameter group
            return [{"params": model.parameters(), "lr": base_lr}]

        # Create parameter groups with scaled learning rates
        param_groups: list[dict[str, Any]] = []
        default_group_params: list[Any] = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # Skip frozen parameters

            matched = False
            for pattern, scale in finetune_lr_scale.items():
                if re.search(pattern, name):
                    param_groups.append(
                        {
                            "name": pattern,
                            "params": [param],
                            "lr": base_lr * scale,
                        }
                    )
                    matched = True
                    break

            if not matched:
                default_group_params.append(param)

        # Add default group with base learning rate
        if default_group_params:
            param_groups.append(
                {
                    "name": "default",
                    "params": default_group_params,
                    "lr": base_lr,
                }
            )

        # Log parameter group configuration
        logger.info(
            f"Created {len(param_groups)} parameter groups for fine-tuning"
        )
        for group in param_groups:
            if "name" in group:
                logger.debug(
                    f"Parameter group '{group['name']}': LR = {group['lr']}"
                )

        return param_groups

    @staticmethod
    def get_patterns_for_epoch(
        current_epoch: int, unfreeze_schedule: dict[int, list[str]]
    ) -> list[str]:
        """Determine patterns to unfreeze for the current epoch.

        Args:
            current_epoch: Current training epoch.
            unfreeze_schedule: Mapping of epochs to patterns to unfreeze.

        Returns:
            List of patterns to unfreeze.
        """
        patterns_to_unfreeze: list[str] = []
        for epoch, patterns in sorted(unfreeze_schedule.items()):
            if current_epoch >= epoch:
                patterns_to_unfreeze.extend(patterns)
        return patterns_to_unfreeze

    @staticmethod
    def log_block_prefixes_debug(model: torch.nn.Module) -> None:
        """Log available parameter block prefixes for debugging.

        Args:
            model: The model instance to inspect.
        """
        param_names: list[str] = [name for name, _ in model.named_parameters()]
        block_prefixes: set[str] = set()
        for name in param_names:
            parts = name.split(".")
            if len(parts) > 1:
                block_prefixes.add(parts[0])
        logger.debug(
            f"Available parameter block prefixes: {sorted(block_prefixes)}"
        )

    @staticmethod
    def adapt_unfreeze_patterns(patterns_to_unfreeze: list[str]) -> list[str]:
        """Adapt user-defined unfreeze patterns to internal model naming.

        Args:
            patterns_to_unfreeze: List of user-defined patterns.

        Returns:
            List of adapted patterns for internal model structure.
        """
        adapted_patterns: list[str] = []
        for pattern in patterns_to_unfreeze:
            if pattern.startswith("stages."):
                stage_num = pattern.split(".")[1]
                adapted_patterns.append(f"layers_{stage_num}")
                adapted_patterns.append(f"stages\\.{stage_num}")
                adapted_patterns.append(f"blocks\\.{stage_num}")
                adapted_patterns.append(f"layers\\.{stage_num}")
            else:
                adapted_patterns.append(pattern)
        logger.info(
            f"Adapting patterns {patterns_to_unfreeze} to {adapted_patterns}"
        )
        return adapted_patterns

    @staticmethod
    def unfreeze_parameters_by_patterns(
        model: torch.nn.Module, adapted_patterns: list[str]
    ) -> int:
        """Unfreeze model parameters matching the adapted patterns.

        Args:
            model: The model instance to modify.
            adapted_patterns: List of patterns to match for unfreezing.

        Returns:
            Number of parameters that were unfrozen.
        """
        unfrozen_count = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:  # Only consider frozen parameters
                should_unfreeze = False
                for adapted_pattern in adapted_patterns:
                    if (
                        re.search(adapted_pattern, name)
                        or adapted_pattern in name
                    ):
                        should_unfreeze = True
                        break
                if should_unfreeze:
                    param.requires_grad = True
                    unfrozen_count += 1
                    logger.debug(f"Unfroze parameter: {name}")
        return unfrozen_count

    @staticmethod
    def log_unfreezing_results(
        current_epoch: int,
        unfrozen_count: int,
        original_patterns: list[str],
        adapted_patterns: list[str],
    ) -> None:
        """Log the results of the unfreezing operation.

        Args:
            current_epoch: Current training epoch.
            unfrozen_count: Number of parameters unfrozen.
            original_patterns: Original user-specified patterns.
            adapted_patterns: Adapted patterns used for matching.
        """
        if unfrozen_count > 0:
            logger.info(
                f"Epoch {current_epoch}: Unfroze {unfrozen_count} parameters "
                f"matching patterns {original_patterns}"
            )
        else:
            logger.warning(
                "No parameters were unfrozen for patterns "
                f"{original_patterns}. Adapted patterns: {adapted_patterns}"
            )
            logger.warning(
                "This may indicate that the pattern names don't match the "
                "model structure."
            )

    @classmethod
    def gradual_unfreeze(
        cls,
        model: torch.nn.Module,
        current_epoch: int,
        unfreeze_schedule: dict[int, list[str]],
    ) -> None:
        """Gradually unfreeze layers based on the current epoch and schedule.

        This implements the gradual unfreezing technique for transfer learning,
        where deeper layers are unfrozen later in training.

        Args:
            model: The model instance to modify.
            current_epoch: Current training epoch.
            unfreeze_schedule: Mapping of epochs to patterns to unfreeze.
                Example: {5: ['stages.0'], 10: ['stages.1'], 15: ['stages.2']}
                The patterns are automatically adapted to the actual model
                structure (e.g., 'stages.0' will also match 'layers_0' in
                SwinV2 models).
        """
        original_patterns_to_unfreeze = cls.get_patterns_for_epoch(
            current_epoch, unfreeze_schedule
        )

        if not original_patterns_to_unfreeze:
            return

        cls.log_block_prefixes_debug(model)

        adapted_patterns = cls.adapt_unfreeze_patterns(
            original_patterns_to_unfreeze
        )
        unfrozen_count = cls.unfreeze_parameters_by_patterns(
            model, adapted_patterns
        )

        cls.log_unfreezing_results(
            current_epoch,
            unfrozen_count,
            original_patterns_to_unfreeze,
            adapted_patterns,
        )
