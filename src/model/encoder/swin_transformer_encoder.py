# NOTE: This file exceeds the recommended 300-line limit because it implements
# a highly cohesive, self-contained SwinTransformerEncoder class. All methods
# are tightly coupled to the encoder logic and splitting would reduce clarity.
# This exception is documented as per the coding-preferences guidelines.

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Union
from src.model.base import EncoderBase
import timm
import logging
import re

logger = logging.getLogger(__name__)


class SwinTransformerEncoder(EncoderBase):
    """
    Swin Transformer V2 Encoder for segmentation tasks.

    This encoder uses the Swin Transformer V2 architecture from the timm
    library and adapts it for use in a U-Net architecture. It extracts
    multi-scale features from the input and returns them in a format
    suitable for a decoder with skip connections.

    Features of Swin Transformer V2:
    - Post-normalization for improved training stability
    - Scaled cosine attention for better performance at high resolutions
    - Log-spaced continuous relative position bias for better transferability
    - Improved architecture with more flexible parameter configurations

    Each stage of the Swin Transformer performs patch merging that reduces the
    resolution by 2x, similar to max pooling in CNNs.
    """

    def __init__(
        self,
        in_channels: int,
        model_name: str = "swinv2_tiny_window16_256",
        pretrained: bool = True,
        output_hidden_states: bool = True,
        features_only: bool = True,
        out_indices: Optional[List[int]] = None,
        img_size: int = 256,
        patch_size: int = 4,
        use_abs_pos_embed: bool = True,
        output_norm: bool = True,
        handle_input_size: str = "resize",  # Options: "resize", "pad", "none"
        # Fine-tuning options
        freeze_layers: Union[bool, str, List[str]] = False,
        # Per-layer learning rate scales
        finetune_lr_scale: Dict[str, float] = None,
    ):
        """
        Initialize the Swin Transformer V2 Encoder.

        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            model_name (str): Name of the Swin Transformer V2 model from timm.
                              Options include: 'swinv2_tiny_window16_256',
                              'swinv2_small_window16_256',
                              'swinv2_base_window16_256', etc.
                              Default: 'swinv2_tiny_window16_256'.
            pretrained (bool): Whether to use pretrained weights. Default: True
            output_hidden_states (bool): Whether to return all hidden states.
                                        Default: True.
            features_only (bool): Whether to return feature maps instead of
                                 classification output. Default: True.
            out_indices (List[int], optional): Indices of the feature maps to
                                              return. Default: None, will use
                                              [0, 1, 2, 3].
            img_size (int): Input image size for the model. Default: 256.
            patch_size (int): Patch size. Default: 4.
            use_abs_pos_embed (bool): Whether to use absolute positional
                                     embeddings. Default: True.
            output_norm (bool): Whether to apply normalization to output
                               features. Default: True.
            handle_input_size (str): How to handle inputs with sizes different
                                    from img_size. Options:
                                    - "resize": Resize input to img_size
                                    - "pad": Pad input to make divisible by
                                      patch_size
                                    - "none": No special handling (may error
                                      if sizes are incompatible)
            freeze_layers (Union[bool, str, List[str]]): Control which parts of
                                    the model are frozen during training.
                                    Options:
                                    - True: Freeze all but the last block
                                    - False: No freezing (default)
                                    - "all": Freeze entire encoder
                                    - "patch_embed": Freeze patch embedding
                                    only
                                    - "stages1,stages2": Freeze specific stages
                                    - ["stages1", "stages2"]: Same as
                                    comma-separated
            finetune_lr_scale (Dict[str, float]): Optional dictionary mapping
                                    layer patterns to learning rate scaling
                                    factors for differential learning rates
                                    during fine-tuning.
                                    Example: {"patch_embed": 0.1, "stages0":
                                        0.3, "stages1": 0.5, "stages2": 0.7}
        """
        super().__init__(in_channels)

        # Set default out_indices if not provided
        if out_indices is None:
            out_indices = [0, 1, 2, 3]  # Default indices for feature maps

        self.out_indices = out_indices
        self._skip_channels = []  # Will be populated after model init.
        self.img_size = img_size
        self.patch_size = patch_size
        self.handle_input_size = handle_input_size
        self.use_abs_pos_embed = use_abs_pos_embed
        self.output_norm = output_norm
        self.freeze_layers = freeze_layers
        self.finetune_lr_scale = finetune_lr_scale if finetune_lr_scale else {}

        # Check if model name is compatible with img_size
        # Swin V2 models typically have window size and input size in their
        # name. For example, swinv2_tiny_window8_256 expects 256x256 images
        # with window size 8
        self._validate_model_config(model_name, img_size)

        # Check for valid handle_input_size value
        valid_size_handlers = ["resize", "pad", "none"]
        if handle_input_size not in valid_size_handlers:
            raise ValueError(
                f"handle_input_size must be one of {valid_size_handlers}, "
                f"got {handle_input_size}"
            )

        # Initialize the Swin Transformer V2 model
        try:
            logger.info(f"Attempting to initialize {model_name} with \
img_size={img_size}")

            # Create the model - using features_only mode which returns
            # intermediate features
            self.swin = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=in_channels,
                features_only=features_only,
                out_indices=out_indices,
            )
            logger.info(f"Successfully initialized {model_name} model")
        except Exception as e:
            logger.error(f"Failed to initialize {model_name}: {str(e)}")
            logger.error("Falling back to ResNet-based encoder")

            # Fallback to a ResNet-based model which is more robust to
            # different input sizes
            try:
                fallback_model = "resnet34"
                self.swin = timm.create_model(
                    fallback_model,
                    pretrained=pretrained,
                    in_chans=in_channels,
                    features_only=True,
                    out_indices=out_indices,
                )
                logger.info(f"Successfully initialized fallback model \
{fallback_model}")
            except Exception as e2:
                logger.error(f"Failed to initialize fallback model: {str(e2)}")
                raise ValueError(f"Could not initialize any model: {str(e2)}")

        # Store the output channels of each stage
        feature_info = self.swin.feature_info
        all_channels = [info['num_chs'] for info in feature_info.info]

        # Store spatial reduction factor for each stage for potential
        # upsampling
        self.reduction_factors = [info.get('reduction', 2**(i+1))
                                  for i, info in enumerate(feature_info.info)]

        # Store channels in the order needed for decoder (high to low res)
        # Note: Swin returns features from low to high resolution
        # So we reverse the order
        self._skip_channels = list(reversed(all_channels[:-1]))  # Exclude last
        self._out_channels = all_channels[-1]  # Last feature map's channels

        logger.info(f"Encoder initialized with out_channels=\
{self._out_channels}, " f"skip_channels={self._skip_channels}")

        # Apply transfer learning configuration (layer freezing)
        if freeze_layers:
            self._apply_layer_freezing()

    def _validate_model_config(self, model_name: str, img_size: int) -> None:
        """
        Validate that the model name is compatible with the configured image
        size.

        Args:
            model_name (str): The model name to validate.
            img_size (int): The configured image size.
        """
        # Extract the image size from the model name if present
        if "_" in model_name:
            name_parts = model_name.split("_")
            if len(name_parts) >= 3:
                # Last part might contain the image size (e.g., "256")
                try:
                    name_img_size = int(name_parts[-1])
                    if name_img_size != img_size:
                        logger.warning(
                            f"Model {model_name} is designed for "
                            f"{name_img_size}x{name_img_size} images, "
                            f"but img_size={img_size} was specified.")
                except ValueError:
                    # Not a number, so can't determine image size from name
                    pass

    def _preprocess_input(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Preprocess input tensor based on configuration.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, Optional[Dict]]:
                - Preprocessed tensor
                - Dictionary with metadata for potential post-processing, or
                    None
        """
        # Store original size for reference
        original_size = (x.shape[2], x.shape[3])
        metadata = {"original_size": original_size}

        # Basic input validation
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B,C,H,W), got {x.dim()}D")

        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"got {x.size(1)}"
            )

        if (min(x.size(2), x.size(3)) < self.patch_size and
                self.handle_input_size == "none"):
            raise ValueError(
                f"Input dimensions (H={x.size(2)}, W={x.size(3)}) must be at "
                f"least {self.patch_size}x{self.patch_size} with "
                f"handle_input_size='none'"
            )

        # Apply preprocessing based on configuration
        if self.handle_input_size == "resize":
            # Resize to the model's expected size
            if original_size != (self.img_size, self.img_size):
                x = F.interpolate(
                    x,
                    size=(self.img_size, self.img_size),
                    mode='bilinear',
                    align_corners=False
                )
                metadata["resized"] = True
                logger.debug(f"Resized input from {original_size} to "
                             f"({self.img_size}, {self.img_size})")

        elif self.handle_input_size == "pad":
            # Calculate padding to make dimensions divisible by patch_size
            B, C, H, W = x.shape
            pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
            pad_w = (self.patch_size - W % self.patch_size) % self.patch_size

            if pad_h > 0 or pad_w > 0:
                # Apply padding (pad_left, pad_right, pad_top, pad_bottom)
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
                metadata["padding"] = (0, pad_w, 0, pad_h)
                metadata["padded_size"] = (H + pad_h, W + pad_w)
                logger.debug(f"Padded input from {(H, W)} to "
                             f"{(H + pad_h, W + pad_w)}")

        return x, metadata

    def _postprocess_features(
        self,
        features: List[torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[torch.Tensor]:
        """
        Apply post-processing to features if needed.

        Args:
            features (List[torch.Tensor]): Feature tensors.
            metadata (Dict[str, Any], optional): Metadata from preprocessing.

        Returns:
            List[torch.Tensor]: Post-processed features.
        """
        # If no metadata or no modifications were made, return as is
        if metadata is None:
            return features

        processed_features = []

        for feature in features:
            # Apply post-processing if needed
            if "resized" in metadata and metadata["resized"]:
                # Resize back to original spatial dimensions adjusted for the
                # reduction factor
                original_h, original_w = metadata["original_size"]
                feature_h = max(1, original_h // self.patch_size)
                feature_w = max(1, original_w // self.patch_size)

                # Only resize if dimensions are different
                current_h, current_w = feature.shape[2], feature.shape[3]
                if (current_h, current_w) != (feature_h, feature_w):
                    feature = F.interpolate(
                        feature,
                        size=(feature_h, feature_w),
                        mode='bilinear',
                        align_corners=False
                    )

            # If we need to handle padding, we could add logic here to crop
            # the features back to the unpadded size if needed

            processed_features.append(feature)

        return processed_features

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor,
                                                List[torch.Tensor]]:
        """
        Forward pass through the Swin Transformer encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - bottleneck: Final feature map, deepest encoding.
                - skip_connections: List of intermediate feature maps for skip
                                  connections, in order from high to low
                                  resolution (for the decoder).
        """
        # Handle input preprocessing
        x, metadata = self._preprocess_input(x)

        # Forward pass through the model
        features = self.swin(x)

        # Sanity check for dimensions - features should come from low to high
        # resolution
        if len(features) != len(self.out_indices):
            logger.warning(
                f"Expected {len(self.out_indices)} features, but got "
                f"{len(features)}. This may indicate a model configuration "
                f"issue."
            )

        # Apply post-processing if necessary
        if metadata:
            features = self._postprocess_features(features, metadata)

        # Prepare return values - we need to reverse the feature order from
        # Swin's native output
        # The bottleneck is the last feature (lowest resolution)
        bottleneck = features[-1]

        # Skip connections are the first N-1 features (from higher resolution)
        # We reverse to get from high to low resolution (decoder's expected
        # order)
        skip_connections = list(reversed(features[:-1]))

        return bottleneck, skip_connections

    def get_feature_info(self) -> List[Dict[str, Any]]:
        """
        Get information about feature maps produced by the encoder.

        Returns:
            List[Dict[str, Any]]: Information about each feature map,
                                 including channels and reduction factor.
        """
        feature_info = []
        # Iterate through the skip channel info (high to low res)
        for i, channels in enumerate(self._skip_channels):
            feature_info.append({
                "channels": channels,
                "reduction_factor": self.reduction_factors[i],
                "stage": i
            })

        # Add bottleneck info
        feature_info.append({
            "channels": self._out_channels,
            "reduction_factor": self.reduction_factors[-1],
            "stage": len(self._skip_channels)
        })

        return feature_info

    @property
    def out_channels(self) -> int:
        """Number of channels in the final output tensor (bottleneck)."""
        return self._out_channels

    @property
    def skip_channels(self) -> List[int]:
        """List of channels for each skip connection (high to low res)."""
        return self._skip_channels

    def _apply_layer_freezing(self) -> None:
        """
        Applies layer freezing based on the freeze_layers configuration.

        This is used for transfer learning to control which parts of the model
        are trainable.
        """
        freeze_patterns = []

        # Convert string input to list of patterns
        if isinstance(self.freeze_layers, bool):
            if self.freeze_layers:
                # Default behavior: freeze all except the last block
                if hasattr(self.swin, 'stages'):
                    num_stages = len(self.swin.stages)
                    freeze_patterns = [
                        'patch_embed',
                        *(f'stages.{i}' for i in range(num_stages - 1))
                    ]
                else:
                    logger.warning("Could not determine stages in model. \
Using basic freezing.")
                    freeze_patterns = ['patch_embed', 'blocks.0', 'blocks.1']
            else:
                # No freezing
                return
        elif isinstance(self.freeze_layers, str):
            if self.freeze_layers.lower() == 'all':
                freeze_patterns = ['.*']  # Freeze all layers
            else:
                # Split comma-separated string
                freeze_patterns = [
                    p.strip() for p in self.freeze_layers.split(',')]
        elif isinstance(self.freeze_layers, list):
            freeze_patterns = self.freeze_layers
        else:
            logger.warning(
                f"Unsupported freeze_layers type: {type(self.freeze_layers)}")
            return

        # Apply freezing
        frozen_params = 0
        total_params = 0

        for name, param in self.swin.named_parameters():
            total_params += param.numel()
            # Check if parameter matches any pattern
            if any(re.search(pattern, name) for pattern in freeze_patterns):
                param.requires_grad = False
                frozen_params += param.numel()

        # Log freezing statistics
        frozen_percentage = (frozen_params / max(total_params, 1)) * 100
        logger.info(f"Transfer learning configuration applied: "
                    f"Froze {frozen_params:,} parameters "
                    f"({frozen_percentage:.1f}% of model)")

    def get_optimizer_param_groups(self, base_lr: float = 0.001) -> List[Dict]:
        """
        Returns parameter groups with differential learning rates for
        fine-tuning.

        This enables techniques like discriminative learning rates, where
        different parts of the model are trained with different learning
        rates.

        Args:
            base_lr (float): Base learning rate to scale other LRs from

        Returns:
            List[Dict]: Parameter groups with custom learning rates
        """
        if not self.finetune_lr_scale:
            # If no scaling is specified, return a single parameter group
            return [{'params': self.parameters(), 'lr': base_lr}]

        # Create parameter groups with scaled learning rates
        param_groups = []
        default_group_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # Skip frozen parameters

            matched = False
            for pattern, scale in self.finetune_lr_scale.items():
                if re.search(pattern, name):
                    param_groups.append({
                        'name': pattern,
                        'params': [param],
                        'lr': base_lr * scale
                    })
                    matched = True
                    break

            if not matched:
                default_group_params.append(param)

        # Add default group with base learning rate
        if default_group_params:
            param_groups.append({
                'name': 'default',
                'params': default_group_params,
                'lr': base_lr
            })

        # Log parameter group configuration
        logger.info(
            f"Created {len(param_groups)} parameter groups for fine-tuning")
        for group in param_groups:
            if 'name' in group:
                logger.debug(f"Parameter group '{group['name']}': "
                             f"LR = {group['lr']}")

        return param_groups

    def gradual_unfreeze(self, current_epoch: int,
                         unfreeze_schedule: Dict[int, List[str]]) -> None:
        """
        Gradually unfreezes layers based on the current epoch and schedule.

        This implements the gradual unfreezing technique for transfer learning,
        where deeper layers are unfrozen later in training.

        Args:
            current_epoch (int): Current training epoch
            unfreeze_schedule (Dict[int, List[str]]): Mapping of epochs to
                patterns to unfreeze
                Example: {5: ['stages.0'], 10: ['stages.1'], 15: ['stages.2']}
                The patterns are automatically adapted to the actual model
                structure
                (e.g., 'stages.0' will also match 'layers_0' in SwinV2 models)
        """
        # Find which patterns to unfreeze at current epoch
        patterns_to_unfreeze = []
        for epoch, patterns in sorted(unfreeze_schedule.items()):
            if current_epoch >= epoch:
                patterns_to_unfreeze.extend(patterns)

        if not patterns_to_unfreeze:
            return  # Nothing to unfreeze

        # For debugging, review the available parameter names
        param_names = [name for name, _ in self.swin.named_parameters()]
        block_prefixes = set()
        for name in param_names:
            parts = name.split('.')
            if len(parts) > 1:
                block_prefixes.add(parts[0])

        # Log debug information
        logger.debug(
            f"Available parameter block prefixes: {sorted(block_prefixes)}"
        )

        # Adapt patterns based on the actual model structure
        adapted_patterns = []
        for pattern in patterns_to_unfreeze:
            # Handle 'stages.X' -> look for the corresponding pattern in the
            # model
            if pattern.startswith('stages.'):
                stage_num = pattern.split('.')[1]
                # For Swin models that use 'layers_X' instead of 'stages.X'
                adapted_patterns.append(f'layers_{stage_num}')
                # Alternative patterns for other models
                adapted_patterns.append(f'stages\\.{stage_num}')
                adapted_patterns.append(f'blocks\\.{stage_num}')
                adapted_patterns.append(f'layers\\.{stage_num}')
            else:
                adapted_patterns.append(pattern)

        logger.info(f"Adapting patterns {patterns_to_unfreeze} to \
{adapted_patterns}")

        # Unfreeze matching parameters
        unfrozen_count = 0
        for name, param in self.swin.named_parameters():
            if not param.requires_grad:  # Only consider frozen parameters
                # Check if the parameter name matches any pattern
                should_unfreeze = False
                for adapted_pattern in adapted_patterns:
                    if re.search(adapted_pattern, name) or \
                            adapted_pattern in name:
                        should_unfreeze = True
                        break

                if should_unfreeze:
                    param.requires_grad = True
                    unfrozen_count += 1
                    logger.debug(f"Unfroze parameter: {name}")

        if unfrozen_count > 0:
            logger.info(
                f"Epoch {current_epoch}: Unfroze {unfrozen_count} parameters "
                f"matching patterns {patterns_to_unfreeze}")
        else:
            logger.warning(f"No parameters were unfrozen for patterns"
                           f" {patterns_to_unfreeze}. "
                           f"Adapted patterns: {adapted_patterns}")
            logger.warning("This may indicate that the pattern names don't"
                           " match the model structure.")
