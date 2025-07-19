"""Configuration classes for Swin Transformer V2 Encoder.

This module contains the configuration dataclass that controls all aspects
of the SwinTransformerEncoder behavior, from model selection to training
parameters.
"""

from dataclasses import dataclass, field


@dataclass
class SwinTransformerEncoderConfig:
    """Configuration class for SwinTransformerEncoder initialization.

    This dataclass contains all configuration parameters needed to initialize
    and customize a Swin Transformer encoder. It provides sensible defaults
    while allowing fine-grained control over model behavior.

    Configuration Categories:
        - Model Selection: Specify which Swin variant to use
        - Architecture: Control model structure and feature extraction
        - Input Handling: Configure input size adaptation strategies
        - Training: Set up layer freezing and fine-tuning parameters
        - Validation: Parameters for configuration validation

    Attributes:
        model_name: Name of the Swin Transformer model from timm library.
            Common options include:
            - "swinv2_tiny_window16_256": Tiny model with 256x256 input
            - "swinv2_small_window16_256": Small model variant
            - "swinv2_base_window16_256": Base model variant
            - "swinv2_large_window16_256": Large model variant
        pretrained: Whether to load ImageNet pretrained weights. Recommended
            for transfer learning and generally better performance.
        output_hidden_states: Whether to output intermediate hidden states.
            Set to True for multi-scale feature extraction.
        features_only: Whether to extract only feature maps without
            classification head. Should be True for segmentation tasks.
        out_indices: List of stage indices to extract features from.
            Default [0,1,2,3] extracts features from all 4 stages for
            complete feature pyramid.
        img_size: Expected input image size for model initialization.
            Must match the size in model_name for optimal performance.
        patch_size: Size of image patches. Typically 4 for Swin models.
            Determines the initial downsampling factor.
        use_abs_pos_embed: Whether to use absolute position embeddings.
            Recommended for better position awareness.
        output_norm: Whether to apply normalization to output features.
            Helps with training stability and feature consistency.
        handle_input_size: Strategy for handling input size mismatches:
            - "resize": Resize input to match expected size
            - "pad": Pad input with zeros to match expected size
            - "none": Pass input as-is (may cause errors if size mismatches)
        freeze_layers: Specification for layer freezing during training:
            - False: No layers frozen
            - True: All layers frozen
            - str: Pattern to match layer names for freezing
            - list[str]: List of patterns for selective freezing
        finetune_lr_scale: Dictionary mapping layer patterns to learning
            rate scale factors for differential learning rates during
            fine-tuning. None means uniform learning rate.
        min_model_name_parts_for_size_check: Minimum number of parts in
            model name required for automatic size validation.
        expected_input_dims: Expected number of dimensions in input tensor.
            Should be 4 for batched 2D images (N, C, H, W).

    Examples:
        >>> # Basic configuration for tiny model
        >>> config = SwinTransformerEncoderConfig(
        ...     model_name="swinv2_tiny_window16_256",
        ...     pretrained=True,
        ...     img_size=256
        ... )

        >>> # Configuration with layer freezing
        >>> config = SwinTransformerEncoderConfig(
        ...     model_name="swinv2_base_window16_256",
        ...     freeze_layers=["patch_embed", "layers.0"],
        ...     finetune_lr_scale={"layers.3": 1.0, "layers.2": 0.5}
        ... )

        >>> # Configuration for different input handling
        >>> config = SwinTransformerEncoderConfig(
        ...     handle_input_size="pad",
        ...     img_size=512,
        ...     out_indices=[1, 2, 3]  # Skip stage 0
        ... )

    Notes:
        - Model name should include size information for automatic validation
        - Pretrained weights are highly recommended for transfer learning
        - out_indices should be consecutive for proper feature pyramid
        - Layer freezing patterns use regex matching for flexibility
    """

    model_name: str = "swinv2_tiny_window16_256"
    pretrained: bool = True
    output_hidden_states: bool = True
    features_only: bool = True
    out_indices: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    img_size: int = 256
    patch_size: int = 4
    use_abs_pos_embed: bool = True
    output_norm: bool = True
    handle_input_size: str = "resize"  # Options: "resize", "pad", "none"
    freeze_layers: bool | str | list[str] = False
    finetune_lr_scale: dict[str, float] | None = None
    min_model_name_parts_for_size_check: int = 3
    expected_input_dims: int = 4
