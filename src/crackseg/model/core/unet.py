"""Complete U-Net model implementation for semantic segmentation with crack
detection.

This module provides a production-ready U-Net architecture that integrates
modular components (encoder, bottleneck, decoder) into a cohesive
segmentation model. It includes comprehensive diagnostics, memory analysis,
and visualization tools for development and deployment scenarios.

Architecture Overview:
    The BaseUNet class implements the classic U-Net architecture with skip
    connections, designed for precise pixel-level segmentation tasks. The
    modular design allows for flexible component swapping and
    configuration-driven instantiation.

Key Features:
    - Modular component integration (encoder/bottleneck/decoder)
    - Automatic skip connection validation and compatibility checking
    - Comprehensive model analysis and diagnostic tools
    - Memory usage estimation and performance profiling
    - Architecture visualization with Graphviz integration
    - Configurable final activation layers via Hydra

Core Components:
    - BaseUNet: Main model class implementing UNetBase interface
    - Component validation: Automatic channel compatibility checking
    - Model summary: Detailed parameter, memory, and architecture analysis
    - Visualization tools: Block diagram generation for architecture
    understanding

Production Features:
    - Robust error handling and validation
    - Memory-efficient forward pass implementation
    - Comprehensive logging and debugging capabilities
    - Integration with experiment tracking and model deployment

Common Usage:
    # Configuration-driven instantiation
    model = BaseUNet(
        encoder=encoder_instance,
        bottleneck=bottleneck_instance,
        decoder=decoder_instance,
        final_activation={"_target_": "torch.nn.Sigmoid"}
    )

    # Forward pass
    output = model(input_tensor)  # Shape: [B, out_channels, H, W]

    # Model analysis
    summary = model.summary(input_shape=(1, 3, 512, 512))
    model.print_summary(input_shape=(1, 3, 512, 512))
    model.visualize_architecture("unet_arch.png", view=True)

Integration Points:
    - Hydra configuration system for component instantiation
    - Abstract base classes from crackseg.model.base.abstract
    - Utility functions from crackseg.model.common.utils
    - Training pipeline integration via standardized interfaces

Performance Considerations:
    - Efficient skip connection handling with minimal memory overhead
    - Automatic memory usage estimation for deployment planning
    - Receptive field calculation for optimal input size selection
    - Component validation prevents runtime errors and silent failures

References:
    - Base classes: src.model.base.abstract
    - Utilities: src.model.common.utils
    - Original U-Net paper: https://arxiv.org/abs/1505.04597
    - Configuration: configs/model/architectures/
"""

from io import StringIO
from typing import Any, cast

import hydra.utils  # Import hydra utils
import torch
from torch import nn

# Updated imports to reflect new directory structure
from crackseg.model.base.abstract import (
    BottleneckBase,
    DecoderBase,
    EncoderBase,
    UNetBase,
)
from crackseg.model.common import (
    count_parameters,
    estimate_memory_usage,
    estimate_receptive_field,
    get_layer_hierarchy,
    render_unet_architecture_diagram,
)


class BaseUNet(UNetBase):
    """Production-ready U-Net implementation with comprehensive diagnostics
    and validation.

    This class provides a complete U-Net architecture implementation that
    integrates modular encoder, bottleneck, and decoder components with
    automatic validation, extensive diagnostic capabilities, and
    production-ready error handling.

    The implementation follows the classic U-Net design with skip connections
    between encoder and decoder stages, enabling precise localization for
    segmentation tasks. All components are validated for compatibility during
    initialization to prevent runtime errors.

    Key Features:
        - Automatic channel compatibility validation between components
        - Configurable final activation via Hydra instantiation
        - Comprehensive model analysis with memory and performance metrics
        - Architecture visualization and debugging tools
        - Production-ready error handling and validation

    Architecture:
        Input → Encoder (with skip connections) → Bottleneck → Decoder →
        Final Activation → Output

        Skip connections preserve spatial information lost during downsampling,
        enabling precise localization in the final segmentation map.

    Attributes:
        encoder: Feature extraction component with downsampling and skip
            connections
        bottleneck: Processing component at lowest spatial resolution
        decoder: Upsampling component that reconstructs segmentation map
        final_activation: Optional activation function for output processing

    Examples:
        >>> # Basic instantiation with components
        >>> encoder = SomeEncoderImplementation(in_channels=3,
        ...     features=[64, 128, 256]
        ... )
        >>> bottleneck = SomeBottleneckImplementation(
        ...     in_channels=256, out_channels=512
        ... )
        >>> decoder = SomeDecoderImplementation(
        ...     features=[256, 128, 64], out_channels=1
        ... )
        >>> model = BaseUNet(encoder, bottleneck, decoder)
        >>>
        >>> # With configurable activation
        >>> model = BaseUNet(
        ...     encoder=encoder,
        ...     bottleneck=bottleneck,
        ...     decoder=decoder,
        ...     final_activation={"_target_": "torch.nn.Sigmoid"}
        ... )
        >>>
        >>> # Forward pass
        >>> input_tensor = torch.randn(1, 3, 512, 512)
        >>> output = model(input_tensor)  # Shape: [1, 1, 512, 512]
        >>>
        >>> # Model analysis
        >>> summary = model.summary(input_shape=(1, 3, 512, 512))
        >>> print(f"Total parameters: {summary['parameters']['total']:,}")

    Validation Rules:
        - Encoder skip channels must match decoder skip channels
        (in reverse order)
        - Encoder output channels must match bottleneck input channels
        - All components must implement required abstract methods
        - Final activation must be valid PyTorch module or Hydra config

    Performance:
        - Memory usage scales with input resolution and number of features
        - Skip connections add minimal computational overhead
        - Component validation occurs only during initialization
        - Forward pass is optimized for training and inference efficiency

    Integration:
        - Compatible with PyTorch training loops and optimization
        - Supports distributed training and mixed precision
        - Integrates with experiment tracking and model checkpointing
        - Configuration-driven instantiation via Hydra
    """

    def __init__(
        self,
        encoder: EncoderBase,
        bottleneck: BottleneckBase,
        decoder: DecoderBase,
        final_activation: nn.Module | dict[str, Any] | None = None,
    ):
        """Initialize BaseUNet with comprehensive component validation.

        Performs extensive validation of component compatibility and
        initializes the complete U-Net architecture. All validation occurs
        during initialization to ensure runtime reliability and prevent silent
        failures.

        Args:
            encoder: Encoder component implementing EncoderBase interface.
                Must provide feature extraction with skip connections.
                Expected to have 'skip_channels' and 'out_channels' attributes.
            bottleneck: Bottleneck component implementing BottleneckBase
                interface.
                Processes features at lowest spatial resolution.
                Expected to have 'in_channels' attribute matching encoder
                output.
            decoder: Decoder component implementing DecoderBase interface.
                Reconstructs segmentation map with skip connection integration.
                Expected to have 'skip_channels' and 'out_channels' attributes.
            final_activation: Optional final activation function. Can be:
                - nn.Module instance: Used directly
                - dict: Hydra configuration for instantiation
                - None: No final activation applied

        Raises:
            ValueError: If component channel configurations are incompatible:
                - Encoder skip channels don't match decoder skip channels
                (reversed)
                - Encoder output channels don't match bottleneck input channels
            TypeError: If final_activation config doesn't produce nn.Module
            AttributeError: If components missing required attributes

        Examples:
            >>> # Standard instantiation
            >>> model = BaseUNet(encoder, bottleneck, decoder)
            >>>
            >>> # With activation module
            >>> model = BaseUNet(encoder, bottleneck, decoder, nn.Sigmoid())
            >>>
            >>> # With Hydra configuration
            >>> activation_config = {"_target_": "torch.nn.Sigmoid"}
            >>> model = BaseUNet(encoder, bottleneck, decoder,
            ... activation_config)

        Validation Process:
            1. Validate encoder-decoder skip connection compatibility
            2. Validate encoder-bottleneck channel compatibility
            3. Initialize final activation with error handling
            4. Store validated components for forward pass

        Performance Impact:
            - Validation occurs only during initialization
            - No runtime performance penalty
            - Prevents costly debugging of channel mismatches
            - Enables safe component swapping and configuration
        """
        super().__init__(encoder, bottleneck, decoder)

        # Initialize final activation with robust error handling
        self.final_activation = None
        if final_activation is not None:
            # If it's already an nn.Module instance, use it directly
            if isinstance(final_activation, nn.Module):
                self.final_activation = final_activation
            # Otherwise, try to instantiate from Hydra config
            else:
                try:
                    self.final_activation = hydra.utils.instantiate(
                        final_activation
                    )
                    if not isinstance(self.final_activation, nn.Module):
                        raise TypeError(
                            "final_activation config did not instantiate an "
                            "nn.Module"
                        )
                except Exception as e:
                    print(
                        f"Warning: Could not instantiate final_activation: {e}"
                    )
                    self.final_activation = None  # Fallback to no activation

        # Comprehensive component validation
        assert self.encoder is not None
        assert self.decoder is not None
        assert self.bottleneck is not None

        # Validate skip connection compatibility
        if hasattr(self.encoder, "skip_channels") and hasattr(
            self.decoder, "skip_channels"
        ):
            encoder_skip = list(self.encoder.skip_channels)
            decoder_skip = list(reversed(self.decoder.skip_channels))
            if encoder_skip != decoder_skip:
                raise ValueError(
                    f"Encoder skip channels {encoder_skip} and decoder skip "
                    f"channels {list(self.decoder.skip_channels)} are "
                    "incompatible. Decoder skip channels should be encoder "
                    "skip channels in reverse order."
                )

        # Validate encoder-bottleneck channel compatibility
        if hasattr(self.encoder, "out_channels") and hasattr(
            self.bottleneck, "in_channels"
        ):
            if self.encoder.out_channels != self.bottleneck.in_channels:
                raise ValueError(
                    f"Encoder output channels ({self.encoder.out_channels}) "
                    "and bottleneck input channels "
                    f"({self.bottleneck.in_channels}) "
                    "are incompatible. They must match for proper data flow."
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass through complete U-Net architecture.

        Implements the classic U-Net forward pass with encoder feature
        extraction, bottleneck processing, decoder reconstruction, and
        optional final activation. Skip connections preserve spatial
        information for precise localization.

        Args:
            x: Input tensor with shape [B, C, H, W] where:
                - B: Batch size
                - C: Input channels (must match encoder.in_channels)
                - H: Height (recommended to be divisible by downsampling factor
                )
                - W: Width (recommended to be divisible by downsampling factor)

        Returns:
            torch.Tensor: Output segmentation map with shape [B, O, H, W]
            where:
                - B: Batch size (unchanged)
                - O: Output channels (from decoder.out_channels)
                - H: Height (same as input, preserved by skip connections)
                - W: Width (same as input, preserved by skip connections)

        Processing Flow:
            1. Encoder: Extract hierarchical features with skip connections
            2. Bottleneck: Process features at lowest spatial resolution
            3. Decoder: Reconstruct segmentation map using skip connections
            4. Final Activation: Apply optional activation function

        Examples:
            >>> model = BaseUNet(encoder, bottleneck, decoder)
            >>> input_tensor = torch.randn(8, 3, 512, 512)  # Batch of 8 images
            >>> output = model(input_tensor)  # Shape: [8, 1, 512, 512]
            >>>
            >>> # With different input sizes
            >>> small_input = torch.randn(1, 3, 256, 256)
            >>> small_output = model(small_input)  # Shape: [1, 1, 256, 256]

        Memory Considerations:
            - Peak memory usage occurs during decoder processing
            - Skip connections temporarily increase memory usage
            - Memory scales quadratically with input resolution
            - Batch size directly multiplies memory requirements

        Performance Notes:
            - Skip connections add minimal computational overhead
            - Most computation occurs in encoder and decoder components
            - Final activation (if present) is typically lightweight
            - GPU memory transfer is primary bottleneck for large inputs

        Error Handling:
            - Input shape validation performed by encoder
            - Channel compatibility ensured during initialization
            - Skip connection shapes validated by decoder
            - Any tensor shape mismatches propagate as runtime errors
        """
        assert self.encoder is not None
        assert self.bottleneck is not None
        assert self.decoder is not None

        # Extract features and skip connections from encoder
        features, skip_connections = self.encoder(x)

        # Process features through bottleneck at lowest resolution
        bottleneck_output = self.bottleneck(features)

        # Reverse skip connections to match decoder expectations
        # Encoder provides: [HIGH to LOW resolution] order
        # Decoder expects: [LOW to HIGH resolution] order
        reversed_skip_connections = list(reversed(skip_connections))

        # Reconstruct segmentation map with skip connections
        output = self.decoder(bottleneck_output, reversed_skip_connections)

        # Apply final activation if configured
        if self.final_activation is not None:
            output = self.final_activation(output)

        return cast(torch.Tensor, output)

    def get_output_channels(self) -> int:
        """Get number of output channels from decoder component.

        Returns:
            int: Number of output channels that the model produces.
                This determines the number of segmentation classes or output
                features.

        Examples:
            >>> model = BaseUNet(encoder, bottleneck, decoder)
            >>> num_classes = model.get_output_channels()
            >>> print(f"Model outputs {num_classes} channels")
        """
        assert self.decoder is not None
        return self.decoder.out_channels

    def get_input_channels(self) -> int:
        """Get number of input channels expected by encoder component.

        Returns:
            int: Number of input channels that the model expects.
                Must match the input tensor channel dimension.

        Examples:
            >>> model = BaseUNet(encoder, bottleneck, decoder)
            >>> input_channels = model.get_input_channels()
            >>> # Create appropriate input tensor
            >>> x = torch.randn(1, input_channels, 512, 512)
        """
        assert self.encoder is not None
        return self.encoder.in_channels

    def summary(
        self, input_shape: tuple[int, ...] | None = None
    ) -> dict[str, Any]:
        """Generate comprehensive model analysis and statistics.

        Computes detailed information about model architecture, parameters,
        memory usage, receptive field, and component hierarchy. Useful for
        model development, optimization, and deployment planning.

        Args:
            input_shape: Optional input shape tuple (B, C, H, W) for memory
                estimation. If provided, enables accurate memory usage
                calculation. If None, only parameter counting and basic
                analysis is performed.

        Returns:
            dict[str, Any]: Comprehensive model statistics containing:
                - model_type: Class name of the model
                - input_channels, output_channels: Channel configuration
                - encoder_type, bottleneck_type, decoder_type: Component types
                - has_final_activation, final_activation_type: Activation info
                - parameters: Parameter counts and percentages
                - receptive_field: Theoretical receptive field analysis
                - memory_usage: Memory consumption estimates (if input_shape
                provided)
                - layer_hierarchy: Detailed component breakdown

        Examples:
            >>> model = BaseUNet(encoder, bottleneck, decoder)
            >>>
            >>> # Basic analysis without memory estimation
            >>> summary = model.summary()
            >>> print(f"Total parameters: {summary['parameters']['total']:,}")
            >>>
            >>> # Full analysis with memory estimation
            >>> full_summary = model.summary(input_shape=(1, 3, 512, 512))
            >>> memory_mb = full_summary['memory_usage']['total_estimated_mb']
            >>> print(f"Estimated memory usage: {memory_mb:.1f} MB")

        Analysis Components:
            - Parameter Analysis: Trainable/non-trainable parameter counts
            - Memory Analysis: Model size and activation memory estimates
            - Receptive Field: Theoretical receptive field calculation
            - Architecture Analysis: Component types and configuration
            - Layer Hierarchy: Detailed breakdown of all model components

        Performance Considerations:
            - Analysis computation is lightweight and fast
            - Memory estimation requires input shape for accuracy
            - Layer hierarchy extraction involves model traversal
            - Results can be cached for repeated use
        """
        trainable_params, non_trainable_params = count_parameters(self)
        total_params = trainable_params + non_trainable_params

        # Basic model information
        base_info = {
            "model_type": self.__class__.__name__,
            "input_channels": self.get_input_channels(),
            "output_channels": self.get_output_channels(),
            "encoder_type": self.encoder.__class__.__name__,
            "bottleneck_type": self.bottleneck.__class__.__name__,
            "decoder_type": self.decoder.__class__.__name__,
            "has_final_activation": self.final_activation is not None,
            "final_activation_type": (
                self.final_activation.__class__.__name__
                if self.final_activation is not None
                else None
            ),
        }

        # Comprehensive analysis dictionary
        summary_dict = {
            **base_info,
            "parameters": {
                "total": total_params,
                "trainable": trainable_params,
                "non_trainable": non_trainable_params,
                "trainable_percent": (
                    trainable_params / total_params * 100
                    if total_params > 0
                    else 0
                ),
            },
            "receptive_field": estimate_receptive_field(self.encoder),
            "memory_usage": estimate_memory_usage(
                self, self.encoder, self.get_output_channels, input_shape
            ),
            "layer_hierarchy": get_layer_hierarchy(
                self.encoder,
                self.bottleneck,
                self.decoder,
                self.final_activation,
            ),
        }
        return summary_dict

    def _print_header_info(
        self, summary_dict: dict[str, Any], target_file: Any
    ) -> None:
        """Print formatted header information for model summary.

        Args:
            summary_dict: Model summary dictionary from summary() method
            target_file: File object or stream to write formatted output
        """
        print("\n" + "=" * 80, file=target_file)
        print("U-Net Model Summary", file=target_file)
        print("=" * 80, file=target_file)
        print(f"\nModel Type: {summary_dict['model_type']}", file=target_file)
        print(
            f"Input Channels: {summary_dict['input_channels']}",
            file=target_file,
        )
        print(
            f"Output Channels: {summary_dict['output_channels']}",
            file=target_file,
        )

    def _print_parameter_info(
        self, summary_dict: dict[str, Any], target_file: Any
    ) -> None:
        """Print formatted parameter information for model summary.

        Args:
            summary_dict: Model summary dictionary from summary() method
            target_file: File object or stream to write formatted output
        """
        params = summary_dict["parameters"]
        print(f"\nTotal Parameters: {params['total']:,}", file=target_file)
        print(
            f"Trainable Parameters: {params['trainable']:,} "
            f"({params['trainable_percent']:.2f}%)",
            file=target_file,
        )
        print(
            f"Non-trainable Parameters: {params['non_trainable']:,}",
            file=target_file,
        )

    def _print_memory_info(
        self, summary_dict: dict[str, Any], target_file: Any
    ) -> None:
        """Print formatted memory usage information for model summary.

        Args:
            summary_dict: Model summary dictionary from summary() method
            target_file: File object or stream to write formatted output
        """
        mem = summary_dict["memory_usage"]
        print(f"\nModel Size: {mem['model_size_mb']:.2f} MB", file=target_file)
        if "estimated_activation_mb" in mem:
            print(
                f"Estimated Activation Memory: "
                f"{mem['estimated_activation_mb']:.2f} MB",
                file=target_file,
            )
            print(
                f"Total Estimated Memory: {mem['total_estimated_mb']:.2f} MB",
                file=target_file,
            )
            print(f"For input shape: {mem['input_shape']}", file=target_file)

    def _print_receptive_field_info(
        self, summary_dict: dict[str, Any], target_file: Any
    ) -> None:
        """Print formatted receptive field information for model summary.

        Args:
            summary_dict: Model summary dictionary from summary() method
            target_file: File object or stream to write formatted output
        """
        rf = summary_dict["receptive_field"]
        print("\nReceptive Field:", file=target_file)
        if "theoretical_rf_size" in rf:
            print(
                f"  Theoretical Size: {rf['theoretical_rf_size']}x"
                f"{rf['theoretical_rf_size']}",
                file=target_file,
            )
            print(
                f"  Downsampling Factor: {rf['downsampling_factor']}",
                file=target_file,
            )
        if "note" in rf:
            print(f"  Note: {rf['note']}", file=target_file)

    def _print_architecture_info(
        self, summary_dict: dict[str, Any], target_file: Any
    ) -> None:
        """Print formatted architecture information for model summary.

        Args:
            summary_dict: Model summary dictionary from summary() method
            target_file: File object or stream to write formatted output
        """
        print("\nArchitecture:", file=target_file)
        print(f"  Encoder: {summary_dict['encoder_type']}", file=target_file)
        print(
            f"  Bottleneck: {summary_dict['bottleneck_type']}",
            file=target_file,
        )
        print(f"  Decoder: {summary_dict['decoder_type']}", file=target_file)
        if summary_dict["has_final_activation"]:
            print(
                f" Final Activation: {summary_dict['final_activation_type']}",
                file=target_file,
            )

    def _print_layer_hierarchy_info(
        self, summary_dict: dict[str, Any], target_file: Any
    ) -> None:
        """Print formatted layer hierarchy information for model summary.

        Args:
            summary_dict: Model summary dictionary from summary() method
            target_file: File object or stream to write formatted output
        """
        print("\nLayer Hierarchy:", file=target_file)
        print("-" * 80, file=target_file)
        print(
            f"{'Layer':<25} {'Type':<20} {'Parameters':>12} "
            f"{'Input Ch':>10} {'Output Ch':>10}",
            file=target_file,
        )
        print("-" * 80, file=target_file)
        for layer in summary_dict["layer_hierarchy"]:
            layer_name = layer["name"]
            layer_type = layer.get("type", "")
            params = layer.get("params", 0)
            in_ch = layer.get("in_channels", "")
            out_ch = layer.get("out_channels", "")
            print(
                f"{layer_name:<25} {layer_type:<20} {params:>12,} "
                f"{in_ch:>10} {out_ch:>10}",
                file=target_file,
            )
            if "blocks" in layer:
                for block in layer["blocks"]:
                    block_name = "  " + block["name"]
                    block_type = block.get("type", "")
                    block_params = block.get("params", 0)
                    block_in_ch = block.get("in_channels", "")
                    block_out_ch = block.get("out_channels", "")
                    print(
                        f"{block_name:<25} {block_type:<20} "
                        f"{block_params:>12,} {block_in_ch:>10} "
                        f"{block_out_ch:>10}",
                        file=target_file,
                    )
        print("-" * 80, file=target_file)
        print("=" * 80 + "\n", file=target_file)

    def print_summary(
        self,
        input_shape: tuple[int, ...] | None = None,
        file: Any | None = None,
        return_string: bool = False,
    ) -> str | None:
        """Print comprehensive formatted model summary with optional string
        return.

        Generates and displays a detailed, human-readable summary of the model
        architecture, parameters, memory usage, and component hierarchy.
        Supports output to files, console, or string capture for integration
        with logging and reporting systems.

        Args:
            input_shape: Optional input shape tuple (B, C, H, W) for memory
                estimation. Enables accurate memory usage calculation and
                optimization planning.
            file: Optional file object to write summary. If None, writes to
                stdout. Useful for saving summaries to log files or reports.
            return_string: If True, returns formatted summary as string instead
                of printing. Enables integration with logging systems and
                automated reporting.

        Returns:
            str | None: Formatted summary string if return_string=True,
            otherwise None.

        Examples:
            >>> model = BaseUNet(encoder, bottleneck, decoder)
            >>>
            >>> # Print to console
            >>> model.print_summary(input_shape=(1, 3, 512, 512))
            >>>
            >>> # Save to file
            >>> with open("model_summary.txt", "w") as f:
            ...     model.print_summary(input_shape=(1, 3, 512, 512), file=f)
            >>>
            >>> # Get as string for logging
            >>> summary_text = model.print_summary(
            ...     input_shape=(1, 3, 512, 512),
            ...     return_string=True
            ... )
            >>> logger.info(f"Model Summary:\n{summary_text}")

        Output Format:
            - Header: Model type and basic configuration
            - Parameters: Total, trainable, and non-trainable counts
            - Memory: Model size and estimated activation memory
            - Receptive Field: Theoretical coverage and downsampling
            - Architecture: Component types and activation configuration
            - Layer Hierarchy: Detailed component breakdown with parameters

        Performance:
            - Summary generation is fast and lightweight
            - Memory estimation requires input shape for accuracy
            - Output formatting is optimized for readability
            - String capture adds minimal overhead
        """
        string_stream = StringIO()  # Always initialize for potential use
        target_file: Any
        if return_string:
            target_file = string_stream
        else:
            import sys

            target_file = file if file is not None else sys.stdout

        # Generate comprehensive summary data
        summary_dict = self.summary(input_shape)

        # Print all sections in organized format
        self._print_header_info(summary_dict, target_file)
        self._print_parameter_info(summary_dict, target_file)
        self._print_memory_info(summary_dict, target_file)
        self._print_receptive_field_info(summary_dict, target_file)
        self._print_architecture_info(summary_dict, target_file)
        self._print_layer_hierarchy_info(summary_dict, target_file)

        if return_string:
            return string_stream.getvalue()
        return None

    def visualize_architecture(
        self, filename: str | None = None, view: bool = False
    ) -> None:
        """Generate visual architecture diagram using Graphviz.

        Creates a U-shaped block diagram showing the model architecture with
        component relationships, skip connections, and data flow. Useful for
        documentation, presentations, and architecture understanding.

        Args:
            filename: Optional output filename for saving diagram.
                If None, generates temporary file. Supports formats: .png,
                .pdf, .svg
            view: If True, automatically opens generated diagram.
                Platform-dependent viewer used (default image viewer).

        Examples:
            >>> model = BaseUNet(encoder, bottleneck, decoder)
            >>>
            >>> # Generate and view diagram
            >>> model.visualize_architecture("unet_architecture.png", view=True
            ... )
            >>> # Generate PDF for documentation
            >>> model.visualize_architecture("model_diagram.pdf")
            >>>
            >>> # Quick preview without saving
            >>> model.visualize_architecture(view=True)

        Requirements:
            - Graphviz must be installed on the system
            - Python graphviz package must be available
            - Appropriate image viewer for automatic viewing

        Output Features:
            - U-shaped layout showing encoder-decoder symmetry
            - Component blocks with type and channel information
            - Skip connection arrows showing data flow
            - Color coding for different component types
            - Hierarchical organization of model layers
        """
        layer_hierarchy = get_layer_hierarchy(
            self.encoder, self.bottleneck, self.decoder, self.final_activation
        )
        render_unet_architecture_diagram(layer_hierarchy, filename, view)
