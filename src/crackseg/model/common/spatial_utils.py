"""Spatial calculation utilities for U-Net models.

This module provides functions for calculating output shapes of various
convolutional and pooling operations, as well as spatial compatibility
verification and tensor manipulation utilities.
"""

import torch
import torch.nn.functional as F


def calculate_output_shape_conv2d(
    input_height: int,
    input_width: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> tuple[int, int]:
    """Calculate output spatial dimensions for Conv2d layer.

    Args:
        input_height: Input tensor height
        input_width: Input tensor width
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding
        dilation: Convolution dilation

    Returns:
        Tuple of (output_height, output_width)
    """
    # Ensure all parameters are tuples
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # Calculate output dimensions using Conv2d formula
    output_height = (
        input_height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
    ) // stride[0] + 1

    output_width = (
        input_width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
    ) // stride[1] + 1

    return output_height, output_width


def calculate_output_shape_upsample_bilinear(
    input_height: int,
    input_width: int,
    scale_factor: float | tuple[float, float],
) -> tuple[int, int]:
    """Calculate output spatial dimensions for bilinear upsampling.

    Args:
        input_height: Input tensor height
        input_width: Input tensor width
        scale_factor: Upsampling scale factor

    Returns:
        Tuple of (output_height, output_width)
    """
    if isinstance(scale_factor, int | float):
        scale_factor = (float(scale_factor), float(scale_factor))

    output_height = int(input_height * scale_factor[0])
    output_width = int(input_width * scale_factor[1])

    return output_height, output_width


def calculate_output_shape_upsample_nearest(
    input_height: int,
    input_width: int,
    scale_factor: float | tuple[float, float],
) -> tuple[int, int]:
    """Calculate output spatial dimensions for nearest neighbor upsampling.

    Args:
        input_height: Input tensor height
        input_width: Input tensor width
        scale_factor: Upsampling scale factor

    Returns:
        Tuple of (output_height, output_width)
    """
    if isinstance(scale_factor, int | float):
        scale_factor = (float(scale_factor), float(scale_factor))

    output_height = int(input_height * scale_factor[0])
    output_width = int(input_width * scale_factor[1])

    return output_height, output_width


def calculate_output_shape_adaptive_avg_pool2d(
    output_size: int | tuple[int, int],
) -> tuple[int, int]:
    """Calculate output spatial dimensions for AdaptiveAvgPool2d.

    Args:
        output_size: Target output size

    Returns:
        Tuple of (output_height, output_width)
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    return output_size


def calculate_output_shape_conv_transpose2d(
    input_height: int,
    input_width: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    output_padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> tuple[int, int]:
    """Calculate output spatial dimensions for ConvTranspose2d layer.

    Args:
        input_height: Input tensor height
        input_width: Input tensor width
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding
        output_padding: Additional size added to output
        dilation: Convolution dilation

    Returns:
        Tuple of (output_height, output_width)
    """
    # Ensure all parameters are tuples
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # Calculate output dimensions using ConvTranspose2d formula
    output_height = (
        (input_height - 1) * stride[0]
        - 2 * padding[0]
        + dilation[0] * (kernel_size[0] - 1)
        + output_padding[0]
        + 1
    )

    output_width = (
        (input_width - 1) * stride[1]
        - 2 * padding[1]
        + dilation[1] * (kernel_size[1] - 1)
        + output_padding[1]
        + 1
    )

    return output_height, output_width


def verify_spatial_compatibility(
    tensor_a: torch.Tensor, tensor_b: torch.Tensor
) -> bool:
    """Verify that two tensors have compatible spatial dimensions.

    Args:
        tensor_a: First tensor
        tensor_b: Second tensor

    Returns:
        True if spatial dimensions match, False otherwise
    """
    if tensor_a.dim() < 2 or tensor_b.dim() < 2:
        return False

    return tensor_a.shape[-2:] == tensor_b.shape[-2:]


def pad_to_size(
    tensor: torch.Tensor, target_height: int, target_width: int
) -> torch.Tensor:
    """Pad tensor to target spatial size.

    Args:
        tensor: Input tensor with shape (..., H, W)
        target_height: Target height
        target_width: Target width

    Returns:
        Padded tensor with shape (..., target_height, target_width)
    """
    current_height, current_width = tensor.shape[-2:]

    if current_height >= target_height and current_width >= target_width:
        return tensor

    pad_height = max(0, target_height - current_height)
    pad_width = max(0, target_width - current_width)

    # Pad format: (left, right, top, bottom)
    padding = (0, pad_width, 0, pad_height)
    return F.pad(tensor, padding, mode="constant", value=0)


def crop_to_size(
    tensor: torch.Tensor, target_height: int, target_width: int
) -> torch.Tensor:
    """Crop tensor to target spatial size.

    Args:
        tensor: Input tensor with shape (..., H, W)
        target_height: Target height
        target_width: Target width

    Returns:
        Cropped tensor with shape (..., target_height, target_width)
    """
    current_height, current_width = tensor.shape[-2:]

    if current_height <= target_height and current_width <= target_width:
        return tensor

    start_h = (current_height - target_height) // 2
    start_w = (current_width - target_width) // 2

    end_h = start_h + target_height
    end_w = start_w + target_width

    return tensor[..., start_h:end_h, start_w:end_w]
