"""Bottleneck components for the CrackSeg project."""

from src.model.bottleneck.cnn_bottleneck import BottleneckBlock
from src.model.components.aspp import ASPPModule

__all__ = ['BottleneckBlock', 'ASPPModule']
