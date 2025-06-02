"""
Registry system configuration for the model components.

Centralizes all registry initialization and management, ensuring consistent
component registration across the system. Provides access to registries for
encoders, decoders, bottlenecks, and specialized components like ConvLSTM,
SwinV2, ASPP, and CBAM attention.
"""

import logging
from typing import Any

from torch import nn

# Update the import to reflect the new structure
from .registry import Registry

# Create logger
log = logging.getLogger(__name__)

# Initialize registries for main component types
encoder_registry = Registry(nn.Module, "Encoder")
bottleneck_registry = Registry(nn.Module, "Bottleneck")
decoder_registry = Registry(nn.Module, "Decoder")
architecture_registry = Registry(nn.Module, "Architecture")

# Initialize registries for specialized components
component_registries = {
    "attention": Registry(base_class=nn.Module, name="Attention"),
    "convlstm": Registry(base_class=nn.Module, name="ConvLSTM"),
}


def register_component(
    registry_type: str, name: str | None = None, tags: list[str] | None = None
):
    """
    Decorator to register a component with specified registry type.

    Args:
        registry_type (str): Type of registry ('encoder', 'bottleneck',
                           'decoder', 'architecture', 'attention', 'convlstm')
        name (str, optional): Name to register the component with.
        tags (List[str], optional): Tags for categorizing components.

    Returns:
        callable: Decorator function

    Example:
        @register_component('encoder', name='SwinV2Tiny')
        class MySwinV2Encoder(EncoderBase):
            pass
    """
    registry_map = {
        "encoder": encoder_registry,
        "bottleneck": bottleneck_registry,
        "decoder": decoder_registry,
        "architecture": architecture_registry,
        "attention": component_registries.get("attention"),
        "convlstm": component_registries.get("convlstm"),
    }

    registry = registry_map.get(registry_type.lower())
    if registry is None:
        raise ValueError(
            f"Unknown or invalid registry type: {registry_type}. "
            f"Available types: {', '.join(registry_map.keys())}"
        )
    return registry.register(name=name, tags=tags)


def get_registry(registry_type: str) -> Registry[Any]:
    """
    Get a specific registry by type.

    Args:
        registry_type (str): Type of registry to retrieve.

    Returns:
        Registry: The requested registry.

    Raises:
        ValueError: If registry type is not recognized.
    """
    registry_map = {
        "encoder": encoder_registry,
        "bottleneck": bottleneck_registry,
        "decoder": decoder_registry,
        "architecture": architecture_registry,
        "attention": component_registries.get("attention"),
        "convlstm": component_registries.get("convlstm"),
    }

    registry = registry_map.get(registry_type.lower())
    if registry is None:
        raise ValueError(
            f"Unknown or invalid registry type: {registry_type}. "
            f"Available types: {', '.join(registry_map.keys())}"
        )
    from typing import cast

    return cast(Registry[Any], registry)


def list_available_components() -> dict[str, list[str]]:
    """
    List all registered components across all registries.

    Returns:
        Dict[str, List[str]]: Dictionary mapping registry types to component
                             lists.
    """
    result = {
        "encoders": encoder_registry.list_components(),
        "decoders": decoder_registry.list_components(),
        "bottlenecks": bottleneck_registry.list_components(),
        "architectures": architecture_registry.list_components(),
        "attention": [],
        "convlstm": [],
    }
    attention_registry = component_registries.get("attention")
    if isinstance(attention_registry, Registry):
        result["attention"] = attention_registry.list_components()
    convlstm_registry = component_registries.get("convlstm")
    if isinstance(convlstm_registry, Registry):
        result["convlstm"] = convlstm_registry.list_components()
    return result


def register_hybrid_architecture(
    name: str,
    encoder_type: str,
    bottleneck_type: str,
    decoder_type: str,
    tags: list[str] | None = None,
) -> bool:
    """
    Register a hybrid architecture model configuration.

    Args:
        name (str): Name for the hybrid architecture.
        encoder_type (str): Type of encoder used in the hybrid.
        bottleneck_type (str): Type of bottleneck used in the hybrid.
        decoder_type (str): Type of decoder used in the hybrid.
        tags (List[str], optional): Tags for categorizing the architecture.

    Returns:
        bool: True if registration was successful.

    Raises:
        ValueError: If component types don't exist in their respective
        registries.
    """
    # Verify components exist in respective registries
    if encoder_type not in encoder_registry:
        raise ValueError(
            f"Encoder type '{encoder_type}' not found in registry"
        )
    if bottleneck_type not in bottleneck_registry:
        raise ValueError(
            f"Bottleneck type '{bottleneck_type}' not found in registry"
        )
    if decoder_type not in decoder_registry:
        raise ValueError(
            f"Decoder type '{decoder_type}' not found in registry"
        )

    # Store hybrid architecture configuration
    # Creating a metadata dict with component types
    architecture_metadata = {
        "encoder_type": encoder_type,
        "bottleneck_type": bottleneck_type,
        "decoder_type": decoder_type,
    }

    # Register the hybrid architecture with its metadata as tags
    if tags is None:
        tags = []

    # Add component type tags
    component_tags = [
        f"encoder:{encoder_type}",
        f"bottleneck:{bottleneck_type}",
        f"decoder:{decoder_type}",
    ]
    tags.extend(component_tags)

    # Here we'll just use the architecture_registry and add the metadata as
    # tags to the architecture
    # This lets us query architectures by their components later
    log.info(
        f"Registering hybrid architecture '{name}' with components: "
        f"{architecture_metadata}"
    )

    # Registration would normally be done with a decorator, but since we're
    # just registering a configuration, we'll set it directly
    # In a real implementation, we might have a HybridArchitecture class
    architecture_registry.register(name=name, tags=tags)

    return True
