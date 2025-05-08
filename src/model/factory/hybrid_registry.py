"""
Hybrid Architecture Registry Module.

Provides advanced registration and management for hybrid architectures that
combine multiple component types. Includes:

1. Metadata structure for hybrid architectures
2. Validation systems for component compatibility
3. Query capabilities to search architectures by component types
4. Dependency management between components
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, field

# Actualizar importación para usar referencia relativa
from .registry_setup import architecture_registry

# Create logger
log = logging.getLogger(__name__)


@dataclass
class ComponentReference:
    """
    Reference to a registered component with metadata.

    Attributes:
        registry_type: Type of registry the component belongs to
        component_name: Name of the component in the registry
        optional: Whether this component is optional in the architecture
        tags: Tags associated with this component usage
    """
    registry_type: str
    component_name: str
    optional: bool = False
    tags: List[str] = field(default_factory=list)

    def validate(self) -> bool:
        """
        Validate that this component reference points to a valid registered
        component.

        Returns:
            bool: True if valid, raises ValueError otherwise
        """
        # Actualizar importación para usar referencia relativa
        from .registry_setup import get_registry

        try:
            registry = get_registry(self.registry_type)
            if self.component_name not in registry:
                raise ValueError(
                    f"Component '{self.component_name}' not found in "
                    f"'{self.registry_type}' registry"
                )
            return True
        except ValueError as e:
            if self.optional:
                log.warning(
                    f"Optional component '{self.component_name}' not found: "
                    f"{e}"
                )
                return False
            raise


@dataclass
class HybridArchitectureDescriptor:
    """
    Descriptor for a hybrid architecture with component relationships.

    Attributes:
        name: Name of the hybrid architecture
        components: Dictionary of component references by role
        metadata: Additional metadata for the architecture
        tags: Tags for categorizing this architecture
    """
    name: str
    components: Dict[str, ComponentReference]
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def validate(self) -> bool:
        """
        Validate all component references and their compatibility.

        Returns:
            bool: True if valid, raises ValueError otherwise
        """
        # Validate each component exists
        for role, component_ref in self.components.items():
            try:
                component_ref.validate()
            except ValueError as e:
                raise ValueError(
                    f"Invalid component for role '{role}' in "
                    f"architecture '{self.name}': {e}"
                )

        # Validate encoder, bottleneck, decoder compatibility if present
        # This is just a basic example - real validation would be more complex
        if all(k in self.components for k in ['encoder', 'bottleneck',
                                              'decoder']):
            # Additional compatibility checks could go here
            pass

        return True

    def get_all_component_names(self) -> List[str]:
        """Get names of all required components in this architecture."""
        return [
            c.component_name for c in self.components.values()
            if not c.optional
        ]

    def to_tag_list(self) -> List[str]:
        """
        Convert this descriptor to a list of tags for registration.

        Returns:
            List[str]: Tags representing this architecture's components
        """
        component_tags = []

        # Add role:component tags
        for role, component_ref in self.components.items():
            tag = f"{role}:{component_ref.component_name}"
            component_tags.append(tag)

        # Add general tags if any component has them
        for component_ref in self.components.values():
            for tag in component_ref.tags:
                if tag not in component_tags:
                    component_tags.append(tag)

        # Add the descriptor's own tags
        component_tags.extend(self.tags)

        return component_tags


class HybridRegistry:
    """
    Registry manager for hybrid architectures with advanced query capabilities.
    """

    def __init__(self):
        self._descriptors: Dict[str, HybridArchitectureDescriptor] = {}

    def register(self, descriptor: HybridArchitectureDescriptor) -> bool:
        """
        Register a hybrid architecture using its descriptor.

        Args:
            descriptor: Hybrid architecture descriptor

        Returns:
            bool: True if registration was successful
        """
        # Validate the architecture descriptor
        descriptor.validate()

        # Check for duplicate names
        if descriptor.name in self._descriptors:
            raise ValueError(
                f"Hybrid architecture '{descriptor.name}' already registered"
            )

        # Store the descriptor
        self._descriptors[descriptor.name] = descriptor

        # Register with the architecture registry for factory support
        tags = descriptor.to_tag_list()

        # Create a dummy class for registration since architecture_registry
        # expects a class to register
        # Actualizar importación para reflejar la nueva estructura
        from src.model.base.abstract import UNetBase

        class DummyArchitecture(UNetBase):
            """Dummy architecture class for registration purposes."""
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.name = descriptor.name
                self.components = descriptor.components

            def forward(self, x):
                return x

        # Use the decorator to register the dummy architecture
        architecture_registry.register(name=descriptor.name,
                                       tags=tags)(DummyArchitecture)

        log.info(
            f"Registered hybrid architecture '{descriptor.name}' with "
            f"{len(descriptor.components)} components"
        )

        return True

    def query_by_component(
        self,
        component_name: str,
        role: Optional[str] = None
    ) -> List[str]:
        """
        Query hybrid architectures that use a specific component.

        Args:
            component_name: Name of the component to query for
            role: Optional role the component must play

        Returns:
            List[str]: Names of matching architectures
        """
        matches = []

        for name, descriptor in self._descriptors.items():
            if role:
                # Check if component is used in the specified role
                if (role in descriptor.components and
                        descriptor.components[role].component_name ==
                        component_name):
                    matches.append(name)
            else:
                # Check if component is used in any role
                for component_ref in descriptor.components.values():
                    if component_ref.component_name == component_name:
                        matches.append(name)
                        break

        return matches

    def get_descriptor(self, name: str) -> HybridArchitectureDescriptor:
        """
        Get a hybrid architecture descriptor by name.

        Args:
            name: Name of the architecture

        Returns:
            HybridArchitectureDescriptor: The descriptor

        Raises:
            ValueError: If architecture not found
        """
        if name not in self._descriptors:
            raise ValueError(f"Hybrid architecture '{name}' not found")
        return self._descriptors[name]

    def list_architectures(self) -> List[str]:
        """
        List all registered hybrid architectures.

        Returns:
            List[str]: Names of all registered architectures
        """
        return list(self._descriptors.keys())


# Create a global instance of the hybrid registry
hybrid_registry = HybridRegistry()


def register_standard_hybrid(
    name: str,
    encoder_type: str,
    bottleneck_type: str,
    decoder_type: str,
    attention_type: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> bool:
    """
    Register a standard hybrid architecture with encoder, bottleneck, and
    decoder.

    This is a simplified interface for the most common case of hybrid
    architectures that follow the encoder-bottleneck-decoder pattern.

    Args:
        name: Name for the hybrid architecture
        encoder_type: Type of encoder used
        bottleneck_type: Type of bottleneck used
        decoder_type: Type of decoder used
        attention_type: Optional attention module type
        tags: Additional tags for the architecture

    Returns:
        bool: True if registration was successful
    """
    components = {
        'encoder': ComponentReference('encoder', encoder_type),
        'bottleneck': ComponentReference('bottleneck', bottleneck_type),
        'decoder': ComponentReference('decoder', decoder_type),
    }

    # Add attention if specified
    if attention_type:
        components['attention'] = ComponentReference(
            'attention', attention_type, optional=True
        )

    # Create the descriptor
    descriptor = HybridArchitectureDescriptor(
        name=name,
        components=components,
        tags=tags or []
    )

    # Register with the hybrid registry
    return hybrid_registry.register(descriptor)


def register_complex_hybrid(
    name: str,
    components: Dict[str, Tuple[str, str]],
    tags: Optional[List[str]] = None
) -> bool:
    """
    Register a complex hybrid architecture with custom component roles.

    Args:
        name: Name for the hybrid architecture
        components: Dictionary mapping role names to (registry_type,
            component_name)
        tags: Additional tags for the architecture

    Returns:
        bool: True if registration was successful
    """
    component_refs = {}

    for role, (registry_type, component_name) in components.items():
        component_refs[role] = ComponentReference(registry_type,
                                                  component_name)

    # Create the descriptor
    descriptor = HybridArchitectureDescriptor(
        name=name,
        components=component_refs,
        tags=tags or []
    )

    # Register with the hybrid registry
    return hybrid_registry.register(descriptor)


def query_architectures_by_component(
    component_name: str,
    role: Optional[str] = None
) -> List[str]:
    """
    Find hybrid architectures that use a specific component.

    Args:
        component_name: Name of the component to search for
        role: Optional specific role the component must play

    Returns:
        List[str]: Names of matching architectures
    """
    return hybrid_registry.query_by_component(component_name, role)


def query_architectures_by_tag(tag: str) -> List[str]:
    """
    Find hybrid architectures with a specific tag.

    Args:
        tag: Tag to search for

    Returns:
        List[str]: Names of matching architectures
    """
    matching = []

    for name, descriptor in hybrid_registry._descriptors.items():
        if tag in descriptor.tags or tag in descriptor.to_tag_list():
            matching.append(name)

    return matching
