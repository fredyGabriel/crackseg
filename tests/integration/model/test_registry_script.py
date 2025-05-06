"""
Test script for the registry system.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.abspath("."))

# Import registry modules
from src.model.registry_setup import (  # noqa: E
    encoder_registry,
    bottleneck_registry,
    architecture_registry,
    component_registries,
    register_component,
    get_registry,
    list_available_components,
    register_hybrid_architecture
)

from src.model.components.registry_support import (  # noqa: E
    register_convlstm_components,
    register_swinv2_components,
    register_aspp_components,
    register_cbam_components,
    register_all_components
)

# Register all components
register_all_components()

# Print available components
print("Available components:", list_available_components())

# Test component registries
print("\nComponent registries:")
for name, registry in component_registries.items():
    print(f"- {name}: {registry.list()}")

print("\nRegistry completed successfully!")
