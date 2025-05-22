"""
Demo script for the registry system.

This script muestra cómo registrar y listar componentes en el sistema de
registros de modelos.
No es un test automatizado, sino una referencia para desarrolladores.
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.abspath("."))

# Import registry modules
from src.model.components.registry_support import (
    register_all_components,
)
from src.model.registry_setup import (
    component_registries,
    list_available_components,
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
