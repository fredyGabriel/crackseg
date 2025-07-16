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
from crackseg.model.components.registry_support import (
    register_all_components,
)
from crackseg.model.registry_setup import (
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
    print(
        f"- {name}: "
        f"{registry.list_available() if registry is not None else None}"
    )

print("\nRegistry completed successfully!")
