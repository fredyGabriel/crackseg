"""
Demo script for validating the hybrid architecture registry system.

This script shows how to register, list, and query hybrid architectures
in the registry system.
It is not an automated test, but a reference for developers.
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.abspath("."))

# Import registry modules
# Import base classes for creating mock components
from src.model.base import DecoderBase

# Import hybrid registry modules
from src.model.hybrid_registry import (
    hybrid_registry,
    query_architectures_by_component,
    query_architectures_by_tag,
    register_complex_hybrid,
    register_standard_hybrid,
)
from src.model.registry_setup import (
    architecture_registry,
    bottleneck_registry,
    decoder_registry,
    encoder_registry,
)

print("Testing Hybrid Registry Implementation (Subtask 21.2)\n")
print("-" * 50)


# Create and register mock components for testing
class MockDecoder(DecoderBase):
    def __init__(self, in_channels, out_channels):
        super().__init__()

    def forward(self, x, skip_connections=None):
        return x


# Register our mock components
decoder_registry.register(name="MockDecoder")(MockDecoder)
print("✓ Registered MockDecoder for testing")

# Step 1: Register test hybrid architectures
print("\nRegistering test hybrid architectures:")
print("-" * 50)

# First try registering a standard hybrid architecture
try:
    register_standard_hybrid(
        name="TestStandardHybrid",
        encoder_type="SwinV2" if "SwinV2" in encoder_registry else "ResNet",
        bottleneck_type=(
            "ASPPModule" if "ASPPModule" in bottleneck_registry else "Identity"
        ),
        decoder_type="MockDecoder",  # This should now work
        tags=["test", "hybrid"],
    )
    print("✓ Successfully registered standard hybrid architecture")
except Exception as e:
    print(f"× Error registering standard architecture: {e}")

# Then register a complex hybrid with custom roles
try:
    components = {
        "custom_encoder": (
            "encoder",
            "SwinV2" if "SwinV2" in encoder_registry else "ResNet",
        ),
        "custom_decoder": ("decoder", "MockDecoder"),
    }
    register_complex_hybrid(
        name="TestComplexHybrid",
        components=components,
        tags=["test", "complex", "custom-roles"],
    )
    print("✓ Successfully registered complex hybrid architecture")
except Exception as e:
    print(f"× Error registering complex architecture: {e}")

# Step 2: List all registered hybrid architectures
print("\nListing registered hybrid architectures:")
print("-" * 50)
architectures = hybrid_registry.list_architectures()
if architectures:
    for arch in architectures:
        print(f"- {arch}")
    print(f"✓ Found {len(architectures)} hybrid architectures")
else:
    print("× No hybrid architectures found")

# Step 3: Test querying architectures by component
print("\nQuerying architectures by component:")
print("-" * 50)
try:
    # Now we know MockDecoder exists
    component_name = "MockDecoder"
    matches = query_architectures_by_component(component_name)
    if matches:
        print(f"Architectures using {component_name}:")
        for arch in matches:
            print(f"- {arch}")
        print(f"✓ Found {len(matches)} matches")
    else:
        print(f"× No architectures found using {component_name}")
except Exception as e:
    print(f"× Error querying by component: {e}")

# Step 4: Test querying architectures by tag
print("\nQuerying architectures by tag:")
print("-" * 50)
try:
    # Use tags that should exist
    tag = "hybrid"
    matches = query_architectures_by_tag(tag)
    if matches:
        print(f"Architectures with tag '{tag}':")
        for arch in matches:
            print(f"- {arch}")
        print(f"✓ Found {len(matches)} matches")
    else:
        print(f"× No architectures found with tag '{tag}'")
except Exception as e:
    print(f"× Error querying by tag: {e}")

# Step 5: Verify integration with main architecture registry
print("\nVerifying integration with main architecture registry:")
print("-" * 50)
try:
    arch_registry_items = architecture_registry.list()
    # Check if hybrid architectures are also in the main registry
    hybrid_in_main = [
        arch for arch in architectures if arch in arch_registry_items
    ]
    if hybrid_in_main:
        print(
            "Hybrid architectures found in main registry: "
            f"{len(hybrid_in_main)}/{len(architectures)}"
        )
        print(
            "✓ Hybrid registry is correctly integrated with main architecture "
            "registry"
        )
    else:
        print("× No hybrid architectures found in main registry")
except Exception as e:
    print(f"× Error verifying integration: {e}")

print("\nHybrid Registry Test Complete")
print("-" * 50)
