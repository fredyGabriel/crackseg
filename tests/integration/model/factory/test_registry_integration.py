"""
Integration tests for the Registry system.

Tests complete component registration workflows, instantiation from registry,
and fallback behavior for the model component registry system.
"""

import pytest
import torch
from torch import nn

from src.crackseg.model.factory.registry import Registry
from src.crackseg.model.factory.registry_setup import (
    encoder_registry,
    get_registry,
    register_component,
)


class MockEncoder(nn.Module):
    """Mock encoder for testing integration."""

    def __init__(self, in_channels: int = 3, out_channels: int = 64) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MockDecoder(nn.Module):
    """Mock decoder for testing integration."""

    def __init__(self, in_channels: int = 64, out_channels: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MockArchitecture(nn.Module):
    """Mock architecture for testing integration."""

    def __init__(self, encoder: MockEncoder, decoder: MockDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.decoder(features)


class TestRegistryIntegration:
    """Test integration workflows for the registry system."""

    def test_complete_component_registration_workflow(self) -> None:
        """Test complete workflow of registering and using components."""
        # Create test registries
        test_encoder_registry = Registry(nn.Module, "TestEncoder")
        test_decoder_registry = Registry(nn.Module, "TestDecoder")
        test_arch_registry = Registry(nn.Module, "TestArchitecture")

        # Register components
        @test_encoder_registry.register("TestEncoder")
        class TestEncoder(MockEncoder):
            pass

        @test_decoder_registry.register("TestDecoder")
        class TestDecoder(MockDecoder):
            pass

        @test_arch_registry.register("TestArchitecture")
        class TestArchitecture(MockArchitecture):
            pass

        # Verify registration
        assert "TestEncoder" in test_encoder_registry
        assert "TestDecoder" in test_decoder_registry
        assert "TestArchitecture" in test_arch_registry

        # Instantiate components
        encoder = test_encoder_registry.instantiate("TestEncoder")
        decoder = test_decoder_registry.instantiate("TestDecoder")

        # Verify instantiation
        assert isinstance(encoder, TestEncoder)
        assert isinstance(decoder, TestDecoder)

        # Test forward pass
        input_tensor = torch.randn(1, 3, 64, 64)
        encoder_output = encoder(input_tensor)
        assert encoder_output.shape == (1, 64, 64, 64)

    def test_registry_utility_functions(self) -> None:
        """Test utility functions for registry management."""
        # Test get_registry function
        encoder_reg = get_registry("encoder")
        assert encoder_reg is encoder_registry

        # Test register_component decorator
        @register_component("encoder", name="IntegrationTestEncoder")
        class IntegrationTestEncoder(MockEncoder):
            pass

        # Verify registration through utility function
        assert "IntegrationTestEncoder" in encoder_registry
        assert (
            encoder_registry.get("IntegrationTestEncoder")
            == IntegrationTestEncoder
        )

    def test_component_instantiation_from_registry(self) -> None:
        """Test instantiating components with various parameters."""
        test_registry = Registry(nn.Module, "TestRegistry")

        @test_registry.register("ParameterizedComponent")
        class ParameterizedComponent(nn.Module):
            def __init__(
                self, param1: int = 10, param2: str = "default"
            ) -> None:
                super().__init__()
                self.param1 = param1
                self.param2 = param2
                self.layer = nn.Linear(10, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.layer(x)

        # Test instantiation with default parameters
        component1 = test_registry.instantiate("ParameterizedComponent")
        assert component1.param1 == 10
        assert component1.param2 == "default"

        # Test instantiation with custom parameters
        component2 = test_registry.instantiate(
            "ParameterizedComponent", param1=42, param2="custom"
        )
        assert component2.param1 == 42
        assert component2.param2 == "custom"

    def test_registry_component_listing_and_filtering(self) -> None:
        """Test listing and filtering components with tags."""
        test_registry = Registry(nn.Module, "TestRegistry")

        # Register components with different tags
        @test_registry.register("Component1", tags=["tag1", "tag2"])
        class Component1(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        @test_registry.register("Component2", tags=["tag2", "tag3"])
        class Component2(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        @test_registry.register("Component3", tags=["tag1", "tag3"])
        class Component3(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        # Test listing all components
        all_components = test_registry.list_components()
        assert len(all_components) == 3
        assert "Component1" in all_components
        assert "Component2" in all_components
        assert "Component3" in all_components

        # Test listing with tags
        components_with_tags = test_registry.list_with_tags()
        assert components_with_tags["Component1"] == ["tag1", "tag2"]
        assert components_with_tags["Component2"] == ["tag2", "tag3"]
        assert components_with_tags["Component3"] == ["tag1", "tag3"]

        # Test filtering by tag
        tag1_components = test_registry.filter_by_tag("tag1")
        assert len(tag1_components) == 2
        assert "Component1" in tag1_components
        assert "Component3" in tag1_components

        tag2_components = test_registry.filter_by_tag("tag2")
        assert len(tag2_components) == 2
        assert "Component1" in tag2_components
        assert "Component2" in tag2_components

    def test_registry_error_handling(self) -> None:
        """Test error handling in registry operations."""
        test_registry = Registry(nn.Module, "TestRegistry")

        # Test getting non-existent component
        with pytest.raises(KeyError, match="not found"):
            test_registry.get("NonExistentComponent")

        # Test instantiating non-existent component
        with pytest.raises(KeyError, match="not found"):
            test_registry.instantiate("NonExistentComponent")

        # Test unregistering non-existent component
        with pytest.raises(KeyError, match="not found"):
            test_registry.unregister("NonExistentComponent")

        # Test getting registry with invalid type
        with pytest.raises(
            ValueError, match="Unknown or invalid registry type"
        ):
            get_registry("invalid_type")

    def test_registry_thread_safety_integration(self) -> None:
        """Test thread safety in integration scenarios."""
        import threading
        import time

        test_registry = Registry(nn.Module, "TestRegistry")
        results: list[bool] = []

        def register_and_instantiate(component_id: int) -> None:
            try:
                # Register component
                @test_registry.register(f"ThreadComponent{component_id}")
                class ThreadComponent(nn.Module):
                    def __init__(self) -> None:
                        super().__init__()
                        self.id = component_id

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        return x

                # Small delay to increase chance of race conditions
                time.sleep(0.001)

                # Instantiate component
                instance = test_registry.instantiate(
                    f"ThreadComponent{component_id}"
                )
                assert isinstance(instance, ThreadComponent)
                assert instance.id == component_id

                results.append(True)
            except Exception:
                results.append(False)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=register_and_instantiate, args=(i,)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All operations should succeed
        assert all(results)
        assert len(test_registry) == 5

    def test_registry_persistence_across_imports(self) -> None:
        """
        Test that registry state persists across different import contexts.
        """
        # Import registries from different modules
        from src.crackseg.model.factory.registry_setup import (
            architecture_registry as arch_reg_1,
        )

        # Register a component
        @arch_reg_1.register("PersistenceTestComponent")
        class PersistenceTestComponent(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        # Re-import to verify persistence
        from src.crackseg.model.factory.registry_setup import (
            architecture_registry as arch_reg_2,
        )

        # Verify component is still available
        assert "PersistenceTestComponent" in arch_reg_2
        assert (
            arch_reg_2.get("PersistenceTestComponent")
            == PersistenceTestComponent
        )

        # Verify same registry instance
        assert arch_reg_1 is arch_reg_2

    def test_registry_component_unregistration(self) -> None:
        """Test unregistering components and verifying cleanup."""
        test_registry = Registry(nn.Module, "TestRegistry")

        # Register components
        @test_registry.register("ComponentToRemove")
        class ComponentToRemove(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        @test_registry.register("ComponentToKeep")
        class ComponentToKeep(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        # Verify initial state
        assert len(test_registry) == 2
        assert "ComponentToRemove" in test_registry
        assert "ComponentToKeep" in test_registry

        # Unregister one component
        test_registry.unregister("ComponentToRemove")

        # Verify cleanup
        assert len(test_registry) == 1
        assert "ComponentToRemove" not in test_registry
        assert "ComponentToKeep" in test_registry

        # Verify tags are also cleaned up
        components_with_tags = test_registry.list_with_tags()
        assert "ComponentToRemove" not in components_with_tags
        assert "ComponentToKeep" in components_with_tags

    def test_registry_force_overwrite_behavior(self) -> None:
        """Test force overwrite behavior in integration scenarios."""
        test_registry = Registry(nn.Module, "TestRegistry")

        # Register initial component
        @test_registry.register("OverwriteTestComponent")
        class OriginalComponent(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2

        # Verify original registration
        assert test_registry.get("OverwriteTestComponent") == OriginalComponent

        # Force overwrite with new component
        @test_registry.register("OverwriteTestComponent", force=True)
        class NewComponent(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 3

        # Verify overwrite
        assert test_registry.get("OverwriteTestComponent") == NewComponent

        # Test instantiation uses new component
        instance = test_registry.instantiate("OverwriteTestComponent")
        assert isinstance(instance, NewComponent)

        # Test forward pass behavior
        input_tensor = torch.tensor([1.0])
        output = instance(input_tensor)
        assert output.item() == 3.0  # New component multiplies by 3
