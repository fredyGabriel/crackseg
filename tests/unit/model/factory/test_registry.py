"""
Unit tests for the Registry system.

Tests singleton pattern, component registration, retrieval, and thread safety
for the model component registry system.
"""

import threading
from typing import Any

import pytest
import torch
from torch import nn

from src.crackseg.model.factory.registry import Registry
from src.crackseg.model.factory.registry_setup import (
    architecture_registry,
    encoder_registry,
    get_registry,
)


class MockComponent(nn.Module):
    """Mock component for testing registry functionality."""

    def __init__(self, name: str = "test") -> None:
        super().__init__()
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TestRegistrySingleton:
    """Test singleton pattern behavior of registries."""

    def test_registry_instances_are_singletons(self) -> None:
        """Test that registry instances are singletons across imports."""
        # Import registries from different modules to test singleton behavior
        from src.crackseg.model.factory.registry_setup import (
            architecture_registry as arch_reg_1,
        )

        # Re-import to verify same instances
        from src.crackseg.model.factory.registry_setup import (
            architecture_registry as arch_reg_2,
        )
        from src.crackseg.model.factory.registry_setup import (
            encoder_registry as enc_reg_1,
        )
        from src.crackseg.model.factory.registry_setup import (
            encoder_registry as enc_reg_2,
        )

        # Verify same instances
        assert arch_reg_1 is arch_reg_2
        assert enc_reg_1 is enc_reg_2

        # Verify different registries are different instances
        assert arch_reg_1 is not enc_reg_1

    def test_registry_identity_preserved_across_operations(self) -> None:
        """Test that registry identity is preserved across operations."""
        registry_id_before = id(architecture_registry)

        # Perform some operations
        @architecture_registry.register("TestComponent")
        class TestComponent(MockComponent):
            pass

        registry_id_after = id(architecture_registry)

        # Registry should be the same instance
        assert registry_id_before == registry_id_after


class TestRegistryRegistration:
    """Test component registration functionality."""

    def test_basic_registration(self) -> None:
        """Test basic component registration with decorator."""
        test_registry = Registry(nn.Module, "TestRegistry")

        @test_registry.register("TestComponent")
        class TestComponent(MockComponent):
            pass

        assert "TestComponent" in test_registry
        assert test_registry.get("TestComponent") == TestComponent

    def test_registration_without_name(self) -> None:
        """Test registration using class name when no name provided."""
        test_registry = Registry(nn.Module, "TestRegistry")

        @test_registry.register()
        class TestComponent(MockComponent):
            pass

        assert "TestComponent" in test_registry
        assert test_registry.get("TestComponent") == TestComponent

    def test_registration_with_tags(self) -> None:
        """Test registration with tags."""
        test_registry = Registry(nn.Module, "TestRegistry")

        @test_registry.register("TestComponent", tags=["test", "mock"])
        class TestComponent(MockComponent):
            pass

        assert "TestComponent" in test_registry
        component_tags = test_registry._tags["TestComponent"]
        assert "test" in component_tags
        assert "mock" in component_tags

    def test_registration_force_overwrite(self) -> None:
        """Test force overwrite of existing registration."""
        test_registry = Registry(nn.Module, "TestRegistry")

        @test_registry.register("TestComponent")
        class TestComponent1(MockComponent):
            pass

        @test_registry.register("TestComponent", force=True)
        class TestComponent2(MockComponent):
            pass

        # Should be overwritten
        assert test_registry.get("TestComponent") == TestComponent2

    def test_registration_same_class_silent(self) -> None:
        """Test that re-registering the same class is allowed silently."""
        test_registry = Registry(nn.Module, "TestRegistry")

        @test_registry.register("TestComponent")
        class TestComponent(MockComponent):
            pass

        # Store the original class reference
        original_class = TestComponent

        # Re-register the same class (this should work silently)
        # Note: In Python, we can't redefine the same class name in the same
        # scope
        # So we test the behavior by calling the decorator again on the same
        # class
        decorated_class = test_registry.register("TestComponent")(
            TestComponent
        )

        # Should still work and be the same class
        assert test_registry.get("TestComponent") == original_class
        assert decorated_class == original_class

    def test_registration_different_class_error(self) -> None:
        """Test error when registering different class with same name."""
        test_registry = Registry(nn.Module, "TestRegistry")

        @test_registry.register("TestComponent")
        class TestComponent1(MockComponent):
            pass

        # Try to register different class with same name
        with pytest.raises(
            ValueError, match="already registered with a different class"
        ):

            @test_registry.register("TestComponent")
            class TestComponent2(MockComponent):
                pass

    def test_registration_invalid_base_class(self) -> None:
        """Test error when registering class that doesn't inherit from base
        class."""
        test_registry = Registry(nn.Module, "TestRegistry")

        class InvalidComponent:  # Doesn't inherit from nn.Module
            pass

        with pytest.raises(TypeError, match="must inherit from"):
            # Use type: ignore to bypass type checking for this test case
            test_registry.register("InvalidComponent")(InvalidComponent)  # type: ignore


class TestRegistryRetrieval:
    """Test component retrieval functionality."""

    def test_get_existing_component(self) -> None:
        """Test retrieving an existing component."""
        test_registry = Registry(nn.Module, "TestRegistry")

        @test_registry.register("TestComponent")
        class TestComponent(MockComponent):
            pass

        retrieved = test_registry.get("TestComponent")
        assert retrieved == TestComponent

    def test_get_nonexistent_component(self) -> None:
        """Test error when retrieving non-existent component."""
        test_registry = Registry(nn.Module, "TestRegistry")

        with pytest.raises(KeyError, match="not found"):
            test_registry.get("NonExistentComponent")

    def test_list_components(self) -> None:
        """Test listing all registered components."""
        test_registry = Registry(nn.Module, "TestRegistry")

        @test_registry.register("Component1")
        class Component1(MockComponent):
            pass

        @test_registry.register("Component2")
        class Component2(MockComponent):
            pass

        components = test_registry.list_components()
        assert "Component1" in components
        assert "Component2" in components
        assert len(components) == 2

    def test_list_with_tags(self) -> None:
        """Test listing components with their tags."""
        test_registry = Registry(nn.Module, "TestRegistry")

        @test_registry.register("Component1", tags=["tag1", "tag2"])
        class Component1(MockComponent):
            pass

        @test_registry.register("Component2", tags=["tag2"])
        class Component2(MockComponent):
            pass

        components_with_tags = test_registry.list_with_tags()
        assert "Component1" in components_with_tags
        assert "Component2" in components_with_tags
        assert components_with_tags["Component1"] == ["tag1", "tag2"]
        assert components_with_tags["Component2"] == ["tag2"]

    def test_filter_by_tag(self) -> None:
        """Test filtering components by tag."""
        test_registry = Registry(nn.Module, "TestRegistry")

        @test_registry.register("Component1", tags=["tag1", "tag2"])
        class Component1(MockComponent):
            pass

        @test_registry.register("Component2", tags=["tag2"])
        class Component2(MockComponent):
            pass

        @test_registry.register("Component3", tags=["tag3"])
        class Component3(MockComponent):
            pass

        tag2_components = test_registry.filter_by_tag("tag2")
        assert "Component1" in tag2_components
        assert "Component2" in tag2_components
        assert "Component3" not in tag2_components
        assert len(tag2_components) == 2

    def test_contains_operator(self) -> None:
        """Test the 'in' operator for registry."""
        test_registry = Registry(nn.Module, "TestRegistry")

        @test_registry.register("TestComponent")
        class TestComponent(MockComponent):
            pass

        assert "TestComponent" in test_registry
        assert "NonExistent" not in test_registry

    def test_length_operator(self) -> None:
        """Test the len() operator for registry."""
        test_registry = Registry(nn.Module, "TestRegistry")

        assert len(test_registry) == 0

        @test_registry.register("Component1")
        class Component1(MockComponent):
            pass

        assert len(test_registry) == 1

        @test_registry.register("Component2")
        class Component2(MockComponent):
            pass

        assert len(test_registry) == 2


class TestRegistryInstantiation:
    """Test component instantiation functionality."""

    def test_instantiate_component(self) -> None:
        """Test instantiating a registered component."""
        test_registry = Registry(nn.Module, "TestRegistry")

        @test_registry.register("TestComponent")
        class TestComponent(MockComponent):
            def __init__(self, value: int = 42) -> None:
                super().__init__()
                self.value = value

        instance = test_registry.instantiate("TestComponent", value=100)
        assert isinstance(instance, TestComponent)
        assert instance.value == 100

    def test_instantiate_with_default_args(self) -> None:
        """Test instantiation with default arguments."""
        test_registry = Registry(nn.Module, "TestRegistry")

        @test_registry.register("TestComponent")
        class TestComponent(MockComponent):
            def __init__(self, value: int = 42) -> None:
                super().__init__()
                self.value = value

        instance = test_registry.instantiate("TestComponent")
        assert isinstance(instance, TestComponent)
        assert instance.value == 42

    def test_instantiate_nonexistent_component(self) -> None:
        """Test error when instantiating non-existent component."""
        test_registry = Registry(nn.Module, "TestRegistry")

        with pytest.raises(KeyError, match="not found"):
            test_registry.instantiate("NonExistentComponent")


class TestRegistryThreadSafety:
    """Test thread safety of registry operations."""

    def test_concurrent_registration(self) -> None:
        """Test concurrent registration from multiple threads."""
        test_registry = Registry(nn.Module, "TestRegistry")
        results: list[bool] = []

        def register_component(component_id: int) -> None:
            try:

                @test_registry.register(f"Component{component_id}")
                class TestComponent(MockComponent):
                    def __init__(self) -> None:
                        super().__init__()
                        self.id = component_id

                results.append(True)
            except Exception:
                results.append(False)

        # Create multiple threads registering components
        threads = []
        for i in range(10):
            thread = threading.Thread(target=register_component, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All registrations should succeed
        assert all(results)
        assert len(test_registry) == 10

    def test_concurrent_retrieval(self) -> None:
        """Test concurrent retrieval from multiple threads."""
        test_registry = Registry(nn.Module, "TestRegistry")

        # Register some components first
        @test_registry.register("Component1")
        class Component1(MockComponent):
            pass

        @test_registry.register("Component2")
        class Component2(MockComponent):
            pass

        results: list[Any] = []

        def retrieve_component(component_name: str) -> None:
            try:
                component = test_registry.get(component_name)
                results.append(component)
            except Exception as e:
                results.append(e)

        # Create multiple threads retrieving components
        threads = []
        for i in range(20):  # 10 each for Component1 and Component2
            component_name = f"Component{(i % 2) + 1}"
            thread = threading.Thread(
                target=retrieve_component, args=(component_name,)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All retrievals should succeed
        assert len(results) == 20
        assert all(isinstance(result, type) for result in results)

    def test_concurrent_listing(self) -> None:
        """Test concurrent listing operations."""
        test_registry = Registry(nn.Module, "TestRegistry")

        # Register some components
        for i in range(5):

            @test_registry.register(f"Component{i}")
            class TestComponent(MockComponent):
                pass

        results: list[list[str]] = []

        def list_components() -> None:
            try:
                components = test_registry.list_components()
                results.append(components)
            except Exception:
                results.append([])

        # Create multiple threads listing components
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=list_components)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All listings should succeed and be consistent
        assert len(results) == 10
        for result in results:
            assert len(result) == 5
            assert all(f"Component{i}" in result for i in range(5))


class TestRegistryProperties:
    """Test registry properties and metadata."""

    def test_registry_name_property(self) -> None:
        """Test registry name property."""
        test_registry = Registry(nn.Module, "TestRegistry")
        assert test_registry.name == "TestRegistry"

    def test_registry_base_class_property(self) -> None:
        """Test registry base class property."""
        test_registry = Registry(nn.Module, "TestRegistry")
        assert test_registry.base_class == nn.Module

    def test_registry_repr(self) -> None:
        """Test registry string representation."""
        test_registry = Registry(nn.Module, "TestRegistry")
        repr_str = repr(test_registry)
        assert "TestRegistry" in repr_str
        assert "components=[]" in repr_str


class TestRegistrySetup:
    """Test registry setup and utility functions."""

    def test_get_registry_valid_type(self) -> None:
        """Test getting registry with valid type."""
        registry = get_registry("encoder")
        assert registry is encoder_registry

    def test_get_registry_invalid_type(self) -> None:
        """Test error when getting registry with invalid type."""
        with pytest.raises(
            ValueError, match="Unknown or invalid registry type"
        ):
            get_registry("invalid_type")

    def test_registry_setup_initialization(self) -> None:
        """Test that registry setup creates all expected registries."""
        from src.crackseg.model.factory.registry_setup import (
            bottleneck_registry,
            decoder_registry,
        )

        # Verify all main registries exist
        assert encoder_registry is not None
        assert bottleneck_registry is not None
        assert decoder_registry is not None
        assert architecture_registry is not None

        # Verify they have correct names
        assert encoder_registry.name == "Encoder"
        assert bottleneck_registry.name == "Bottleneck"
        assert decoder_registry.name == "Decoder"
        assert architecture_registry.name == "Architecture"


class TestRegistryUnregister:
    """Test component unregistration functionality."""

    def test_unregister_existing_component(self) -> None:
        """Test unregistering an existing component."""
        test_registry = Registry(nn.Module, "TestRegistry")

        @test_registry.register("TestComponent")
        class TestComponent(MockComponent):
            pass

        assert "TestComponent" in test_registry

        test_registry.unregister("TestComponent")
        assert "TestComponent" not in test_registry

    def test_unregister_nonexistent_component(self) -> None:
        """Test error when unregistering non-existent component."""
        test_registry = Registry(nn.Module, "TestRegistry")

        with pytest.raises(KeyError, match="not found"):
            test_registry.unregister("NonExistentComponent")

    def test_unregister_affects_length(self) -> None:
        """Test that unregistering affects registry length."""
        test_registry = Registry(nn.Module, "TestRegistry")

        @test_registry.register("Component1")
        class Component1(MockComponent):
            pass

        @test_registry.register("Component2")
        class Component2(MockComponent):
            pass

        assert len(test_registry) == 2

        test_registry.unregister("Component1")
        assert len(test_registry) == 1
        assert "Component2" in test_registry
