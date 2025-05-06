"""
Tests for thread safety in registry operations.

These tests verify that registry operations are thread-safe and can be
performed concurrently without race conditions or data corruption.
"""

import threading
import random
from concurrent.futures import ThreadPoolExecutor
import torch.nn as nn

from src.model.registry import Registry


class SimpleComponent(nn.Module):
    """A simple component for testing registry thread safety."""
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        return x


class TestRegistryThreadSafety:
    """Test thread safety of the Registry class."""

    def setup_method(self):
        """Set up a test registry."""
        self.registry = Registry(nn.Module, "TestRegistry")
        self.registered_components = set()
        self.errors = []
        self.lock = threading.Lock()

    def register_component(self, name):
        """Thread-safe registration of a component."""
        try:
            # Create component with unique name
            component = type(f"Component{name}", (SimpleComponent,), {})

            # Register component
            self.registry.register(name=name)(component)

            # Track successful registrations
            with self.lock:
                self.registered_components.add(name)

        except Exception as e:
            with self.lock:
                self.errors.append((name, str(e)))

    def unregister_component(self, name):
        """Thread-safe unregistration of a component."""
        try:
            # Only attempt to unregister if we know it was registered
            with self.lock:
                if name in self.registered_components:
                    self.registry.unregister(name)
                    self.registered_components.remove(name)
        except Exception as e:
            with self.lock:
                self.errors.append((name, str(e)))

    def get_component(self, name):
        """Thread-safe retrieval of a component."""
        try:
            # Only attempt to get if we know it was registered
            with self.lock:
                if name in self.registered_components:
                    self.registry.get(name)
        except Exception as e:
            with self.lock:
                self.errors.append((name, str(e)))

    def test_concurrent_registration(self):
        """Test concurrent registration of components."""
        # Number of components to register
        num_components = 100

        # Create component names
        component_names = [f"TestComponent{i}" for i in range(num_components)]

        # Register components concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(self.register_component, component_names)

        # Check no errors occurred
        assert not self.errors, f"Errors occurred: {self.errors}"

        # Verify all components were registered
        for name in component_names:
            assert name in self.registry, f"Component {name} not registered"

        # Verify count matches
        assert len(self.registry) == num_components, (
            f"Expected {num_components} components, got {len(self.registry)}"
        )

    def test_concurrent_unregistration(self):
        """Test concurrent unregistration of components."""
        # First register components sequentially for setup
        num_components = 100
        component_names = [f"TestComponent{i}" for i in range(num_components)]

        for name in component_names:
            self.register_component(name)

        # Clear errors from registration phase
        self.errors = []

        # Unregister a subset of components concurrently
        to_unregister = component_names[:50]
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(self.unregister_component, to_unregister)

        # Check no errors occurred
        assert not self.errors, f"Errors occurred: {self.errors}"

        # Verify correct components were unregistered
        for name in to_unregister:
            assert name not in self.registry, (
                f"Component {name} still registered"
            )

        # Verify remaining components are still registered
        remaining = component_names[50:]
        for name in remaining:
            assert name in self.registry, f"Component {name} not registered"

        # Verify count matches
        assert len(self.registry) == len(remaining), (
            f"Expected {len(remaining)} components, got {len(self.registry)}"
        )

    def test_mixed_concurrent_operations(self):
        """Test concurrent registration, unregistration, and retrieval."""
        # Prepare component names
        num_components = 200
        base_names = [f"MixedComponent{i}" for i in range(num_components)]

        # Register initial batch sequentially
        initial_batch = base_names[:100]
        for name in initial_batch:
            self.register_component(name)

        # Clear errors from registration phase
        self.errors = []

        # Prepare concurrent operations
        operations = []

        # Add registrations
        for name in base_names[100:]:
            operations.append((self.register_component, name))

        # Add some unregistrations
        for name in random.sample(initial_batch, 50):
            operations.append((self.unregister_component, name))

        # Add some retrievals
        for _ in range(100):
            name = random.choice(initial_batch)
            operations.append((self.get_component, name))

        # Shuffle operations for more randomness
        random.shuffle(operations)

        # Execute mixed operations concurrently
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(op, arg) for op, arg in operations]
            for future in futures:
                future.result()  # Ensure all complete

        # Check no errors occurred
        assert not self.errors, f"Errors occurred: {self.errors}"

        # Verify registry internals are consistent
        for name in self.registered_components:
            assert name in self.registry, (
                f"Component {name} should be registered"
            )

        # Verify count matches registered set
        assert len(self.registry) == len(self.registered_components), (
            f"Registry has {len(self.registry)} items but tracked set has "
            f"{len(self.registered_components)}"
        )

    def test_concurrent_list_and_filter(self):
        """Test concurrent list and filter operations."""
        # Prepare tagged components
        num_components = 100
        tags = ["tagA", "tagB", "tagC"]

        # Register components with different tag combinations
        for i in range(num_components):
            name = f"TaggedComponent{i}"
            # Assign random tags
            component_tags = random.sample(tags, random.randint(1, len(tags)))
            component = type(name, (SimpleComponent,), {})
            self.registry.register(name=name, tags=component_tags)(component)
            self.registered_components.add(name)

        # Prepare operations to list and filter concurrently
        operations = []

        # Add list operations
        for _ in range(50):
            operations.append((self.registry.list, []))

        # Add list_with_tags operations
        for _ in range(50):
            operations.append((self.registry.list_with_tags, []))

        # Add filter_by_tag operations
        for _ in range(50):
            tag = random.choice(tags)
            operations.append((self.registry.filter_by_tag, [tag]))

        # Execute operations concurrently
        results = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for op, args in operations:
                futures.append(executor.submit(op, *args))

            for future in futures:
                results.append(future.result())

        # Verify results - we don't check specific values, just that operations
        # completed
        assert len(results) == len(operations), "Not all operations completed"

        # Ensure list results match registry size
        list_results = [
            r for r, (op, _) in zip(results, operations)
            if op == self.registry.list
        ]
        for result in list_results:
            assert len(result) == num_components, (
                "List result doesn't match component count"
            )
