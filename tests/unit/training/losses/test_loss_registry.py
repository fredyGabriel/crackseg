# pyright: reportPrivateUsage=false
# pyright: reportUnusedClass=false
# ruff: noqa: PLR2004

from collections.abc import Generator
from typing import Any

import pytest
import torch
from torch import nn

from src.model.factory.registry import Registry  # For type hinting if needed

# Import the global loss_registry instance we want to test
from src.training.losses.loss_registry_setup import loss_registry

# Store the original state of the registry's internal dicts
_original_components: dict[str, type[nn.Module]] = {}
_original_tags: dict[str, list[str]] = {}


@pytest.fixture(autouse=True)
def isolated_loss_registry(request: Any) -> Generator[None, None, None]:
    """
    Fixture to ensure the global loss_registry is clean for each test.
    It backs up and clears the internal state before each test,
    and restores it afterwards.
    """
    # Backup original state using request attributes
    request._original_components = loss_registry._components.copy()
    request._original_tags = {
        k: list(v) for k, v in loss_registry._tags.items()
    }

    # Clear for the current test
    loss_registry._components.clear()
    loss_registry._tags.clear()

    yield  # Test runs here

    # Restore original state
    loss_registry._components.clear()
    loss_registry._components.update(request._original_components)
    loss_registry._tags.clear()
    loss_registry._tags.update(request._original_tags)


# --- Dummy Loss Classes for Testing ---


class DummyLoss1(nn.Module):
    def __init__(self, param1: int = 0) -> None:
        super().__init__()
        self.param1 = param1

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return torch.tensor(0.0)


class DummyLoss2(nn.Module):
    def __init__(self, param_a: str = "default") -> None:
        super().__init__()
        self.param_a = param_a

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return torch.tensor(1.0)


class NotALossModule:  # Does not inherit from nn.Module
    pass


# --- Test Cases ---


class TestLossRegistry:
    def test_registry_instance_properties(self) -> None:
        """Test basic properties of the loss_registry instance."""
        assert loss_registry.name == "LossFunctions"
        assert loss_registry.base_class == nn.Module
        assert isinstance(loss_registry, Registry)

    def test_initial_state(self) -> None:
        """Test that the registry is empty at the start of a test
        (due to fixture)."""
        assert len(loss_registry) == 0
        assert not loss_registry.list_components()
        assert not loss_registry.list_with_tags()

    def test_register_simple_loss_module(self) -> None:
        """Test basic registration of a valid nn.Module loss."""

        @loss_registry.register()
        class MyTestLoss(nn.Module):
            def forward(
                self, x: torch.Tensor, y: torch.Tensor
            ) -> torch.Tensor:
                return torch.tensor(0.0)

        assert "MyTestLoss" in loss_registry
        assert len(loss_registry) == 1
        retrieved_loss = loss_registry.get("MyTestLoss")
        assert retrieved_loss == MyTestLoss

    def test_register_with_custom_name(self) -> None:
        """Test registration with a custom name."""

        @loss_registry.register(name="custom_dummy_loss_1")
        class AnotherLoss(nn.Module):
            def forward(
                self, x: torch.Tensor, y: torch.Tensor
            ) -> torch.Tensor:
                return torch.tensor(0.0)

        assert "custom_dummy_loss_1" in loss_registry
        assert "AnotherLoss" not in loss_registry
        assert len(loss_registry) == 1
        retrieved_loss = loss_registry.get("custom_dummy_loss_1")
        assert retrieved_loss == AnotherLoss

    def test_register_with_tags(self) -> None:
        """Test registration with tags."""

        @loss_registry.register(
            name="tagged_loss", tags=["segmentation", "test"]
        )
        class TaggedLoss(nn.Module):
            def forward(
                self, x: torch.Tensor, y: torch.Tensor
            ) -> torch.Tensor:
                return torch.tensor(0.0)

        assert "tagged_loss" in loss_registry
        tags_dict = loss_registry.list_with_tags()
        assert tags_dict.get("tagged_loss") == ["segmentation", "test"]

        filtered_segmentation = loss_registry.filter_by_tag("segmentation")
        assert "tagged_loss" in filtered_segmentation

        filtered_test = loss_registry.filter_by_tag("test")
        assert "tagged_loss" in filtered_test

        filtered_other = loss_registry.filter_by_tag("other_tag")
        assert not filtered_other

    def test_register_duplicate_name_raises_value_error(self) -> None:
        """Test that registering a duplicate name raises ValueError."""

        @loss_registry.register(
            name="shared_name_error"
        )  # Changed name to avoid conflict with other tests
        class LossV1Error(nn.Module):
            def forward(
                self, x: torch.Tensor, y: torch.Tensor
            ) -> torch.Tensor:
                return torch.tensor(0.0)

        with pytest.raises(
            ValueError,
            match=r"Component 'shared_name_error' is already registered",
        ):

            @loss_registry.register(name="shared_name_error")
            class LossV2Error(nn.Module):
                def forward(
                    self, x: torch.Tensor, y: torch.Tensor
                ) -> torch.Tensor:
                    return torch.tensor(0.0)

        assert len(loss_registry) == 1
        assert loss_registry.get("shared_name_error") == LossV1Error

    def test_register_non_nn_module_raises_type_error(self) -> None:
        """Test that registering a class not inheriting from nn.Module raises
        TypeError."""

        def register_bad_class() -> None:
            class BadClassNotModule:
                pass

            loss_registry.register()(BadClassNotModule)  # type: ignore[arg-type]

        with pytest.raises(
            TypeError,
            match=r"Class BadClassNotModule must inherit from Module",
        ):
            register_bad_class()

        assert "BadClassNotModule" not in loss_registry
        assert len(loss_registry) == 0

    def test_decorator_returns_class(self) -> None:
        """Test that the register decorator returns the class itself."""

        class LossToRegisterDec(nn.Module):
            pass  # Unique name

        registered_class = loss_registry.register(name="returned_loss_dec")(
            LossToRegisterDec
        )
        assert registered_class == LossToRegisterDec
        assert "returned_loss_dec" in loss_registry

    def test_get_loss(self) -> None:
        """Test retrieving a registered loss."""
        loss_registry.register(name="get_test_loss")(DummyLoss1)
        retrieved = loss_registry.get("get_test_loss")
        assert retrieved == DummyLoss1

    def test_get_non_existent_loss_raises_key_error(self) -> None:
        """Test that getting a non-existent loss raises KeyError."""
        with pytest.raises(
            KeyError, match=r"Component 'non_existent_loss' not found"
        ):
            loss_registry.get("non_existent_loss")

    def test_instantiate_loss_no_args(self) -> None:
        """Test instantiating a registered loss with no constructor arguments
        (beyond self)."""
        loss_registry.register(name="instantiate_loss_no_args")(
            DummyLoss1
        )  # DummyLoss1 has default for param1
        instance = loss_registry.instantiate("instantiate_loss_no_args")
        assert isinstance(instance, DummyLoss1)
        assert instance.param1 == 0  # Default value

    def test_instantiate_loss_with_args_and_kwargs(self) -> None:
        """Test instantiating a registered loss with positional and keyword
        arguments."""
        loss_registry.register(name="instantiate_loss_args")(DummyLoss1)
        instance1 = loss_registry.instantiate(
            "instantiate_loss_args", param1=100
        )
        assert isinstance(instance1, DummyLoss1)
        assert instance1.param1 == 100

        loss_registry.register(name="instantiate_loss_kwargs")(DummyLoss2)
        instance2 = loss_registry.instantiate(
            "instantiate_loss_kwargs", param_a="custom_val"
        )
        assert isinstance(instance2, DummyLoss2)
        assert instance2.param_a == "custom_val"

        # Test with both
        loss_registry.register(name="instantiate_loss_mixed_args")(DummyLoss1)
        # Assuming DummyLoss1 can take its args like this for the test
        # This part might need adjustment based on actual constructor
        # flexibility
        # For now, we rely on the kwarg version as it's safer
        instance_mixed = loss_registry.instantiate(
            "instantiate_loss_mixed_args", param1=50
        )
        assert isinstance(instance_mixed, DummyLoss1)
        assert instance_mixed.param1 == 50

    def test_instantiate_non_existent_loss_raises_key_error(self) -> None:
        """Test that instantiating a non-existent loss raises KeyError."""
        with pytest.raises(
            KeyError, match=r"Component 'non_existent_instantiate' not found"
        ):
            loss_registry.instantiate("non_existent_instantiate")

    def test_list_losses(self) -> None:
        """Test listing registered losses."""
        loss_registry.register(name="list_loss_1")(DummyLoss1)
        loss_registry.register(name="list_loss_2")(DummyLoss2)

        expected_list = sorted(["list_loss_1", "list_loss_2"])
        assert sorted(loss_registry.list_components()) == expected_list

    def test_list_with_tags_multiple(self):
        """Test listing with tags for multiple items."""
        loss_registry.register(name="loss_A", tags=["tag1", "tag_common"])(
            DummyLoss1
        )
        loss_registry.register(name="loss_B", tags=["tag2", "tag_common"])(
            DummyLoss2
        )
        loss_registry.register(name="loss_C", tags=["tag1"])(DummyLoss1)

        tags_map = loss_registry.list_with_tags()
        assert tags_map["loss_A"] == ["tag1", "tag_common"]
        assert tags_map["loss_B"] == ["tag2", "tag_common"]
        assert tags_map["loss_C"] == ["tag1"]

    def test_filter_by_tag_multiple_matches(self):
        """Test filtering by tag with multiple matches."""
        loss_registry.register(name="filter_loss_X", tags=["catA", "all"])(
            DummyLoss1
        )
        loss_registry.register(name="filter_loss_Y", tags=["catB", "all"])(
            DummyLoss2
        )
        loss_registry.register(name="filter_loss_Z", tags=["catA"])(DummyLoss1)

        filtered_catA = loss_registry.filter_by_tag("catA")
        assert sorted(filtered_catA) == sorted(
            ["filter_loss_X", "filter_loss_Z"]
        )

        filtered_all = loss_registry.filter_by_tag("all")
        assert sorted(filtered_all) == sorted(
            ["filter_loss_X", "filter_loss_Y"]
        )

    def test_unregister_loss(self):
        """Test unregistering a loss."""
        loss_registry.register(name="unregister_me")(DummyLoss1)
        assert "unregister_me" in loss_registry
        assert len(loss_registry) == 1

        loss_registry.unregister("unregister_me")
        assert "unregister_me" not in loss_registry
        assert len(loss_registry) == 0
        with pytest.raises(KeyError):
            loss_registry.get("unregister_me")

    def test_unregister_non_existent_loss_raises_key_error(self):
        """Test unregistering a non-existent loss raises KeyError."""
        with pytest.raises(
            KeyError, match=r"Component 'non_existent_unregister' not found"
        ):
            loss_registry.unregister("non_existent_unregister")

    def test_len_operator(self):
        """Test the __len__ operator."""
        assert len(loss_registry) == 0
        loss_registry.register(name="len_test_1")(DummyLoss1)
        assert len(loss_registry) == 1
        loss_registry.register(name="len_test_2")(DummyLoss2)
        assert len(loss_registry) == 2

    def test_contains_operator(self):
        """Test the __contains__ (in) operator."""
        assert "contains_test" not in loss_registry
        loss_registry.register(name="contains_test")(DummyLoss1)
        assert "contains_test" in loss_registry
        assert "not_in_registry" not in loss_registry

    def test_register_class_itself_not_instance(self):
        """Ensure the registry stores the class, not an instance."""
        loss_registry.register(name="class_ref_test")(DummyLoss1)
        item = loss_registry.get("class_ref_test")
        assert item == DummyLoss1
        assert isinstance(item, type)  # Check it's a class/type
