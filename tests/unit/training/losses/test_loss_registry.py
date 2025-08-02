# pyright: reportPrivateUsage=false
# pyright: reportUnusedClass=false
# ruff: noqa: PLR2004

from collections.abc import Generator
from typing import Any

import pytest
import torch
from torch import nn

from crackseg.model.factory.registry import Registry
from crackseg.training.losses.loss_registry_setup import loss_registry

# Store the original state of the registry's internal dicts
_original_components: dict[str, type[nn.Module]] = {}
_original_tags: dict[str, list[str]] = {}


@pytest.fixture(autouse=True)
def isolated_loss_registry(
    request: Any,
) -> Generator[Registry[nn.Module], None, None]:
    """
    Fixture to ensure the global loss_registry is clean for each test.

    It backs up and clears the internal state before each test,
    and restores it afterwards.
    """
    # Backup original state and explicitly type them
    original_components: dict[str, type[nn.Module]] = (
        loss_registry._components.copy()
    )
    original_tags: dict[str, list[str]] = {
        k: list(v) for k, v in loss_registry._tags.items()
    }

    # Clear for the current test
    loss_registry._components.clear()  # type: ignore[attr-defined]
    loss_registry._tags.clear()  # type: ignore[attr-defined]

    yield loss_registry  # Test runs here, providing the registry instance

    # Restore original state
    loss_registry._components.clear()  # type: ignore[attr-defined]
    loss_registry._tags.clear()  # type: ignore[attr-defined]
    loss_registry._components.update(original_components)
    loss_registry._tags.update(original_tags)


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
    def test_registry_instance_properties(
        self, isolated_loss_registry: Registry[nn.Module]
    ) -> None:
        """Test basic properties of the loss_registry instance."""
        assert isolated_loss_registry.name == "LossFunctions"
        assert isolated_loss_registry.base_class is nn.Module
        assert isinstance(isolated_loss_registry, Registry)

    def test_initial_state(
        self, isolated_loss_registry: Registry[nn.Module]
    ) -> None:
        """Test that the registry is empty at the start of a test."""
        assert len(isolated_loss_registry) == 0
        assert not isolated_loss_registry.list_components()
        assert not isolated_loss_registry.list_with_tags()

    def test_register_simple_loss_module(
        self, isolated_loss_registry: Registry[nn.Module]
    ) -> None:
        """Test basic registration of a valid nn.Module loss."""

        @isolated_loss_registry.register()
        class MyTestLoss(nn.Module):
            def forward(
                self, x: torch.Tensor, y: torch.Tensor
            ) -> torch.Tensor:
                return torch.tensor(0.0)

        assert "MyTestLoss" in isolated_loss_registry
        assert len(isolated_loss_registry) == 1
        retrieved_loss = isolated_loss_registry.get("MyTestLoss")
        assert retrieved_loss == MyTestLoss

    def test_register_with_custom_name(
        self, isolated_loss_registry: Registry[nn.Module]
    ) -> None:
        """Test registration with a custom name."""

        @isolated_loss_registry.register(name="custom_dummy_loss_1")
        class AnotherLoss(nn.Module):
            def forward(
                self, x: torch.Tensor, y: torch.Tensor
            ) -> torch.Tensor:
                return torch.tensor(0.0)

        assert "custom_dummy_loss_1" in isolated_loss_registry
        assert "AnotherLoss" not in isolated_loss_registry
        assert len(isolated_loss_registry) == 1
        retrieved_loss = isolated_loss_registry.get("custom_dummy_loss_1")
        assert retrieved_loss == AnotherLoss

    def test_register_with_tags(
        self, isolated_loss_registry: Registry[nn.Module]
    ) -> None:
        """Test registration with tags."""

        @isolated_loss_registry.register(
            name="tagged_loss", tags=["segmentation", "test"]
        )
        class TaggedLoss(nn.Module):
            def forward(
                self, x: torch.Tensor, y: torch.Tensor
            ) -> torch.Tensor:
                return torch.tensor(0.0)

        assert "tagged_loss" in isolated_loss_registry
        tags_dict = isolated_loss_registry.list_with_tags()
        assert tags_dict.get("tagged_loss") == ["segmentation", "test"]

        filtered_segmentation = isolated_loss_registry.filter_by_tag(
            "segmentation"
        )
        assert "tagged_loss" in filtered_segmentation

        filtered_test = isolated_loss_registry.filter_by_tag("test")
        assert "tagged_loss" in filtered_test

        filtered_other = isolated_loss_registry.filter_by_tag("other_tag")
        assert not filtered_other

    def test_register_duplicate_name_raises_value_error(
        self, isolated_loss_registry: Registry[nn.Module]
    ) -> None:
        """Test that registering a duplicate name raises ValueError."""

        @isolated_loss_registry.register(
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

            @isolated_loss_registry.register(name="shared_name_error")
            class LossV2Error(nn.Module):
                def forward(
                    self, x: torch.Tensor, y: torch.Tensor
                ) -> torch.Tensor:
                    return torch.tensor(0.0)

        assert len(isolated_loss_registry) == 1
        assert isolated_loss_registry.get("shared_name_error") == LossV1Error

    def test_register_non_nn_module_raises_type_error(
        self, isolated_loss_registry: Registry[nn.Module]
    ) -> None:
        """Test that registering a class not inheriting from nn.Module raises
        TypeError."""

        def register_bad_class() -> None:
            class BadClassNotModule:
                pass

            isolated_loss_registry.register()(BadClassNotModule)  # type: ignore[arg-type]

        with pytest.raises(
            TypeError,
            match=r"Class BadClassNotModule must inherit from Module",
        ):
            register_bad_class()

        assert "BadClassNotModule" not in isolated_loss_registry
        assert len(isolated_loss_registry) == 0

    def test_decorator_returns_class(
        self, isolated_loss_registry: Registry[nn.Module]
    ) -> None:
        """Test that the register decorator returns the class itself."""

        class LossToRegisterDec(nn.Module):
            pass  # Unique name

        registered_class = isolated_loss_registry.register(
            name="returned_loss_dec"
        )(LossToRegisterDec)
        assert registered_class == LossToRegisterDec
        assert "returned_loss_dec" in isolated_loss_registry

    def test_get_loss(
        self, isolated_loss_registry: Registry[nn.Module]
    ) -> None:
        """Test retrieving a registered loss."""
        isolated_loss_registry.register(name="get_test_loss")(DummyLoss1)
        retrieved = isolated_loss_registry.get("get_test_loss")
        assert retrieved == DummyLoss1

    def test_get_non_existent_loss_raises_key_error(
        self, isolated_loss_registry: Registry[nn.Module]
    ) -> None:
        """Test that getting a non-existent loss raises KeyError."""
        with pytest.raises(
            KeyError, match=r"Component 'non_existent_loss' not found"
        ):
            isolated_loss_registry.get("non_existent_loss")

    def test_instantiate_loss_no_args(
        self, isolated_loss_registry: Registry[nn.Module]
    ) -> None:
        """Test instantiating a registered loss with no constructor arguments
        (beyond self)."""
        isolated_loss_registry.register(name="instantiate_loss_no_args")(
            DummyLoss1
        )  # DummyLoss1 has default for param1
        instance = isolated_loss_registry.instantiate(
            "instantiate_loss_no_args"
        )
        assert isinstance(instance, DummyLoss1)
        assert instance.param1 == 0  # Default value

    def test_instantiate_loss_with_args_and_kwargs(
        self, isolated_loss_registry: Registry[nn.Module]
    ) -> None:
        """Test instantiating a registered loss with positional and keyword
        arguments."""
        isolated_loss_registry.register(name="instantiate_loss_args")(
            DummyLoss1
        )
        instance1 = isolated_loss_registry.instantiate(
            "instantiate_loss_args", param1=100
        )
        assert isinstance(instance1, DummyLoss1)
        assert instance1.param1 == 100

        isolated_loss_registry.register(name="instantiate_loss_kwargs")(
            DummyLoss2
        )
        instance2 = isolated_loss_registry.instantiate(
            "instantiate_loss_kwargs", param_a="custom_val"
        )
        assert isinstance(instance2, DummyLoss2)
        assert instance2.param_a == "custom_val"

        # Test with both
        instance3 = isolated_loss_registry.instantiate(
            "instantiate_loss_args", 200
        )
        assert isinstance(instance3, DummyLoss1)
        assert instance3.param1 == 200

    def test_instantiate_non_existent_loss_raises_key_error(
        self, isolated_loss_registry: Registry[nn.Module]
    ) -> None:
        """Test that instantiating a non-existent loss raises KeyError."""
        with pytest.raises(
            KeyError, match=r"Component 'non_existent_loss_instance' not found"
        ):
            isolated_loss_registry.instantiate("non_existent_loss_instance")

    def test_list_losses(
        self, isolated_loss_registry: Registry[nn.Module]
    ) -> None:
        """Test listing registered losses."""
        isolated_loss_registry.register(name="list_loss_1")(DummyLoss1)
        isolated_loss_registry.register(name="list_loss_2")(DummyLoss2)
        losses = isolated_loss_registry.list_components()
        assert sorted(losses) == ["list_loss_1", "list_loss_2"]

    def test_list_with_tags_multiple(
        self, isolated_loss_registry: Registry[nn.Module]
    ) -> None:
        """Test listing losses with their tags."""

        @isolated_loss_registry.register(name="tagged1", tags=["a", "b"])
        class Tagged1(nn.Module):
            pass

        @isolated_loss_registry.register(name="tagged2", tags=["b", "c"])
        class Tagged2(nn.Module):
            pass

        @isolated_loss_registry.register(name="untagged")
        class Untagged(nn.Module):
            pass

        tags_dict = isolated_loss_registry.list_with_tags()
        assert tags_dict["tagged1"] == ["a", "b"]
        assert tags_dict["tagged2"] == ["b", "c"]
        assert "untagged" not in tags_dict

    def test_filter_by_tag_multiple_matches(
        self, isolated_loss_registry: Registry[nn.Module]
    ) -> None:
        """Test filtering by tags with multiple matches."""

        @isolated_loss_registry.register(name="f_tagged1", tags=["a", "b"])
        class FTagged1(nn.Module):
            pass

        @isolated_loss_registry.register(name="f_tagged2", tags=["b", "c"])
        class FTagged2(nn.Module):
            pass

        @isolated_loss_registry.register(name="f_tagged3", tags=["a"])
        class FTagged3(nn.Module):
            pass

        tag_a_losses = isolated_loss_registry.filter_by_tag("a")
        assert sorted(tag_a_losses) == ["f_tagged1", "f_tagged3"]

        tag_b_losses = isolated_loss_registry.filter_by_tag("b")
        assert sorted(tag_b_losses) == ["f_tagged1", "f_tagged2"]

        tag_c_losses = isolated_loss_registry.filter_by_tag("c")
        assert tag_c_losses == ["f_tagged2"]

    def test_unregister_loss(
        self, isolated_loss_registry: Registry[nn.Module]
    ) -> None:
        """Test unregistering a loss."""
        isolated_loss_registry.register(name="to_unregister")(DummyLoss1)
        assert "to_unregister" in isolated_loss_registry
        isolated_loss_registry.unregister("to_unregister")
        assert "to_unregister" not in isolated_loss_registry
        assert len(isolated_loss_registry) == 0

    def test_unregister_non_existent_loss_raises_key_error(
        self, isolated_loss_registry: Registry[nn.Module]
    ) -> None:
        """Test that unregistering a non-existent loss raises KeyError."""
        with pytest.raises(
            KeyError, match=r"Component 'non_existent_unregister' not found"
        ):
            isolated_loss_registry.unregister("non_existent_unregister")

    def test_len_operator(
        self, isolated_loss_registry: Registry[nn.Module]
    ) -> None:
        """Test the len() operator on the registry."""
        assert len(isolated_loss_registry) == 0
        isolated_loss_registry.register(name="len_test_1")(DummyLoss1)
        assert len(isolated_loss_registry) == 1
        isolated_loss_registry.register(name="len_test_2")(DummyLoss2)
        assert len(isolated_loss_registry) == 2
        isolated_loss_registry.unregister("len_test_1")
        assert len(isolated_loss_registry) == 1

    def test_contains_operator(
        self, isolated_loss_registry: Registry[nn.Module]
    ) -> None:
        """Test the 'in' operator on the registry."""
        isolated_loss_registry.register(name="contains_test")(DummyLoss1)
        assert "contains_test" in isolated_loss_registry
        assert "not_in_registry" not in isolated_loss_registry

    def test_register_class_itself_not_instance(
        self, isolated_loss_registry: Registry[nn.Module]
    ) -> None:
        """Ensures the registry stores the class, not an instance."""
        isolated_loss_registry.register()(DummyLoss1)
        item = isolated_loss_registry.get("DummyLoss1")
        assert item == DummyLoss1
        assert isinstance(item, type)
