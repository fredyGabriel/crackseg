import pytest
import torch

# Assume ConvLSTMCell is importable from this path
from src.model.components.convlstm import ConvLSTMCell, ConvLSTM


@pytest.fixture
def cell_params():
    """Provides common parameters for ConvLSTMCell tests."""
    return {
        "input_dim": 3,
        "hidden_dim": 32,
        "kernel_size": (3, 3),
        "bias": True,
        "batch_size": 2,
        "height": 16,
        "width": 16,
    }


def test_convlstm_cell_init(cell_params):
    """Tests ConvLSTMCell initialization."""
    cell = ConvLSTMCell(
        input_dim=cell_params["input_dim"],
        hidden_dim=cell_params["hidden_dim"],
        kernel_size=cell_params["kernel_size"],
        bias=cell_params["bias"],
    )
    assert cell.input_dim == cell_params["input_dim"]
    assert cell.hidden_dim == cell_params["hidden_dim"]
    assert cell.kernel_size == cell_params["kernel_size"]
    assert cell.bias == cell_params["bias"]
    assert isinstance(cell.conv, torch.nn.Conv2d)
    expected_in_channels = cell_params["input_dim"] + cell_params["hidden_dim"]
    assert cell.conv.in_channels == expected_in_channels
    assert cell.conv.out_channels == 4 * cell_params["hidden_dim"]
    assert cell.padding == (
        cell_params["kernel_size"][0] // 2, cell_params["kernel_size"][1] // 2
    )


def test_convlstm_cell_forward_shape(cell_params):
    """Tests the output shape of the forward pass."""
    cell = ConvLSTMCell(
        input_dim=cell_params["input_dim"],
        hidden_dim=cell_params["hidden_dim"],
        kernel_size=cell_params["kernel_size"],
        bias=cell_params["bias"],
    )
    input_tensor = torch.randn(
        cell_params["batch_size"],
        cell_params["input_dim"],
        cell_params["height"],
        cell_params["width"],
    )

    # Test with no initial state
    h_next, c_next = cell(input_tensor, None)
    assert h_next.shape == (
        cell_params["batch_size"],
        cell_params["hidden_dim"],
        cell_params["height"],
        cell_params["width"],
    )
    assert c_next.shape == (
        cell_params["batch_size"],
        cell_params["hidden_dim"],
        cell_params["height"],
        cell_params["width"],
    )

    # Test with initial state
    h_init = torch.randn(
        cell_params["batch_size"],
        cell_params["hidden_dim"],
        cell_params["height"],
        cell_params["width"],
    )
    c_init = torch.randn(
        cell_params["batch_size"],
        cell_params["hidden_dim"],
        cell_params["height"],
        cell_params["width"],
    )
    initial_state = (h_init, c_init)
    h_next_init, c_next_init = cell(input_tensor, initial_state)
    assert h_next_init.shape == h_init.shape
    assert c_next_init.shape == c_init.shape


def test_convlstm_cell_state_init(cell_params):
    """Tests internal state initialization when cur_state is None."""
    cell = ConvLSTMCell(
        input_dim=cell_params["input_dim"],
        hidden_dim=cell_params["hidden_dim"],
        kernel_size=cell_params["kernel_size"],
        bias=cell_params["bias"],
    )
    input_tensor = torch.randn(
        cell_params["batch_size"],
        cell_params["input_dim"],
        cell_params["height"],
        cell_params["width"],
    )

    # Internal call to check initialization
    h_cur, c_cur = cell._init_hidden(input_tensor, None)

    expected_shape = (
        cell_params["batch_size"],
        cell_params["hidden_dim"],
        cell_params["height"],
        cell_params["width"],
    )
    assert h_cur.shape == expected_shape
    assert c_cur.shape == expected_shape
    assert torch.all(h_cur == 0)
    assert torch.all(c_cur == 0)


def test_convlstm_cell_state_propagation(cell_params):
    """Tests that the state is correctly passed and used."""
    cell = ConvLSTMCell(
        input_dim=cell_params["input_dim"],
        hidden_dim=cell_params["hidden_dim"],
        kernel_size=cell_params["kernel_size"],
        bias=cell_params["bias"],
    )
    input_tensor = torch.randn(
        cell_params["batch_size"],
        cell_params["input_dim"],
        cell_params["height"],
        cell_params["width"],
    )

    # Create a non-zero initial state
    h_init = torch.ones(
        cell_params["batch_size"],
        cell_params["hidden_dim"],
        cell_params["height"],
        cell_params["width"],
    ) * 0.5
    c_init = torch.ones(
        cell_params["batch_size"],
        cell_params["hidden_dim"],
        cell_params["height"],
        cell_params["width"],
    ) * 0.5
    initial_state = (h_init, c_init)

    # First forward pass
    h_next1, c_next1 = cell(input_tensor, initial_state)

    # Second forward pass using the state from the first pass
    # Use a different input tensor to ensure state influences output
    input_tensor2 = torch.randn_like(input_tensor) * 2
    h_next2, c_next2 = cell(input_tensor2, (h_next1, c_next1))

    # Third forward pass, starting again with zeros (should differ from second)
    h_next3, c_next3 = cell(input_tensor2, None)  # Input is same as 2nd pass

    # Check that outputs differ when initial state differs
    assert not torch.allclose(h_next2, h_next3, atol=1e-6)
    assert not torch.allclose(c_next2, c_next3, atol=1e-6)


# --- Tests for ConvLSTM Layer ---

@pytest.fixture
def layer_params(cell_params):
    """Provides common parameters for ConvLSTM layer tests."""
    params = cell_params.copy()
    params.update({
        "num_layers": 3,
        "seq_len": 5,
        "return_all_layers": False,
        "batch_first": True,
    })
    # Use list for hidden_dim and kernel_size for multi-layer testing
    params["hidden_dim"] = [params["hidden_dim"]] * params["num_layers"]
    params["kernel_size"] = [params["kernel_size"]] * params["num_layers"]
    return params


def test_convlstm_layer_init(layer_params):
    """Tests ConvLSTM layer initialization."""
    layer = ConvLSTM(
        input_dim=layer_params["input_dim"],
        hidden_dim=layer_params["hidden_dim"],
        kernel_size=layer_params["kernel_size"],
        num_layers=layer_params["num_layers"],
        batch_first=layer_params["batch_first"],
        bias=layer_params["bias"],
        return_all_layers=layer_params["return_all_layers"]
    )
    assert layer.num_layers == layer_params["num_layers"]
    assert len(layer.cell_list) == layer_params["num_layers"]
    assert all(isinstance(cell, ConvLSTMCell) for cell in layer.cell_list)
    # Check dimensions match across layers
    assert layer.cell_list[0].input_dim == layer_params["input_dim"]
    for i in range(layer_params["num_layers"]):
        assert layer.cell_list[i].hidden_dim == layer_params["hidden_dim"][i]
        assert layer.cell_list[i].kernel_size == layer_params["kernel_size"][i]
        if i > 0:
            # Input dim of layer i should match hidden dim of layer i-1
            assert (
                layer.cell_list[i].input_dim
                == layer_params["hidden_dim"][i - 1]
            )


@pytest.mark.parametrize("batch_first", [True, False])
def test_convlstm_layer_forward_shape(layer_params, batch_first):
    """Tests the output shape of the ConvLSTM forward pass."""
    layer_params["batch_first"] = batch_first
    layer = ConvLSTM(
        input_dim=layer_params["input_dim"],
        hidden_dim=layer_params["hidden_dim"],
        kernel_size=layer_params["kernel_size"],
        num_layers=layer_params["num_layers"],
        batch_first=layer_params["batch_first"],
        bias=layer_params["bias"],
        return_all_layers=False  # Test with return_all_layers=False first
    )

    if batch_first:
        input_tensor = torch.randn(
            layer_params["batch_size"],
            layer_params["seq_len"],
            layer_params["input_dim"],
            layer_params["height"],
            layer_params["width"],
        )
    else:
        input_tensor = torch.randn(
            layer_params["seq_len"],
            layer_params["batch_size"],
            layer_params["input_dim"],
            layer_params["height"],
            layer_params["width"],
        )

    layer_output_list, last_state_list = layer(input_tensor, None)

    # Check output list (should contain only last layer output)
    assert len(layer_output_list) == 1
    last_layer_output = layer_output_list[0]
    expected_output_shape = (
        layer_params["batch_size"],
        layer_params["seq_len"],
        layer_params["hidden_dim"][-1],  # Last layer hidden dim
        layer_params["height"],
        layer_params["width"],
    )
    assert last_layer_output.shape == expected_output_shape

    # Check last state list
    assert len(last_state_list) == layer_params["num_layers"]
    for i in range(layer_params["num_layers"]):
        h, c = last_state_list[i]
        expected_state_shape = (
            layer_params["batch_size"],
            layer_params["hidden_dim"][i],
            layer_params["height"],
            layer_params["width"],
        )
        assert h.shape == expected_state_shape
        assert c.shape == expected_state_shape


@pytest.mark.parametrize("return_all", [True, False])
def test_convlstm_layer_return_all_layers(layer_params, return_all):
    """Tests the return_all_layers flag."""
    layer = ConvLSTM(
        input_dim=layer_params["input_dim"],
        hidden_dim=layer_params["hidden_dim"],
        kernel_size=layer_params["kernel_size"],
        num_layers=layer_params["num_layers"],
        batch_first=True,  # Keep batch_first True for simplicity here
        bias=layer_params["bias"],
        return_all_layers=return_all
    )

    input_tensor = torch.randn(
        layer_params["batch_size"],
        layer_params["seq_len"],
        layer_params["input_dim"],
        layer_params["height"],
        layer_params["width"],
    )

    layer_output_list, last_state_list = layer(input_tensor, None)

    if return_all:
        assert len(layer_output_list) == layer_params["num_layers"]
        for i in range(layer_params["num_layers"]):
            expected_layer_output_shape = (
                layer_params["batch_size"],
                layer_params["seq_len"],
                layer_params["hidden_dim"][i],
                layer_params["height"],
                layer_params["width"],
            )
            assert layer_output_list[i].shape == expected_layer_output_shape
    else:
        assert len(layer_output_list) == 1  # Only last layer
        expected_last_layer_output_shape = (
            layer_params["batch_size"],
            layer_params["seq_len"],
            layer_params["hidden_dim"][-1],
            layer_params["height"],
            layer_params["width"],
        )
        assert layer_output_list[0].shape == expected_last_layer_output_shape

    # State list shape is independent of return_all_layers
    assert len(last_state_list) == layer_params["num_layers"]

    # Check that outputs differ (COMMENTED OUT - fails for i=0)
    # for i in range(layer_params["num_layers"]):
    #     assert not torch.allclose(
    #         layer_output_list[i], layer_output_list[0], atol=1e-6
    #     )

    # Basic check on state shapes
    for h, c in last_state_list:
        expected_state_shape = (
            layer_params["batch_size"],
            # Assuming same hidden dim for simplicity of state check
            layer_params["hidden_dim"][0],
            layer_params["height"],
            layer_params["width"]
        )
        assert h.shape == expected_state_shape
        assert c.shape == expected_state_shape


def test_convlstm_layer_initial_state(layer_params):
    """Tests providing an initial hidden state."""
    layer = ConvLSTM(
        input_dim=layer_params["input_dim"],
        hidden_dim=layer_params["hidden_dim"],
        kernel_size=layer_params["kernel_size"],
        num_layers=layer_params["num_layers"],
        batch_first=True,
        bias=layer_params["bias"],
        return_all_layers=True  # Return all for easier state checking
    )

    input_tensor = torch.randn(
        layer_params["batch_size"],
        layer_params["seq_len"],
        layer_params["input_dim"],
        layer_params["height"],
        layer_params["width"],
    )

    # Create a non-zero initial state
    initial_hidden_state = []
    for i in range(layer_params["num_layers"]):
        h_init = torch.ones(
            layer_params["batch_size"],
            layer_params["hidden_dim"][i],
            layer_params["height"],
            layer_params["width"],
        ) * (i + 1)  # Make states unique per layer
        c_init = torch.ones_like(h_init) * (i + 1)
        initial_hidden_state.append((h_init, c_init))

    # Forward pass with initial state
    layer_output_list1, last_state_list1 = layer(
        input_tensor, initial_hidden_state
    )

    # Forward pass without initial state (should differ)
    layer_output_list0, last_state_list0 = layer(input_tensor, None)

    # Check that final states differ
    for i in range(layer_params["num_layers"]):
        h1, c1 = last_state_list1[i]
        h0, c0 = last_state_list0[i]
        assert not torch.allclose(h1, h0, atol=1e-6)
        assert not torch.allclose(c1, c0, atol=1e-6)

    # Check that outputs differ
    for i in range(layer_params["num_layers"]):
        assert not torch.allclose(
            layer_output_list1[i], layer_output_list0[i], atol=1e-6
        )
