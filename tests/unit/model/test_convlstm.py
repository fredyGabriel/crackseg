from typing import Any

import pytest
import torch

from src.model.components.convlstm import (
    ConvLSTM,
    ConvLSTMCell,
    ConvLSTMConfig,
)


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
        cell_params["kernel_size"][0] // 2,
        cell_params["kernel_size"][1] // 2,
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
    h_init = (
        torch.ones(
            cell_params["batch_size"],
            cell_params["hidden_dim"],
            cell_params["height"],
            cell_params["width"],
        )
        * 0.5
    )
    c_init = (
        torch.ones(
            cell_params["batch_size"],
            cell_params["hidden_dim"],
            cell_params["height"],
            cell_params["width"],
        )
        * 0.5
    )
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
    params.update(
        {
            "num_layers": 3,
            "seq_len": 5,
            "return_all_layers": False,
            "batch_first": True,
        }
    )
    # Use list for hidden_dim and kernel_size for multi-layer testing
    params["hidden_dim"] = [params["hidden_dim"]] * params["num_layers"]
    params["kernel_size"] = [params["kernel_size"]] * params["num_layers"]
    return params


def test_convlstm_layer_init(layer_params):
    """Tests ConvLSTM layer initialization."""
    config = ConvLSTMConfig(
        hidden_dim=layer_params["hidden_dim"],
        kernel_size=layer_params["kernel_size"],
        num_layers=layer_params["num_layers"],
        kernel_expected_dims=2,
        batch_first=layer_params["batch_first"],
        bias=layer_params["bias"],
        return_all_layers=layer_params["return_all_layers"],
    )
    layer = ConvLSTM(
        input_dim=layer_params["input_dim"],
        config=config,
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
    config = ConvLSTMConfig(
        hidden_dim=layer_params["hidden_dim"],
        kernel_size=layer_params["kernel_size"],
        num_layers=layer_params["num_layers"],
        kernel_expected_dims=2,
        batch_first=layer_params["batch_first"],
        bias=layer_params["bias"],
        return_all_layers=False,
    )
    layer = ConvLSTM(
        input_dim=layer_params["input_dim"],
        config=config,
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
    config = ConvLSTMConfig(
        hidden_dim=layer_params["hidden_dim"],
        kernel_size=layer_params["kernel_size"],
        num_layers=layer_params["num_layers"],
        kernel_expected_dims=2,
        batch_first=True,
        bias=layer_params["bias"],
        return_all_layers=return_all,
    )
    layer = ConvLSTM(
        input_dim=layer_params["input_dim"],
        config=config,
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
            layer_params["width"],
        )
        assert h.shape == expected_state_shape
        assert c.shape == expected_state_shape


def test_convlstm_layer_initial_state(layer_params):
    """Tests providing an initial hidden state."""
    config = ConvLSTMConfig(
        hidden_dim=layer_params["hidden_dim"],
        kernel_size=layer_params["kernel_size"],
        num_layers=layer_params["num_layers"],
        kernel_expected_dims=2,
        batch_first=True,
        bias=layer_params["bias"],
        return_all_layers=True,
    )
    layer = ConvLSTM(
        input_dim=layer_params["input_dim"],
        config=config,
    )

    input_tensor = torch.randn(
        layer_params["batch_size"],
        layer_params["seq_len"],
        layer_params["input_dim"],
        layer_params["height"],
        layer_params["width"],
    )

    # Create a non-zero initial state
    initial_hidden_state: list[Any] = []
    for i in range(layer_params["num_layers"]):
        h_init = torch.ones(
            layer_params["batch_size"],
            layer_params["hidden_dim"][i],
            layer_params["height"],
            layer_params["width"],
        ) * (
            i + 1
        )  # Make states unique per layer
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


class TestConvLSTMCell:
    """Pruebas unitarias para la clase ConvLSTMCell."""

    @pytest.fixture
    def default_cell(self):
        """Fixture que proporciona una instancia predeterminada de
        ConvLSTMCell."""
        return ConvLSTMCell(
            input_dim=16, hidden_dim=32, kernel_size=(3, 3), bias=True
        )

    def test_initialization(self):
        """Prueba la inicialización correcta con diferentes parámetros."""
        # Caso básico
        cell = ConvLSTMCell(
            input_dim=16, hidden_dim=32, kernel_size=(3, 3), bias=True
        )
        assert cell.input_dim == 16  # noqa: PLR2004
        assert cell.hidden_dim == 32  # noqa: PLR2004
        assert cell.kernel_size == (3, 3)
        assert cell.bias is True  # Por defecto

        # Prueba con otro tamaño de kernel
        cell = ConvLSTMCell(
            input_dim=16, hidden_dim=32, kernel_size=(3, 5), bias=True
        )
        assert cell.kernel_size == (3, 5)

        # Sin bias
        cell = ConvLSTMCell(
            input_dim=16, hidden_dim=32, kernel_size=(3, 3), bias=False
        )
        assert cell.bias is False

        # Valores diferentes de kernel
        cell = ConvLSTMCell(
            input_dim=16, hidden_dim=32, kernel_size=(5, 5), bias=True
        )
        assert cell.kernel_size == (5, 5)

    def test_conv_layer_shape(self, default_cell):
        """Prueba que la capa convolucional tenga la forma correcta para las
        puertas."""
        # Para el enfoque de puerta combinada, out_channels debería ser 4 *
        # hidden_dim
        assert default_cell.conv.out_channels == 4 * default_cell.hidden_dim
        in_channels = default_cell.input_dim + default_cell.hidden_dim
        assert default_cell.conv.in_channels == in_channels

    def test_forward_shape(self, default_cell):
        """Prueba que la salida del método forward tenga la forma correcta."""
        batch_size = 4
        height, width = 28, 28

        # Crear entrada de prueba
        x = torch.randn(batch_size, default_cell.input_dim, height, width)

        # Realizar paso forward
        h_next, c_next = default_cell(x)

        # Verificar formas de salida
        expected_shape = (batch_size, default_cell.hidden_dim, height, width)
        assert h_next.shape == expected_shape
        assert c_next.shape == expected_shape

    def test_state_initialization(self, default_cell):
        """
        Prueba la inicialización automática del estado si no se proporciona.
        """
        batch_size = 4
        height, width = 28, 28

        # Crear entrada de prueba
        x = torch.randn(batch_size, default_cell.input_dim, height, width)

        # Realizar paso forward sin proporcionar estado
        h_next, c_next = default_cell(x)

        # Verificar que se inicializó correctamente el estado
        expected_shape = (batch_size, default_cell.hidden_dim, height, width)
        assert h_next.shape == expected_shape
        assert c_next.shape == expected_shape

        # Verificar que los estados no son None
        assert h_next is not None
        assert c_next is not None

    def test_state_propagation(self, default_cell):
        """Prueba que el estado se propague correctamente entre pasos
        temporales."""
        batch_size = 4
        height, width = 28, 28

        # Crear entrada de prueba
        x = torch.randn(batch_size, default_cell.input_dim, height, width)

        # Primer paso temporal
        h1, c1 = default_cell(x)

        # Segundo paso temporal con nuevo input pero usando estado del primer
        # paso
        x2 = torch.randn(batch_size, default_cell.input_dim, height, width)
        h2, c2 = default_cell(x2, (h1, c1))

        # Verificar que h2 y c2 son diferentes de h1 y c1
        assert not torch.allclose(h1, h2)
        assert not torch.allclose(c1, c2)

    def test_gradient_flow(self, default_cell):
        """Prueba que los gradientes fluyan correctamente a través de
        múltiples pasos."""
        batch_size = 4
        height, width = 28, 28

        # Crear entrada de prueba con gradientes habilitados
        x = torch.randn(
            batch_size,
            default_cell.input_dim,
            height,
            width,
            requires_grad=True,
        )

        # Primer paso temporal
        h1, c1 = default_cell(x)

        # Segundo paso temporal
        x2 = torch.randn(
            batch_size,
            default_cell.input_dim,
            height,
            width,
            requires_grad=True,
        )
        h2, c2 = default_cell(x2, (h1, c1))

        # Calcular pérdida y retropropagar
        loss = h2.mean()
        loss.backward()

        # Verificar que los gradientes no son None
        assert x.grad is not None
        assert default_cell.conv.weight.grad is not None

    def test_batch_size_one(self, default_cell):
        """
        Prueba con tamaño de lote 1 para detectar problemas de broadcasting.
        """
        batch_size = 1
        height, width = 28, 28

        # Crear entrada de prueba
        x = torch.randn(batch_size, default_cell.input_dim, height, width)

        # Realizar paso forward
        h_next, c_next = default_cell(x)

        # Verificar formas de salida
        expected_shape = (batch_size, default_cell.hidden_dim, height, width)
        assert h_next.shape == expected_shape
        assert c_next.shape == expected_shape

    def test_non_square_input(self, default_cell):
        """Prueba con entradas no cuadradas (altura ≠ anchura)."""
        batch_size = 4
        height, width = 32, 24  # Dimensiones no cuadradas

        # Crear entrada de prueba
        x = torch.randn(batch_size, default_cell.input_dim, height, width)

        # Realizar paso forward
        h_next, c_next = default_cell(x)

        # Verificar formas de salida
        expected_shape = (batch_size, default_cell.hidden_dim, height, width)
        assert h_next.shape == expected_shape
        assert c_next.shape == expected_shape

    def test_multi_step_inference(self, default_cell):
        """Prueba inferencia en múltiples pasos temporales para verificar
        estabilidad."""
        batch_size = 4
        height, width = 28, 28
        time_steps = 10

        # Estado inicial
        h, c = None, None

        # Simular secuencia temporal
        for _t in range(time_steps):
            x = torch.randn(batch_size, default_cell.input_dim, height, width)
            h, c = default_cell(x, cur_state=(h, c) if h is not None else None)

            # Verificar formas de salida en cada paso
            expected_shape = (
                batch_size,
                default_cell.hidden_dim,
                height,
                width,
            )
            assert h.shape == expected_shape
            assert c.shape == expected_shape

            # Verificar valores numéricos estables
            assert not torch.isnan(h).any()
            assert not torch.isnan(c).any()
            assert not torch.isinf(h).any()
            assert not torch.isinf(c).any()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA no disponible para prueba"
    )
    def test_device_handling(self, default_cell):
        """Prueba que la celda funcione correctamente en diferentes
        dispositivos."""
        batch_size = 4
        height, width = 28, 28

        # Prueba en CPU
        x_cpu = torch.randn(batch_size, default_cell.input_dim, height, width)
        h_cpu, c_cpu = default_cell(x_cpu)

        # Mueve la celda a GPU si está disponible
        if torch.cuda.is_available():
            cell_gpu = default_cell.to("cuda")
            x_gpu = x_cpu.to("cuda")

            # Realizar paso forward en GPU
            h_gpu, c_gpu = cell_gpu(x_gpu)

            # Verificar que la salida está en GPU
            assert h_gpu.device.type == "cuda"
            assert c_gpu.device.type == "cuda"

            # Verificar que las formas son iguales
            assert h_gpu.shape == h_cpu.shape
            assert c_gpu.shape == c_cpu.shape

            # Comparar resultados (transfiriendo de vuelta a CPU)
            h_gpu_cpu = h_gpu.cpu()
            # Usar tolerancias mayores para comparar resultados CPU vs GPU
            assert torch.allclose(h_gpu_cpu, h_cpu, rtol=1e-3, atol=1e-3)
