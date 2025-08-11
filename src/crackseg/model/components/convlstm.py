from dataclasses import dataclass

import torch
from torch import nn

from .utils.convlstm_utils import (
    check_kernel_size_consistency as _check_kernel_helper,
)
from .utils.convlstm_utils import (
    extend_param_for_layers as _extend_for_layers_helper,
)


@dataclass
class ConvLSTMConfig:
    """Configuration for ConvLSTM layer."""

    hidden_dim: int | list[int]
    kernel_size: tuple[int, int] | list[tuple[int, int]]
    num_layers: int
    kernel_expected_dims: int
    batch_first: bool = False
    bias: bool = True
    return_all_layers: bool = False


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell.

    Args:
        input_dim (int): Number of channels in input tensor.
        hidden_dim (int): Number of channels in hidden state.
        kernel_size (tuple[int, int]): Size of the convolutional kernel.
        bias (bool): Whether or not to add the bias.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: tuple[int, int],
        bias: bool,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias

        # Calculate padding to preserve spatial dimensions
        padding_h = kernel_size[0] // 2
        padding_w = kernel_size[1] // 2
        self.padding = (padding_h, padding_w)

        # Convolutional layer for combined gates
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,  # i, f, o, c gates
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.bias_hh = nn.Parameter(torch.zeros(4 * hidden_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / (self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

    def forward(
        self,
        input_tensor: torch.Tensor,
        cur_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ConvLSTM cell.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape
                (batch_size, input_dim, height, width).
            cur_state (tuple[torch.Tensor, torch.Tensor] | None):
                Tuple containing the previous hidden state (h_cur)
                and cell state (c_cur). Each has shape
                (batch_size, hidden_dim, height, width).
                If None, initializes states to zeros.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing the next hidden
                state (h_next) and cell state (c_next).
        """
        h_cur, c_cur = self._init_hidden(input_tensor, cur_state)

        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1
        )

        i = torch.sigmoid(cc_i + self.bias_hh[: self.hidden_dim])
        f = torch.sigmoid(
            cc_f + self.bias_hh[self.hidden_dim : 2 * self.hidden_dim]
        )
        o = torch.sigmoid(
            cc_o + self.bias_hh[2 * self.hidden_dim : 3 * self.hidden_dim]
        )
        g = torch.tanh(cc_g + self.bias_hh[3 * self.hidden_dim :])

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def _init_hidden(
        self,
        input_tensor: torch.Tensor,
        cur_state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initializes hidden state if not provided."""
        batch_size, _, height, width = input_tensor.size()
        device = input_tensor.device

        if cur_state is None:
            # Create zero tensors for initial hidden and cell states
            h_cur = torch.zeros(
                batch_size, self.hidden_dim, height, width, device=device
            )
            c_cur = torch.zeros(
                batch_size, self.hidden_dim, height, width, device=device
            )
        else:
            h_cur, c_cur = cur_state

        return h_cur, c_cur

    def init_hidden_zeros(
        self,
        batch_size: int,
        image_size: tuple[int, int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create zero-initialized hidden state."""
        height, width = image_size
        h_cur = torch.zeros(
            batch_size, self.hidden_dim, height, width, device=device
        )
        c_cur = torch.zeros(
            batch_size, self.hidden_dim, height, width, device=device
        )
        return h_cur, c_cur


class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM.

    Args:
        input_dim (int): Number of channels in input tensor.
        config (ConvLSTMConfig): Configuration object for ConvLSTM parameters.
    """

    def __init__(self, input_dim: int, config: ConvLSTMConfig) -> None:
        super().__init__()

        _check_kernel_helper(config.kernel_size, config.kernel_expected_dims)

        # Ensure hidden_dim and kernel_size are lists for iteration
        hidden_dim_list = _extend_for_layers_helper(
            config.hidden_dim, config.num_layers
        )
        kernel_size_list = _extend_for_layers_helper(
            config.kernel_size, config.num_layers
        )

        if (
            not len(hidden_dim_list)
            == len(kernel_size_list)
            == config.num_layers
        ):
            raise ValueError("Inconsistent list length for dims and kernels.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim_list  # Store the list version
        self.kernel_size = kernel_size_list  # Store the list version
        self.num_layers = config.num_layers
        self.batch_first = config.batch_first
        self.bias = config.bias
        self.return_all_layers = config.return_all_layers
        self.kernel_expected_dims = config.kernel_expected_dims

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = (
                self.input_dim if i == 0 else self.hidden_dim[i - 1]
            )

            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(
        self,
        input_tensor: torch.Tensor,
        hidden_state: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for the ConvLSTM network.

        Args:
            input_tensor (torch.Tensor): Input sequence. Shape depends on
                `batch_first`.
                If `batch_first=True`: (batch, time, channel, height, width)
                If `batch_first=False`: (time, batch, channel, height, width)
            hidden_state (list[tuple[torch.Tensor, torch.Tensor]] | None):
                List of tuples (h, c) for each layer's initial state.
                If None, initializes states to zeros.

        Returns:
            tuple[list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
                - layer_output_list: List of output tensors for each time step.
                    Shape depends on `return_all_layers`.
                - last_state_list: List of tuples (h, c) for the final state
                    of each layer.
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward, can send image size here
            hidden_state = self._init_hidden(
                batch_size=b, image_size=(h, w), device=input_tensor.device
            )

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c],
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(
        self,
        batch_size: int,
        image_size: tuple[int, int],
        device: torch.device,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Initializes hidden states for all layers."""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.cell_list[i].init_hidden_zeros(  # type: ignore[attr-defined]
                    batch_size, image_size, device
                )
            )
        return init_states

    # Helper logic moved to utils: _check_kernel_size_consistency, _extend_for_layers


# Example usage (for illustration, will be tested properly)
if __name__ == "__main__":
    # Example parameters
    _batch_size, _height, _width = 2, 16, 16
    _input_dim, _hidden_dim = 3, 32
    _kernel_size = (3, 3)
    _num_layers = 1
    _kernel_expected_dims_val = 2
    _bias_val = True

    # Create cell
    _convlstm_cell = ConvLSTMCell(
        _input_dim, _hidden_dim, _kernel_size, _bias_val
    )

    # Create dummy input
    _input_tensor = torch.randn(_batch_size, _input_dim, _height, _width)
    # Create config for ConvLSTM
    _convlstm_config = ConvLSTMConfig(
        hidden_dim=_hidden_dim,
        kernel_size=_kernel_size,
        num_layers=_num_layers,
        kernel_expected_dims=_kernel_expected_dims_val,
        bias=_bias_val,
    )
    _convlstm_layer = ConvLSTM(input_dim=_input_dim, config=_convlstm_config)

    # Initial state (optional) for cell
    _h_init_cell = torch.randn(_batch_size, _hidden_dim, _height, _width)
    _c_init_cell = torch.randn(_batch_size, _hidden_dim, _height, _width)
    _initial_state_cell = (_h_init_cell, _c_init_cell)

    # Forward pass cell with initial state
    _h_next_cell, _c_next_cell = _convlstm_cell(
        _input_tensor, _initial_state_cell
    )
    print("Cell Output shapes (with initial state):")
    print(f"h_next_cell: {_h_next_cell.shape}")
    print(f"c_next_cell: {_c_next_cell.shape}")

    # Forward pass cell without initial state (will initialize to zeros)
    _h_next_zeros_cell, _c_next_zeros_cell = _convlstm_cell(
        _input_tensor, None
    )
    print("\nCell Output shapes (without initial state):")
    print(f"h_next_cell_zeros: {_h_next_zeros_cell.shape}")
    print(f"c_next_cell_zeros: {_c_next_zeros_cell.shape}")

    # Example for ConvLSTM layer
    _seq_len = 5
    _layer_input_tensor = torch.randn(
        _batch_size, _seq_len, _input_dim, _height, _width
    )

    # Initial state for layer (optional)
    _initial_layer_state = []
    for _ in range(_num_layers):
        _h_init_layer = torch.randn(_batch_size, _hidden_dim, _height, _width)
        _c_init_layer = torch.randn(_batch_size, _hidden_dim, _height, _width)
        _initial_layer_state.append((_h_init_layer, _c_init_layer))

    _layer_output_list, _last_state_list = _convlstm_layer(
        _layer_input_tensor, _initial_layer_state
    )
    print("\nLayer Output shapes (with initial state):")
    if _convlstm_layer.return_all_layers:
        for i, _out in enumerate(_layer_output_list):
            print(f"Layer {i} output: {_out.shape}")
    else:
        print(f"Last layer output: {_layer_output_list[0].shape}")

    for i, (_h, _c) in enumerate(_last_state_list):
        print(f"Layer {i} final h: {_h.shape}, final c: {_c.shape}")

    _layer_output_list_no_hs, _last_state_list_no_hs = _convlstm_layer(
        _layer_input_tensor, None
    )
    print("\nLayer Output shapes (without initial state):")
    if _convlstm_layer.return_all_layers:
        for i, _out in enumerate(_layer_output_list_no_hs):
            print(f"Layer {i} output: {_out.shape}")
    else:
        print(f"Last layer output: {_layer_output_list_no_hs[0].shape}")

    for i, (_h, _c) in enumerate(_last_state_list_no_hs):
        print(f"Layer {i} final h: {_h.shape}, final c: {_c.shape}")
