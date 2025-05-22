from dataclasses import dataclass

import torch
from torch import nn


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
    ):
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

        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # Compute combined gates
        combined_conv = self.conv(combined)

        # Split combined gates into individual gates
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1
        )

        # Apply activations
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # Compute next cell state
        c_next = f * c_cur + i * g
        # Compute next hidden state
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


class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM.

    Args:
        input_dim (int): Number of channels in input tensor.
        config (ConvLSTMConfig): Configuration object for ConvLSTM parameters.
    """

    def __init__(self, input_dim: int, config: ConvLSTMConfig):
        super().__init__()

        self._check_kernel_size_consistency(
            config.kernel_size, config.kernel_expected_dims
        )

        # Ensure hidden_dim and kernel_size are lists for iteration
        hidden_dim_list = self._extend_for_layers(
            config.hidden_dim, config.num_layers
        )
        kernel_size_list = self._extend_for_layers(
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
        for i in range(self.num_layers):
            cur_input_dim = (
                self.input_dim if i == 0 else self.hidden_dim[i - 1]
            )

            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],  # Use from processed list
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
            # (time, batch, C, H, W) -> (batch, time, C, H, W)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, seq_len, _, h, w = input_tensor.size()

        # Initialize hidden states if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(
                batch_size=b, image_size=(h, w), device=input_tensor.device
            )

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor
        internal_state = []

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                # Input to the current layer is the input tensor for the first
                # layer, or the output of the previous layer for subsequent
                # layers
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c],
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            # Next layer's input is current layer's output
            cur_layer_input = layer_output

            internal_state.append((h, c))
            if self.return_all_layers:
                layer_output_list.append(layer_output)

        if not self.return_all_layers:
            layer_output_list.append(layer_output)  # Only last layer output

        last_state_list = internal_state

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
            # Generate a dummy input tensor for size inference in _init_hidden
            # This avoids passing the actual input tensor down, simplifying API
            dummy_input_size = (batch_size, self.hidden_dim[i]) + image_size
            dummy_input = torch.empty(dummy_input_size, device=device)
            # Use the cell's internal _init_hidden method
            init_states.append(
                self.cell_list[i]._init_hidden(dummy_input, None)
            )
        return init_states

    def _check_kernel_size_consistency(
        self, kernel_size, kernel_expected_dims
    ):
        """Checks if kernel_size format is valid (tuple or list of 2 ints)."""
        # Case 1: Already a tuple of 2 ints
        if (
            isinstance(kernel_size, tuple)
            and len(kernel_size) == kernel_expected_dims
        ):
            if all(isinstance(elem, int) for elem in kernel_size):
                return kernel_size

        # Case 2: List or ListConfig of 2 ints/numbers - convert to tuple
        # This handles both Python lists and OmegaConf ListConfig from YAML
        if (
            hasattr(kernel_size, "__len__")
            and len(kernel_size) == kernel_expected_dims
        ):
            # Try to convert any numeric types to int
            try:
                as_tuple = tuple(int(elem) for elem in kernel_size)
                return as_tuple
            except (ValueError, TypeError):
                # Continue to error if conversion fails
                pass

        # Case 3: List of tuples/lists (for multi-layer consistency)
        is_list_of_valid_pairs = (
            hasattr(kernel_size, "__iter__")
            and hasattr(kernel_size, "__len__")
            and all(
                (
                    hasattr(elem, "__len__")
                    and len(elem) == kernel_expected_dims
                )
                for elem in kernel_size
            )
        )
        if is_list_of_valid_pairs:
            # For a list of kernel sizes (one per layer), validate format
            # Type conversion is handled by _extend_for_layers
            return kernel_size

        # If we get here, format is invalid
        raise ValueError(
            f"kernel_size must be a tuple of {kernel_expected_dims} ints, "
            f"list of {kernel_expected_dims} ints, "
            "or list of tuples/lists for multi-layer ConvLSTM"
        )

    def _extend_for_layers(self, param, num_layers):
        """Extends a parameter like hidden_dim or kernel_size for layers."""
        # Simple version: extend if not a list, otherwise validate length.
        # Consistency check for kernel_size contents happens elsewhere.
        if not isinstance(param, list):
            return [param] * num_layers
        else:
            if len(param) != num_layers:
                raise ValueError(
                    f"Length of list param ({len(param)}) doesn't match "
                    f"num_layers ({num_layers})"
                )
            return param


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
