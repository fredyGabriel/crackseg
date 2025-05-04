import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell.

    Args:
        input_dim (int): Number of channels in input tensor.
        hidden_dim (int): Number of channels in hidden state.
        kernel_size (tuple[int, int]): Size of the convolutional kernel.
        bias (bool): Whether or not to add the bias.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 kernel_size: tuple[int, int],
                 bias: bool):
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
            bias=self.bias
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        cur_state: tuple[torch.Tensor, torch.Tensor] | None = None
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
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim,
                                             dim=1)

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
        cur_state: tuple[torch.Tensor, torch.Tensor] | None
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
        hidden_dim (int | list[int]): Number of channels in hidden state.
            Can be a single int (all layers) or a list.
        kernel_size (tuple[int, int] | list[tuple[int, int]]):
            Size of the convolutional kernel.
            Can be a single tuple (all layers) or a list.
        num_layers (int): Number of ConvLSTM layers.
        batch_first (bool): If True, inputs are
            (batch, time, channel, height, width).
        bias (bool): Whether or not to add the bias in cells.
        return_all_layers (bool): If true, return outputs and states
            for all layers.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int | list[int],
                 kernel_size: tuple[int, int] | list[tuple[int, int]],
                 num_layers: int,
                 batch_first: bool = False,
                 bias: bool = True,
                 return_all_layers: bool = False):
        super().__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Ensure hidden_dim and kernel_size are lists for iteration
        hidden_dim = self._extend_for_layers(hidden_dim, num_layers)
        kernel_size = self._extend_for_layers(kernel_size, num_layers)

        if not len(hidden_dim) == len(kernel_size) == num_layers:
            raise ValueError('Inconsistent list length for dims and kernels.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else\
                self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(
        self,
        input_tensor: torch.Tensor,
        hidden_state: list[tuple[torch.Tensor, torch.Tensor]] | None = None
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
                    cur_state=[h, c]
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
        self, batch_size: int, image_size: tuple[int, int],
        device: torch.device
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Initializes hidden states for all layers."""
        init_states = []
        for i in range(self.num_layers):
            # Generate a dummy input tensor for size inference in _init_hidden
            # This avoids passing the actual input tensor down, simplifying API
            dummy_input_size = (batch_size, self.hidden_dim[i]) + image_size
            dummy_input = torch.empty(dummy_input_size, device=device)
            # Use the cell's internal _init_hidden method
            init_states.append(self.cell_list[i]._init_hidden(dummy_input,
                                                              None))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """Checks if kernel_size format is valid (tuple or list of 2 ints)."""
        # Case 1: Already a tuple of 2 ints
        if isinstance(kernel_size, tuple) and len(kernel_size) == 2:
            if all(isinstance(elem, int) for elem in kernel_size):
                return kernel_size

        # Case 2: List or ListConfig of 2 ints/numbers - convert to tuple
        # This handles both Python lists and OmegaConf ListConfig from YAML
        if hasattr(kernel_size, '__len__') and len(kernel_size) == 2:
            # Try to convert any numeric types to int
            try:
                as_tuple = tuple(int(elem) for elem in kernel_size)
                return as_tuple
            except (ValueError, TypeError):
                # Continue to error if conversion fails
                pass

        # Case 3: List of tuples/lists (for multi-layer consistency)
        is_list_of_valid_pairs = (
            hasattr(kernel_size, '__iter__') and
            hasattr(kernel_size, '__len__') and
            all(
                (hasattr(elem, '__len__') and len(elem) == 2)
                for elem in kernel_size
            )
        )
        if is_list_of_valid_pairs:
            # For a list of kernel sizes (one per layer), validate format
            # Type conversion is handled by _extend_for_layers
            return kernel_size

        # If we get here, format is invalid
        raise ValueError(
            'kernel_size must be a tuple of 2 ints, list of 2 ints, '
            'or list of tuples/lists for multi-layer ConvLSTM'
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
if __name__ == '__main__':
    # Example parameters
    _batch_size, _height, _width = 2, 16, 16
    _input_dim, _hidden_dim = 3, 32
    _kernel_size = (3, 3)
    _bias = True

    # Create cell
    _convlstm_cell = ConvLSTMCell(_input_dim, _hidden_dim, _kernel_size, _bias)

    # Create dummy input
    _input_tensor = torch.randn(_batch_size, _input_dim, _height, _width)

    # Initial state (optional)
    _h_init = torch.randn(_batch_size, _hidden_dim, _height, _width)
    _c_init = torch.randn(_batch_size, _hidden_dim, _height, _width)
    _initial_state = (_h_init, _c_init)

    # Forward pass with initial state
    _h_next, _c_next = _convlstm_cell(_input_tensor, _initial_state)
    print("Output shapes (with initial state):")
    print(f"h_next: {_h_next.shape}")
    print(f"c_next: {_c_next.shape}")

    # Forward pass without initial state (will initialize to zeros)
    _h_next_zeros, _c_next_zeros = _convlstm_cell(_input_tensor, None)
    print("\nOutput shapes (without initial state):")
    print(f"h_next: {_h_next_zeros.shape}")
    print(f"c_next: {_c_next_zeros.shape}")
