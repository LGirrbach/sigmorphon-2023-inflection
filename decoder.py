import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMDecoder(nn.Module):
    """
    Implements a wrapper around pytorch's LSTM for easier sequence processing.
    Note: This implementation uses trainable initialisations of hidden states, if they are not provided.
    Note: This implementation projects the combined hidden states of the forward and backward LSTMs to the common
          hidden dimension
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super(LSTMDecoder, self).__init__()

        # Save arguments
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Initialise modules
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=(dropout if num_layers > 1 else 0.0),
        )

        # Initialise trainable hidden state initialisations
        self.h_0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_size))
        self.c_0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_size))

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor, hidden_state=None):
        batch_size = len(lengths)

        # Pack sequence
        lengths = torch.clamp(
            lengths, 1
        )  # Enforce all lengths are >= 1 (required by pytorch)
        inputs = pack_padded_sequence(
            inputs, lengths, batch_first=True, enforce_sorted=False
        )

        # Prepare hidden states
        if hidden_state is None:
            h_0 = self.h_0.tile((1, batch_size, 1))
            c_0 = self.c_0.tile((1, batch_size, 1))
        else:
            h_0, c_0 = hidden_state

        # Apply LSTM
        encoded, new_hidden_state = self.lstm(inputs, (h_0, c_0))
        encoded, _ = pad_packed_sequence(encoded, batch_first=True)

        return {
            "encoded": encoded,
            "new_hidden_state": new_hidden_state,
            "old_hidden_state": hidden_state,
        }
