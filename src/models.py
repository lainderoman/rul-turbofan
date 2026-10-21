"""PyTorch sequence models for RUL prediction.

Both models take a batch of windows with shape
    (batch size, window length, number of sensors)
and return one RUL value per window, with shape (batch size,).
"""

import torch
from torch import nn


class LSTMModel(nn.Module):
    """A simple LSTM: read the window cycle by cycle, predict RUL from the last step."""

    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        # batch_first=True -> input shape is (batch, time, features)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)          # (batch, time, hidden_size)
        last_step = lstm_out[:, -1, :]       # hidden state at the last cycle
        return self.output(last_step).squeeze(-1)


class TransLSTM(nn.Module):
    """Transformer encoder followed by an LSTM.

    1. A linear layer projects the sensors of each cycle to d_model numbers.
    2. A learned positional embedding tells the model which position
       (cycle 1..window_size) each step has. Self-attention alone does not
       know the order of the steps.
    3. The Transformer encoder lets every cycle "look at" all the other
       cycles of the window (self-attention).
    4. An LSTM then reads the encoded sequence in time order, and its last
       hidden state is mapped to the RUL.
    """

    def __init__(
        self,
        n_features,
        window_size,
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        lstm_hidden_size=64,
        dropout=0.1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        # One learnable vector per position in the window, starting at zero
        self.positional_embedding = nn.Parameter(torch.zeros(1, window_size, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        self.lstm = nn.LSTM(input_size=d_model, hidden_size=lstm_hidden_size, batch_first=True)
        self.output = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        x = self.input_projection(x)         # (batch, time, d_model)
        x = x + self.positional_embedding    # add position information
        x = self.encoder(x)                  # (batch, time, d_model)
        lstm_out, _ = self.lstm(x)           # (batch, time, lstm_hidden_size)
        last_step = lstm_out[:, -1, :]
        return self.output(last_step).squeeze(-1)


def count_parameters(model):
    """Number of trainable parameters (weights) of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
