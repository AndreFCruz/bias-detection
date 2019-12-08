import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List


class LSTM(nn.Module):
    """Generic LSTM"""

    def __init__(self, input_dim: int, hidden_dim: int, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Optional arguments for LSTM layer(s)
        self.lstm_kwargs = {
            'num_layers': 1,
            'dropout': 0,
            'bidirectional': True,
            'bias': True
        }
        self.lstm_kwargs = {
            k: (kwargs[k] if k in kwargs else self.lstm_kwargs[k]) for k in self.lstm_kwargs
        }

        # Define layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, **self.lstm_kwargs)

    def forward(self, x):
        h_n, _ = self.lstm(x)
        return h_n[:,-1,:] # Only use last time-step
