"""
Neural network architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Callable, Optional

class FFNN(nn.Module):
    """Generic Feed Forward Neural Network"""

    def __init__(self, input_dim: int,
                 hidden_layers: List[int] = list(),
                 output_dim: int = 1,
                 dropout: Optional[int] = None,
                 activ_function: Callable[[torch.Tensor], torch.Tensor] = F.relu,
                 use_batch_norm=True):

        super().__init__()
        assert input_dim > 0 and output_dim > 0

        self.activ_function = activ_function
        self.output_activ_function = torch.sigmoid if output_dim == 1 else F.softmax

        self.linear_layers, self.bn_layers = FFNN.make_linear_layers(
            input_dim, hidden_layers, output_dim, use_batch_norm=use_batch_norm
        )

        if dropout is None:         ## No dropout
            self.dropout_layers = None
        else:                       ## Same dropout on all layers
            self.dropout_layers = nn.ModuleList(
                [nn.Dropout(p=dropout) for _ in range(len(hidden_layers))]
            )

        ## Create ParameterList to make it possible for parameters to be found by optimizers
        ## (needed because optimizer can't find layers in a list)
        # self._params = nn.ParameterList([param for l in self.linear_layers for param in l.parameters()])

        assert (len(self.linear_layers) - 1) == len(hidden_layers)
        assert self.bn_layers is None or len(self.bn_layers) == len(hidden_layers)
        assert self.dropout_layers is None or len(self.dropout_layers) == len(hidden_layers)

    @staticmethod
    def make_linear_layers(input_size, hidden_layers, output_size, use_batch_norm=False) -> Tuple[nn.ModuleList, Optional[nn.ModuleList]]:
        if len(hidden_layers) == 0: return [], []

        linear_layers = list()
        bn_layers = list()
        for i in range(len(hidden_layers)):
            inp = input_size if i == 0 else hidden_layers[i-1]
            outp = hidden_layers[i]

            if use_batch_norm:
                linear_layers.append(nn.Linear(inp, outp, bias=False))
                bn_layers.append(nn.BatchNorm1d(outp))
            else:
                linear_layers.append(nn.Linear(inp, outp, bias=True))

        ## Output layer
        linear_layers.append(nn.Linear(hidden_layers[-1], output_size, bias=True))

        return \
            nn.ModuleList(linear_layers), \
            nn.ModuleList(bn_layers) if use_batch_norm else None

    def forward(self, x):
        for idx, lin_layer in enumerate(self.linear_layers):
            x = lin_layer(x)
            if idx == len(self.linear_layers) - 1:
                continue

            if self.bn_layers is not None and idx < len(self.bn_layers):
                x = self.bn_layers[idx](x)

            x = self.activ_function(x)
            if self.dropout_layers is not None and idx < len(self.dropout_layers):
                x = self.dropout_layers[idx](x)

        return self.output_activ_function(x)

    def predict(self, X):
        return self(X)


class Attn(torch.nn.Module):
    """
    Module encapsulating an attention layer.
    [Luong et al. "Effective Approaches to Attention-based Neural Machine Translation"]
    """

    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, 'is not an appropriate attention method.')
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
