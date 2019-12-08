"""
Mixed neural network architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .ffnn_architectures import FFNN


class EmbeddingsAndFeaturesNN(nn.Module):
    """
    This architecture takes multiple inputs, and aims at acquiring
    knowledge from both Embeddings and Feature-based approaches.
    """

    def __init__(self, nn_embeddings, nn_output_dim, num_features, embeddings_dropout=None):
        """
        @nn_embeddings  a pytorch module that takes embeddings as input
        @nn_output_dim  output dimension of given pytorch module
        @num_features   number of scalar features to be concatenated in
                         the last layer
        """

        super().__init__()
        self.embeddings_dropout = embeddings_dropout

        self.nn_embeddings = nn_embeddings
        self.nn_output_dim = nn_output_dim
        self.num_features = num_features

        ## TODO add attention to features ?
        # self.nn_final = FFNN(
        #     nn_output_dim + num_features,
        #     output_dim=1
        # )
        self.final = nn.Linear(nn_output_dim + num_features, 1, bias=True)

    def forward(self, embeddings: torch.Tensor, features: torch.Tensor):
        assert embeddings.shape[0] == features.shape[0], 'Input must have same batch dimension'

        ## apply dropout on embeddings
        if self.embeddings_dropout is not None:
            embeddings = F.dropout2d(embeddings, p=self.embeddings_dropout)

        x_embs = self.nn_embeddings(embeddings)
        x = torch.cat([x_embs, features], 1)
        
        return torch.sigmoid(self.final(x))
