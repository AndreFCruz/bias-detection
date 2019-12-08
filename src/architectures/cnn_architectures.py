"""
Neural network architectures for Convolutional Neural Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from deprecated import deprecated


## NOTE
## Using a 2D Convolution with num_filters==1 and kernel_size==(H, size_of_second_dimention)
## is the same as a 1D Convolution with kernel_size (H,), with the second dimention encoded
## as filters for each time-step (num_filters==size_of_second_dimention)
@deprecated
class CNN(nn.Module):
    """Generic CNN"""

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int]=[3, 4, 5]):
        super().__init__()
        assert out_channels > 0 and in_channels > 0
        ## in_channels -> usually dimention of embeddings

        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        self.convs = nn.ModuleList([nn.Conv2d(
            1,                      # Input channels
            out_channels,
            (K, in_channels)        # Kernel size (height, width)
        ) for K in self.kernel_sizes])

    def forward(self, x):
        x = x.unsqueeze(1)                                          # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]     # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(el, el.size(2)).squeeze(2) for el in x]   # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)                  # concatenate tensors => (N, len(Ks)*Co)
        return x

    def output_dim(self):
        return self.out_channels * len(self.convs)


class TextCNN(nn.Module):
    """
    CNN that takes embedded inputs in shape: (seq_length, embedding_dim)
    In which embedding_dim is the number of filters at each time-step.

    Based on:
    https://github.com/GateNLP/semeval2019-hyperpartisan-bertha-von-suttner/blob/master/CNN_elmo.py
    -> originally embeddings were average-pooled over each sentence,
       making each sentence a time-step.

    Similar to: [Kim, 2014; Yin and Schutze, 2016; Conneau et al., 2016]
    """

    def __init__(self, embed_dim: int, num_filters: int,
                 filter_sizes: List[int] = [2, 3, 4, 5, 6],
                 use_batch_norm=True):

        super().__init__()
        self.embed_dim = embed_dim      ## Number of input channels
        self.num_filters = num_filters  ## Number of output channels
        self.filter_sizes = filter_sizes
        self.use_batch_norm = use_batch_norm

        self.convs = nn.ModuleList([
            nn.Conv1d(self.embed_dim, self.num_filters, K, bias=(False if use_batch_norm else True)) \
                for K in self.filter_sizes
        ])
        self.convs_bn = nn.ModuleList([
            nn.BatchNorm1d(self.num_filters, momentum=0.7) for conv in self.convs
        ]) if self.use_batch_norm else None

    def forward(self, x):
        ## x.shape == (batch_size, seq_len, embed_dim)

        x = [F.relu(conv(x.permute(0, 2, 1))) for conv in self.convs]
        if self.use_batch_norm:
            x = [bn(elem) for bn, elem in zip(self.convs_bn, x)]
        x = [F.max_pool1d(elem, elem.size(2)).squeeze(2) for elem in x]
        return torch.cat(x, 1)

    def output_dim(self):
        return self.num_filters * len(self.convs)
