"""
Dsitributed a given layer/model along the time axis of a given input.
Inspired by Keras' TimeDistributed layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeDistributed(nn.Module):
    """
    Should take (batch_size, time_steps, feature1, ...), and apply its inner module
    to each of the time_steps.
    Inner module takes as input: (batch_size, feature1, ...).
    Mimicks Keras' TimeDistributed layer wrapper.
    """

    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        num_dims = len(x.size())
        if num_dims <= 2:
            return self.module(x)

        # Dimensions corresponding to input features (excludes batch_size and time_steps)
        x_feature_dimensions = [-i for i in range(num_dims - 2, 0, -1)]

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, *[x.size(i) for i in x_feature_dimensions])  # (samples * timesteps, input_size...)

        y = self.module(x_reshape)
        y_feature_dimensions = [-i for i in range(len(y.size()) - 1, 0, -1)]
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), x.size(1), *[y.size(i) for i in y_feature_dimensions])  # (samples, timesteps, output_size)
        else:
            ## NOTE NEVER CHECKED THIS CONDITION AFTER CHANGING CODE
            y = y.view(x.size(0), x.size(1), *[y.size(i) for i in y_feature_dimensions])  # (timesteps, samples, output_size)

        return y
