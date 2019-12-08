"""
Several utilitary functions and modules for building pytorch NNs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchnlp.nn import Attention
from typing import Callable, Any


class TimeDistributedLayer(nn.Module):
    def __init__(self, function: Callable[[Any], Any]):
        super().__init__()
        self.function = function

    def forward(self, X):
        """X.shape = (batch-size, time-steps, dim1, ...)"""
        import ipdb; ipdb.set_trace() ## NOTE THIS IS UNTESTED
        time_steps = list()

        for i in range(X.shape[1]): # For each time-step
            ## step.shape = (batch-size, dim1, ...)
            step = X[:, i, ...]
            time_steps.append(self.function(step).unsqueeze(1))

        return torch.cat(time_steps, dim=1)
        ## return.shape = (batch-size, time-steps, function-output-shape...)
