import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from .time_distributed import TimeDistributed

class SequenceEncoder(nn.Module):
    """
    Element-wise attention model, used to encode a sequence (e.g. a sentence of words).
    """

    def __init__(self, input_dim: int, hidden_dim: int, rnn_constructor: Callable, bidirectional=True, **rnn_kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.rnn = rnn_constructor(
            input_dim, hidden_dim,
            bidirectional=self.bidirectional,
            **rnn_kwargs
        )

        from .attention import AttnWeightedSeq
        self.att = AttnWeightedSeq('general', hidden_dim * (2 if self.bidirectional else 1))

    def forward(self, sentence: torch.Tensor):
        ## Input in shape: (batch_size, num_elems_per_sequence, elem_embed_size)
        f_output, _ = self.rnn(sentence)

        return self.att(f_output, f_output)
        ## Return in shape: (batch_size, hidden_size)


class LSTMSequenceEncoder(SequenceEncoder):
    """Extracts sequence embedding from attention-weighted time-step embeddings"""

    def __init__(self, input_dim: int, hidden_dim: int, batch_first: bool = True, **lstm_kwargs):
        super().__init__(
            input_dim, hidden_dim,
            rnn_constructor=torch.nn.LSTM,
            batch_first=batch_first,
            **lstm_kwargs,
        )
        ## And that's it :)


class HANSequenceEncoder(SequenceEncoder):
    """
    Sequence encoder used in HAN.
    """

    def __init__(self, word_embed_size: int, word_hidden_size: int, batch_first: bool = True):
        super().__init__(
            word_embed_size, word_hidden_size,
            rnn_constructor=torch.nn.GRU,
            bidirectional=True,
            batch_first=batch_first
        )
        ## And that's it :)
