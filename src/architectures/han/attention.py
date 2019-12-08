import torch
import torch.nn as nn
import torch.nn.functional as F


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
        """
        hidden: sequence of queries to query the context with.
        encoder_outputs: context over which to apply the attention mechanism.
        """
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(2)


class AttnWeightedSeq(nn.Module):
    """
    Wrapper for Attention layer, in order to return an attention weighted sequence.
    """

    def __init__(self, *args):
        super().__init__()
        self.att = Attn(*args)

    def forward(self, hidden, encoder_outputs):
        attn_weights = self.att(hidden, encoder_outputs)
        weighted_sequence = torch.sum(attn_weights * encoder_outputs, 1)
        ## Weighted average of sequence elements, out.shape = (batch_size, hidden_dim)
        return weighted_sequence
