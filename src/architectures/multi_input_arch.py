"""
NN architectures that take multiple inputs.
Will be supplied with document-wide ebeddings + featurized data of same document.
    How to exploit this?
    Take featurized data as input to query in attention layers?
    Take previous sentences as input to query of current sentence?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchnlp.nn import Attention

class MultiInputHAN(nn.Module):
    """
    - Hierarchical attention network
    - Multiple input
    - Experimenting with different inputs to attention's query tensor
    """

    def __init__(self, word_embed_size: int, word_hidden_size: int, sent_hidden_size: int,
                 bidirectional: bool = True):
        super().__init__()
        self.word_embed_size = word_embed_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size

        self.bidirectional = bidirectional
        self.batch_first = True     ## This is stuck in True, needs modifications before being able to change

        ## Word-level Attention
        self.word_gru = nn.GRU(
            word_embed_size, word_hidden_size,
            num_layers=1, bidirectional=self.bidirectional,
            bias=True, batch_first=self.batch_first
        )
        word_dim = word_hidden_size * (2 if self.bidirectional else 1)
        self.word_att = Attention(word_dim)

        ## Sentence-level Attention
        self.sent_gru = nn.GRU(
            word_dim, sent_hidden_size,
            num_layers=1, bidirectional=self.bidirectional,
            bias=True, batch_first=self.batch_first
        )
        sent_dim = sent_hidden_size * (2 if self.bidirectional else 1)
        self.sent_att = Attention(sent_dim)

    def forward(self, embs: torch.Tensor, featurized: torch.Tensor = None):
        """
        Input:
            embs.shape = (batch-size, num-sentences, num-words, embeddings-dim)
            featurized.shape = (num-features,)
        Output:
            output.shape = (sentence-hidden-dim,)
        """
        # titles = embs[:, 0, ...]
        # import ipdb; ipdb.set_trace()

        ## Extract sentence-level embeddings
        attended_sentences = []
        for i in range(embs.size(1)):      # For each sentence
            sentence = embs[:, i, ...]
            rnn_out, _ = self.word_gru(sentence)
            rnn_out = rnn_out.contiguous()
            att_out, _att_w = self.word_att(rnn_out, rnn_out)   ## TODO USE PREVIOUS RNN OUTPUT AS WELL ??? Use title?
            attended_sentences.append(
                torch.sum(att_out, dim=1).unsqueeze(1)
            )
        
        out = torch.cat(attended_sentences, dim=1)

        ## Extract document-level embedding
        rnn_out, _ = self.sent_gru(out)
        rnn_out = rnn_out.contiguous()
        att_out, _att_w = self.sent_att(rnn_out, rnn_out)      ## TODO use other inputs as query ?

        out = torch.sum(att_out, dim=1)
        return out

    def get_output_dim(self):
        return self.sent_hidden_size * (2 if self.bidirectional else 1)


if __name__ == '__main__':
    model = MultiInputHAN(300, 50, 50)

