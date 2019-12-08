"""
HIERARCHICAL ATTENTION NETWORK

[Yang, Zichao, et al. "Hierarchical attention networks for document classification." NAACL. 2016.]

Attention is slightly different from that used in hier_att_net.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .time_distributed import TimeDistributed

class HierAttNetTwo(nn.Module):
    
    def __init__(self, word_embed_size: int, word_hidden_size: int, sent_hidden_size: int,
                 max_doc_len: int, max_sent_len: int, dropout: Optional[float] = None):
        super().__init__()
        self.word_embed_size = word_embed_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.dropout = dropout

        bidirectional   = True
        batch_first     = True

        ## Word Attention Model
        dir_mult = 2 if bidirectional else 1
        sent_encoder = SequenceEncoder(word_embed_size, word_hidden_size, max_sent_len, batch_first=batch_first)
        self.time_distributed = TimeDistributed(sent_encoder, batch_first=batch_first)

        ## Sentence Attention Model
        self.doc_encoder = SequenceEncoder(
            word_hidden_size * dir_mult,
            sent_hidden_size, max_doc_len,
            batch_first=batch_first
        )
        self.final = nn.Linear(dir_mult * sent_hidden_size, 1)

    def forward(self, batch: torch.Tensor):
        """
        Input in shape: (batch_size, num_sentences, num_word_per_sentence, word_embed_size)
        """
        ## (?, num_sents, num_words_per_sent, word_embed_size)
        out = self.time_distributed(batch)          ## sentence embeddings
        if self.dropout:
            out = F.dropout2d(out, p=self.dropout, training=self.training)
        ## (?, num_sents, word_hidden_size * 2)
        out = self.doc_encoder(out)                 ## document embeddings
        if self.dropout:
            out = F.dropout(out, p=self.dropout, training=self.training)
        ## (?, sent_hidden_size * 2)
        out = self.final(out)
        ## (?, num_classes)
        return torch.sigmoid(out)


class SequenceEncoder(nn.Module):
    """
    Element-wise attention model, used to encode a sequence (e.g. a sentence of words).
    """

    def __init__(self, word_embed_size: int, word_hidden_size: int,
                 max_output_length: int, batch_first: bool = True):
        super().__init__()
        self.word_embed_size = word_embed_size
        self.word_hidden_size = word_hidden_size
        self.max_output_length = max_output_length

        self.batch_first = batch_first ## refers to the shape of the input sequence
        self.bidirectional = True

        self.gru = nn.GRU(
            word_embed_size, word_hidden_size,
            num_layers=1, bias=True,
            batch_first=False,
            bidirectional=self.bidirectional
        )

        ## Attention
        self.att = Attention(
            word_hidden_size,
            bidirectional=True
        )

        self.h_last = None
        self.save_hidden_state = False

    def forward(self, sentence: torch.Tensor):
        ## Input in shape: (batch_size, num_elems_per_sequence, elem_embed_size)
        batch_size = sentence.size(0)

        if self.batch_first:
            sentence = sentence.permute(1, 0, 2)    ## transpose dims 0 and 1
            ## (seq_len, batch_size, embed_size)

        ## supply GRU with previous hidden state??
        if self.save_hidden_state and self.h_last is not None and self.h_last.size(1) == batch_size:
            f_output, h_last = self.gru(sentence, self.h_last)
        else:
            f_output, h_last = self.gru(sentence)

        self.h_last = h_last if self.save_hidden_state else None
        h_last = h_last.view(batch_size, -1)

        att_weights = self.att(h_last, f_output)

        att_applied = att_weights.unsqueeze(2) * f_output.permute(1, 0, 2)
        return torch.sum(att_applied, dim=1)


class Attention(nn.Module):
    """
    From: https://github.com/bentrevett/pytorch-seq2seq/
    """

    def __init__(self, hidden_dim: int, bidirectional: bool = True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        dir_mult = (2 if self.bidirectional else 1)

        self.attn = nn.Linear(hidden_dim * 2 * dir_mult, hidden_dim * dir_mult)
        self.v = nn.Parameter(torch.rand(hidden_dim * dir_mult))

    def forward(self, hidden, encoder_outputs):

        ## NOTE in this case (dec_hid_dim == 2*enc_hid_dim)
        #hidden = [batch size, dec hid dim] = [batch size, hid dim * 2]
        #encoder_outputs = [src sent len, batch size, hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        #repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        #hidden = [batch size, src sent len, dec hid dim]
        #encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 

        #energy = [batch size, src sent len, dec hid dim]

        energy = energy.permute(0, 2, 1)

        #energy = [batch size, dec hid dim, src sent len]

        #v = [dec hid dim]

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        #v = [batch size, 1, dec hid dim]

        attention = torch.bmm(v, energy).squeeze(1)

        #attention= [batch size, src len]

        return F.softmax(attention, dim=1)


if __name__ == '__main__':
    max_doc_len = 13
    max_sent_len = 27
    word_hidden_size = 60
    sent_hidden_size = 70
    embedding_size = 300

    model = HierAttNetTwo(embedding_size, word_hidden_size, sent_hidden_size, max_sent_len, max_doc_len)
