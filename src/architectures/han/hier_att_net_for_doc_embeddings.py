"""
Same as the HAN in hier_att_net.py but returns the document embeddings instead of a value in [0, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .time_distributed import TimeDistributed
from .sequence_encoder import HANSequenceEncoder

class HierAttNetForDocEmbeddings(nn.Module):
    
    def __init__(self, word_embed_size: int, word_hidden_size: int,
                 sent_hidden_size: int, dropout: Optional[float] = None,
                 batch_first: bool = True):
        super().__init__()
        self.word_embed_size = word_embed_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.dropout = dropout

        ## Word Attention Model
        bidirectional = True
        dir_mult = 2 if bidirectional else 1
        sent_encoder = HANSequenceEncoder(word_embed_size, word_hidden_size, batch_first=batch_first)
        self.time_distributed = TimeDistributed(sent_encoder, batch_first=batch_first)

        ## Sentence Attention Model
        self.doc_encoder = HANSequenceEncoder(
            word_hidden_size * dir_mult,
            sent_hidden_size, batch_first=batch_first
        )

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
        return out
