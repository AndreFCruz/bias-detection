"""
HIERARCHICAL ATTENTION NETWORK

[Yang, Zichao, et al. "Hierarchical attention networks for document classification." NAACL. 2016.]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from .time_distributed import TimeDistributed
from .sequence_encoder import HANSequenceEncoder

class HierAttNet(nn.Module):
    
    def __init__(self, word_embed_size: int, word_hidden_size: int,
                 sent_hidden_size: int, num_classes:int = 1,
                 dropout: Optional[float] = None,
                 batch_first: bool = True,
                 pack_sequences: bool = False):
        super().__init__()
        self.word_embed_size = word_embed_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.pack_sequences = pack_sequences
        if self.pack_sequences:
            print('** TODO: pack sequences in HAN')

        ## Use sigmoid if only one output class, sigmoid otherwise
        self.output_activation = (torch.sigmoid if num_classes == 1 else F.softmax)
        
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
        self.final = nn.Linear(dir_mult * sent_hidden_size, num_classes)

    ## TODO This could work with variable-sized batches (pack sequences by max seq_len in batch)
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
        return self.output_activation(out)


if __name__ == '__main__':
    num_sents = 13
    num_words_per_sent = 27
    word_hidden_size = 60
    sent_hidden_size = 50
    embedding_size = 300
    
    model = HierAttNet(embedding_size, word_hidden_size, sent_hidden_size, batch_first=True)
