"""
[Singhania, Sneha, et al. "3HAN: A deep neural network for fake news detection." 2017.]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .hier_att_net import HierAttNet
from .sequence_encoder import HANSequenceEncoder
from .time_distributed import TimeDistributed


class ThreeHAN(nn.Module):
    """
    word_embed_size == word_hidden_size == sent_hidden_size
    """

    def __init__(self, word_embed_size: int, last_hidden_size: int,
                 dropout: Optional[float] = None):

        super().__init__()
        self.word_embed_size = word_embed_size
        self.hidden_size = word_embed_size // 2
        self.last_hidden_size = last_hidden_size
        self.dropout = dropout
        self.bidirectional = True

        batch_first = True

        ## Word Attention Model
        sent_encoder = HANSequenceEncoder(word_embed_size, self.hidden_size, batch_first=batch_first)
        self.time_distributed = TimeDistributed(sent_encoder, batch_first=batch_first)

        ## Sentence Attention Model
        self.doc_encoder = HANSequenceEncoder(
            word_embed_size, ## == hidden_size * 2
            self.hidden_size, batch_first=batch_first
        )

        ## Last/Third Attention Model
        self.third_encoder = HANSequenceEncoder(
            word_embed_size,
            last_hidden_size, batch_first=batch_first
        )

        self.final = nn.Linear(2 * last_hidden_size, 1)

    def forward(self, batch: torch.Tensor):
        """
        Input in shape: (batch_size, num_sentences, num_word_per_sentence, word_embed_size)
        First sentence is the heading, to be extracted to the last layer.
        """
        heading, text = batch[:, 0, :, :], batch[:, 1:, :, :]

        ## (?, num_sents, num_words_per_sent, word_embed_size)
        out = self.time_distributed(text)           ## sentence embeddings
        if self.dropout:
            out = F.dropout2d(out, p=self.dropout, training=self.training)
        ## (?, num_sents, word_hidden_size * 2)
        out = self.doc_encoder(out)                 ## document embeddings
        if self.dropout:
            out = F.dropout(out, p=self.dropout, training=self.training)
        ## (?, sent_hidden_size * 2)

        heading_and_doc_embs = torch.cat([heading, out.unsqueeze(1)], dim=1)
        out = self.third_encoder(heading_and_doc_embs)
        if self.dropout:
            out = F.dropout(out, p=self.dropout, training=self.training)
        ## (?, doc_hidden_size * 2)

        out = self.final(out)
        ## (?, num_classes)
        return torch.sigmoid(out)


if __name__ == '__main__':
    num_sents = 13
    num_words_per_sent = 27
    last_hidden_size = 50
    embedding_size = 300
    
    model = ThreeHAN(embedding_size, last_hidden_size)
