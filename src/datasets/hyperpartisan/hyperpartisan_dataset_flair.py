"""
Wrapper for the Hyperpartisan News dataset, which provides contextual embedding representations.
Embeddings are lazily fetched (evaluated when instance is retrieved).
"""

import torch
import spacy
from typing import Optional, Sequence, Union
from torch.utils.data import Dataset

import flair
from flair.data import Sentence
from flair.embeddings import \
    FlairEmbeddings, WordEmbeddings, \
    BertEmbeddings, ELMoEmbeddings, \
    DocumentPoolEmbeddings, DocumentLSTMEmbeddings
from .hyperpartisan_dataset import HyperpartisanDataset
from .corpus_reader import NewsArticle
from ..flair_dataset import FlairDataset

class HyperpartisanDatasetFlair(HyperpartisanDataset, FlairDataset):
    """
    Hyperpartisan News Dataset using flair-based embeddings.
    """

    def __init__(self, articles: Sequence[NewsArticle],
                 max_seq_len: int = 200,
                 granularity: Union[str, Sequence[str]] = 'token',
                 use_title: bool = True,
                 max_sent_len: int = 100,
                 embeddings: Sequence[str] = ['word'],
                 avg_layers: Optional[int] = None,
                 use_cuda: bool = False):
        super().__init__(
            articles=articles, max_seq_len=max_seq_len, granularity=granularity,
            max_sent_len=max_sent_len, use_title=use_title,     ## HyperpartisanDataset.__init__ args
            embeddings=embeddings, use_cuda=use_cuda,           ## FlairDataset.__init__ args
        )

        self.embeddings = DocumentPoolEmbeddings(
            self.token_embeddings, pooling='mean'
        )
        self.avg_layers = avg_layers

        print('\nEmbeddings Model:')
        print(self.embeddings, end='\n\n')

        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def _get_span_embedding(self, text: str, max_seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Returns embeddings for the given sentence's text,
        in shape: (embeddings_dim,)
        """
        if len(text) < 2:
            print('Sentence is too short: "{}"'.format(text))
            return torch.zeros(self.embeddings.embedding_length, dtype=torch.float32)

        s = Sentence(text)
        if max_seq_len is not None and len(s) > max_seq_len and self.embeddings_type != 'elmo': ## Don't crop ELMo sentences, just an experiment
            s.tokens = s.tokens[:max_seq_len]
        if self.embeddings_type == 'bert' or self.bert_tokenizer is not None:
            sent_len = self.crop_sentence_to_fit_bert(s)
            if sent_len == 0 or len(s) == 0:
                return torch.zeros(self.embeddings.embedding_length, dtype=torch.float32)

        self.embeddings.embed(s)
        return s.embedding

    def get_tokenwise(self, article: NewsArticle) -> torch.Tensor:
        return self._get_tokenwise_embeddings(
            (article.get_title() if self.use_title else "") + article.get_text(),
            self.max_seq_len)

    def get_documentwise(self, article: NewsArticle) -> torch.Tensor:
        """Returns document-wise embeddings"""
        text = (article.get_title() if self.use_title else "") + article.get_text()
        return self._get_span_embedding(text, self.max_seq_len)

    def get_sentencewise(self, article: NewsArticle):
        X = torch.zeros(self.max_seq_len, self.embeddings.embedding_length, dtype=torch.float32)

        # Title embedding
        if self.use_title:
            X[0] = self._get_span_embedding(article.get_title(), self.max_sent_len)

        # Sentence embeddings
        for i, s in enumerate(self.nlp(article.get_text()).sents,
                              1 if self.use_title else 0):
            if i >= self.max_seq_len:
                break

            X[i] = self._get_span_embedding(s.text, self.max_sent_len)

        if self.avg_layers is not None:
            return self._avg_last_n_layers(X, self.avg_layers)
        return X

    def _avg_last_n_layers(self, X, last_n_layers):
        """
        Averages the last_n_layers from the given embedding representation,
         instead of the default concatenation.
        """
        final_emb_len = X.shape[-1] // last_n_layers
        assert X.shape[-1] % last_n_layers == 0

        X_new = torch.zeros(X.shape[0], final_emb_len, dtype=torch.float32)
        for i, emb in enumerate(X):
            for k in range(last_n_layers):
                X_new[i] += emb[k * final_emb_len : (k+1) * final_emb_len]
            X_new[i] /= last_n_layers

        return X_new

    def get_tokenwise_grouped(self, article: NewsArticle) -> torch.Tensor:
        """Returns token-wise embeddings grouped by sentences"""
        X = torch.zeros(self.max_seq_len, self.max_sent_len, self.get_embeddings_dim(), dtype=torch.float32)

        # Title embedding
        if self.use_title:
            X[0] = self._get_tokenwise_embeddings(article.get_title(), self.max_sent_len)

        # Text embeddings
        for i, sent in enumerate(self.nlp(article.get_text()).sents, 1 if self.use_title else 0):
            if i >= self.max_seq_len:
                break
            X[i] = self._get_tokenwise_embeddings(sent.text, self.max_sent_len)

        return X

    def get_embeddings_dim(self) -> int:
        return \
            self.embeddings.embedding_length if self.avg_layers is None else \
            self.embeddings.embedding_length // self.avg_layers
