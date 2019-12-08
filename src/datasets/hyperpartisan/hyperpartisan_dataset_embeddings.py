"""
Wrapper for the Hyperpartisan News dataset, which provides token-wise representations,
unpooled (in full (num_tokens, embedding_dim) shape), and lazily fetched.
"""

import torch
import spacy
import pickle
import random
from torch.utils.data import Dataset
from typing import Sequence, Callable, Any
from .hyperpartisan_dataset import HyperpartisanDataset

class HyperpartisanDatasetEmbeddings(HyperpartisanDataset):
    """
    Hyperpartisan News Dataset.
    Returns embeddings representations of each article (either
    token-wise, sentence-wise, or document-wise).
    """

    def __init__(self, articles: list, embeddings,
                 max_seq_len: int = 500,
                 granularity: str = 'token',
                 use_title: bool = True,
                 use_token_embedding: Callable[[Any], bool] = lambda tok: tok.is_alpha,
                 max_sent_len: int = 100):

        super().__init__(
            articles, max_seq_len, granularity, max_sent_len=max_sent_len, use_title=use_title
        )

        self.embeddings = embeddings                    # Embeddings to use
        self.use_token_embedding = use_token_embedding  # Filter for which tokens to embed
        self.embeddings_dim = self.embeddings.vector_size
        
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])
        if any('sentence' in gran for gran in self.granularity):
            self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def _get_tokenwise_embeddings(self, tokens: Sequence[spacy.tokens.Token], max_seq_len: int) -> torch.Tensor:
        X = torch.zeros(max_seq_len, self.embeddings_dim, dtype=torch.float32)

        for i, token in enumerate(filter(self.use_token_embedding, tokens)):
            token = token.text
            if i >= max_seq_len:
                break
            try:
                X[i] = torch.from_numpy(self.embeddings.get_vector(token))
            except KeyError:
                ## NOTE Try lower case of token if embedding not found?
                print('Embedding not found for "{}"'.format(token.encode()))

        return X

    def get_tokenwise(self, article):
        """Returns token-wise embeddings"""
        article_text = (article.get_title() + '. ') if self.use_title else ''
        article_text += article.get_text()

        ## Preprocess text (?)
        # article_text = article.preprocess_text(article_text)

        X = self._get_tokenwise_embeddings(
            self.nlp(article_text),
            self.max_seq_len)
        return X

    def get_documentwise(self, article) -> torch.Tensor:
        """Returns document-wise embeddings"""
        X = self.get_tokenwise(article)
        
        # Pool embeddings along the word axis
        X = torch.sum(X, 0, keepdim=True) / self.max_seq_len
        return X

    def get_sentencewise(self, article) -> torch.Tensor:
        """Returns sentence-wise embeddings"""
        X = torch.zeros(self.max_seq_len, self.embeddings_dim, dtype=torch.float32)

        # Title embedding
        if self.use_title:
            title_doc = self.nlp(article.get_title())
            title_embeddings = self._get_tokenwise_embeddings(title_doc, self.max_sent_len)
            X[0] = torch.sum(title_embeddings, 0) / min(self.max_sent_len, len(title_doc))

        # Text embeddings
        for i, sent in enumerate(self.nlp(article.get_text()).sents, 1 if self.use_title else 0):
            if i >= self.max_seq_len:
                break
            sent_embeddings = self._get_tokenwise_embeddings(sent, self.max_sent_len)
            X[i] = torch.sum(sent_embeddings, 0) / min(self.max_sent_len, len(sent))

        return X

    def get_tokenwise_grouped(self, article) -> torch.Tensor:
        """Returns token-wise embeddings grouped by sentences"""
        X = torch.zeros(self.max_seq_len, self.max_sent_len, self.embeddings_dim, dtype=torch.float32)

        # Title embedding
        if self.use_title:
            title_doc = self.nlp(article.get_title())
            X[0] = self._get_tokenwise_embeddings(title_doc, self.max_sent_len)

        # Text embeddings
        for i, sent in enumerate(self.nlp(article.get_text()).sents, 1 if self.use_title else 0):
            if i >= self.max_seq_len:
                break
            X[i] = self._get_tokenwise_embeddings(sent, self.max_sent_len)

        return X

    def get_embeddings_dim(self) -> int:
        return self.embeddings_dim

