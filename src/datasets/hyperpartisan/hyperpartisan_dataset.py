"""
Top-level interface for the "Hyperpartisan News Detection" dataset, from SemEval-2019 Task 4.
"""

import torch
import spacy
from torch.utils.data import Dataset
from typing import Union, Set, Sequence, Optional

class HyperpartisanDataset(Dataset):
    """
    Abstract class for a Hyperpartisan News Dataset
    """

    def __init__(self, articles: list, max_seq_len: int,
                 granularity: Union[str, Sequence[str]],
                 max_sent_len: Optional[int] = None,
                 use_title: bool = True,
                 **kwargs):     ## kwargs needed for MRO resolution in multiple-inheritance

        super().__init__(**kwargs)
        self.articles = articles            # Articles to serve
        self.max_seq_len = max_seq_len      # Max sequence length for returned samples
        self.max_sent_len = max_sent_len
        self.use_title = use_title          # Whether to use title of article

        # Granularity for returned samples
        # if multiple values are provided, results is concatenated for all granularity values
        if type(granularity) is str:
            granularity = {granularity}
        self.granularity: Set[str] = set(granularity)

        # Supported levels of granularity
        self.granularity_levels = {
            'token':    self.get_tokenwise,
            'sentence': self.get_sentencewise,
            'document': self.get_documentwise,
            'tokens_grouped_by_sentence': self.get_tokenwise_grouped,
        }
        self.getter_functions = [self.granularity_levels[g] for g in self.granularity]
        assert self.granularity.issubset(self.granularity_levels.keys()), 'Use one of "{}"'.format(self.granularity_levels.keys())
        assert not ('tokens_grouped_by_sentence' in self.granularity and self.max_sent_len is None)

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = self.articles[idx]

        featurized = [getter(article) for getter in self.getter_functions]
        X = torch.cat(featurized, dim=0)
        y = torch.tensor([self.get_article_label(article)], dtype=torch.float32)

        ## Return list of outputs instead of concatenating ?? May be useful for LSTMs (meh)
        return X, y

    def get_tokenwise(self, article) -> torch.Tensor:
        """
        Returns token-level embeddings.
        Output shape: (num_tokens, embed_size)
        """
        raise NotImplementedError()

    def get_sentencewise(self, article) -> torch.Tensor:
        """
        Returns sentence-level embeddings.
        Output shape: (num_sentences, embed_size)
        """
        raise NotImplementedError()

    def get_documentwise(self, article) -> torch.Tensor:
        """
        Returns document-level embeddings.
        Output shape: (1, embed_size)
        """
        raise NotImplementedError()

    def get_tokenwise_grouped(self, article) -> torch.Tensor:
        """
        Returns tokenwise embeddings, grouped by sentences.
        Output shape: (num_sentences, max_tokens_per_sentence, embed_size)
        """
        raise NotImplementedError()

    @classmethod
    def get_article_label(cls, article) -> float:
        if article.get_hyperpartisan() is None:
            # You shouldn't fetch a label for an unlabeled article...
            print('.', end='')
            return 0.5
        
        return 1 if article.is_hyperpartisan() else 0

    def get_embeddings_dim(self) -> int:
        raise NotImplementedError()

    def shape_per_granularity(self, granularity) -> list:
        assert granularity in self.granularity_levels
        if granularity == 'document':
            return [1, self.get_embeddings_dim()]
        elif granularity == 'tokens_grouped_by_sentence':
            return [self.max_seq_len, self.max_sent_len, self.get_embeddings_dim()]
        else:
            return [self.max_seq_len, self.get_embeddings_dim()]

    def shape(self) -> tuple:
        """This dataset's samples' shapes"""
        out_shape = None
        for g in self.granularity:
            g_shape = self.shape_per_granularity(g)
            if out_shape is None:
                out_shape = g_shape
            else:
                assert out_shape[-1] == g_shape[-1]
                out_shape[0] += g_shape[0]

        return tuple(out_shape)
