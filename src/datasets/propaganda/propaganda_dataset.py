"""
Top-level interface for the "Propaganda Detection" @NLP4IF 2019.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Union, Any
from .corpus_reader import PropagandaArticle

class PropagandaDataset(Dataset):
    """
    Abstract class for a Propaganda News Dataset
    """

    def __init__(self, articles: Union[List[PropagandaArticle], Dict[Any, PropagandaArticle]], **kwargs):
        super().__init__(**kwargs)
        self.articles = list(articles.values()) if type(articles) == dict else articles

        self.id_to_idx = {a.get_id(): idx for idx, a in enumerate(self.articles)}

        ## array of samples, each pointing to the respective article and sentence within
        self.samples = [(art_idx, sent_idx) \
            for art_idx in range(len(self.articles)) \
            for sent_idx in range(len(self.articles[art_idx]))
        ]

        assert sum(len(a) for a in self.articles) == len(self.samples)

    # Abstract Method
    def featurize_sample(self, sentence, article):
        """
        Featurize the given sample.
        """
        raise NotImplementedError()

    def __getitem__(self, sample_idx):
        art_idx, sent_idx = self.samples[sample_idx]
        
        sample_label = self.articles[art_idx].get_label_val(sent_idx)
        X = self.featurize_sample(self.articles[art_idx][sent_idx], self.articles[art_idx])
        y = torch.tensor([sample_label], dtype=torch.float32)

        return X, y

    def __len__(self):
        return len(self.samples)
