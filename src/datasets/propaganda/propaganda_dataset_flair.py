import torch
from torch.utils.data import Dataset
import flair
from flair.data import Sentence
from flair.embeddings import ELMoEmbeddings, StackedEmbeddings
from .propaganda_dataset import PropagandaDataset, PropagandaArticle
from ..flair_dataset import FlairDataset
from typing import Sequence, Optional

class PropagandaDatasetFlair(PropagandaDataset, FlairDataset):
    """
    Propaganda dataset which uses flair for word/sentence representations.
    """

    def __init__(self, articles,
                 embeddings: Sequence[str] = ['elmo'],                 
                 max_seq_len: int = 80,
                 use_cuda: bool = False):
        super().__init__(
            articles=articles,                          ## PropagandaDataset.__init__
            embeddings=embeddings, use_cuda=use_cuda,   ## FlairDataset.__init__
        )
        self.max_seq_len = max_seq_len
        self.embeddings = StackedEmbeddings(
            self.token_embeddings
        )

    def featurize_sample(self, sent: str, article: PropagandaArticle):
        return self._get_tokenwise_embeddings(sent, self.max_seq_len)

    # def _get_tokenwise_embeddings(self, text: str, seq_len: Optional[int] = None) -> torch.Tensor:
    #     ## TODO use self.max_seq_len ?? or pad sequences later
    #     s = Sentence(text)
    #     if len(s) < 2:
    #         return torch.zeros(1, self.get_embeddings_dim(), dtype=torch.float32)
    #     self.embeddings.embed(s)
    #     return torch.cat([tok.embedding.unsqueeze(0) for tok in s.tokens], dim=0)

    def shape(self) -> tuple:
        return tuple((self.max_seq_len, self.get_embeddings_dim()))
