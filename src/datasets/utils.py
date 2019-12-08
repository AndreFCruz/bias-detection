"""
Utils for datasets package.
"""
import pickle
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

class PretrainedEmbeddingsWrapper:
    """Wrapper for pretrained and pregenerated embeddings"""

    @staticmethod
    def from_files(embeddings_matrix_path, word2index_path):
        with open(word2index_path, 'rb') as f:
            word2index = pickle.load(f)
        emb_matrix = np.load(embeddings_matrix_path)
        return PretrainedEmbeddingsWrapper(emb_matrix, word2index)

    def __init__(self, embeddings_matrix, word2index, oov_index=0):
        self.embeddings_matrix = embeddings_matrix
        self.word2index = word2index
        self.oov_idx = oov_index

    def get_vector(self, token: str):
        if token in self.word2index:
            return self.embeddings_matrix[self.word2index[token]]
        else:
            print('Embedding for token "{}" not found, using o.o.v. placeholder'
                    .format(token.encode()))
            return self.embeddings_matrix[self.oov_idx]

    @property
    def vector_size(self):
        return self.embeddings_matrix.shape[-1]


def generate_random_unit_vec(shape):
    vec = np.random.rand(*shape)
    return vec / np.linalg.norm(vec)

def dataset_to_tensor(dataset: Dataset) -> Tensor:
    """
    Evaluates a torch Dataset into a Tensor of size:
    (num_samples, ..., .., .)
    """
    dataloader = DataLoader(dataset, batch_size=len(dataset))
    return next(iter(dataloader))[0]
