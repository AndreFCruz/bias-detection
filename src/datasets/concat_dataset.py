"""
Dataset for concatenating two datasets along the sample axis.
Two datasets of size: (length, features_1), (length, features_2)
 become one dataset of size (length, features_1 + features_2)
"""

import torch
from torch.utils.data import Dataset

class ConcatFeaturesDataset(Dataset):
    """
    NOTE For tensorized data this is the same as the TensorDataset :)
    """

    def __init__(self, *datasets):
        self.datasets = datasets
        assert len({len(d) for d in self.datasets}) == 1, 'Datasets must all have the same number of samples!'

    ## NOTE this is probably stackable/recursible
    ## (with inner datasets of the same type)
    def __getitem__(self, idx):
        items = [ds[idx] for ds in self.datasets]
        Xs = tuple(atomic_x for *x, y in items for atomic_x in x)   # concatenation of all datasets' items (X)
        y = items[0][1]                                             # first dataset's document label (y)

        return Xs + (y,)   # tuples may be unpacked on function return

    def __len__(self):
        return len(self.datasets[0])

    def shape(self) -> tuple:
        return self.datasets[0].shape()
