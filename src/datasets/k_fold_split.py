"""
Utilitary functions for splitting a dataset into subsets.
"""

import torch
from torch.utils.data import Dataset, Subset


def k_fold_split(dataset, k):
    """Return k pairs of (train_subset, test_subset) representing k-fold splits"""
    # TODO
    # can be done by using sklearn, but only with already tensorized dataset
    pass
