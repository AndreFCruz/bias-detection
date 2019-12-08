#!/usr/bin/env python3

"""
This script is intended to serialize a pytorch Dataset.
e.g. datasets.HyperpartisanDatasetEmbeddings datasets.HyperpartisanDatasetFlair
"""

import os
import sys
import torch
from torch.utils.data import TensorDataset
import argparse
import numpy as np
from tqdm import tqdm
from typing import Sequence


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--news-dir', dest='news_dir',
                        help='Hyperpartisan news XML directory',
                        required=True, type=str, metavar='DIR_PATH')
    parser.add_argument('--ground-truth-dir', dest='ground_truth_dir',
                        help='Ground truth XML directory',
                        default=None, type=str, metavar='DIR_PATH')

    parser.add_argument('--save-path', dest='save_path',
                        help='File path to save dataset on',
                        required=True, type=str, metavar='PATH')

    parser.add_argument('--cpu-threads', dest='cpu_threads',
                        help='Number of OpenMP threads to use',
                        default=None, type=int, metavar='N')

    ## Generic model args
    from arg_parser import add_model_args
    parser = add_model_args(parser)

    from arg_parser import add_dataset_args
    parser = add_dataset_args(parser)

    args = parser.parse_args()
    if args.granularity is None:
        print('\n** Warning: "granularity" NOT set **\n', file=sys.stderr)

    return args


def get_data_from_dataloader(dataloader):
    X, Y = [], []
    for inputs, labels in tqdm(dataloader):
        X.append(inputs)
        Y.append(labels)

    X = torch.cat(X)
    Y = torch.cat(Y)
    return X, Y


def dataset_to_numpy(dataset, dataloader_workers=0, batch_size=32):
    """
    Converts the given pytorch Dataset into a numpy array.
    """
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=dataloader_workers,
        pin_memory=torch.cuda.is_available()
    )

    X, Y = get_data_from_dataloader(dataloader)
    return X.detach().numpy(), Y.detach().numpy()


def serialize_pytorch_dataset(dataset, file_path, dataloader_workers=0, batch_size=32):
    """
    Serializes the given pytorch Dataset into a numpy array file.
    """
    X, Y = dataset_to_numpy(dataset, dataloader_workers=dataloader_workers, batch_size=batch_size)
    np.savez_compressed(
        file_path if file_path.endswith('.npz') else (file_path + '.npz'),
        X=X, Y=Y
    )


def load_serialized_data(file_path) -> Sequence[torch.Tensor]:
    """
    Loads the data in the given path (saved as compressed numpy array)
    intro a pytorch TensorDataset.
    """
    arr = np.load(file_path)
    X, Y = arr['X'], arr['Y']
    return torch.from_numpy(X), torch.from_numpy(Y)


def main():
    ## Parse command line args
    args = parse_args()
    from pprint import PrettyPrinter
    PrettyPrinter(indent=4).pprint(vars(args))
    print()

    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
        print('Setting number of CPU threads: "{}"'.format(args.cpu_threads))

    ## Load embeddings
    from arg_utils import load_embeddings
    embeddings = load_embeddings(args)

    ## Dataset + Dataloader ## NOTE Update dataset here
    from arg_utils import construct_hyperpartisan_flair_dataset, \
                          construct_propaganda_flair_dataset, \
                          construct_propaganda_features_dataset
    construct_dataset = construct_propaganda_features_dataset
    dataset = construct_dataset(args.news_dir, args.ground_truth_dir, embeddings, args)

    ## Save dataset
    serialize_pytorch_dataset(dataset, args.save_path, dataloader_workers=args.dataloader_workers, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
