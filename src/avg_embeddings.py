"""
Script for averaging word embeddings (last dimension) of a dataset (np.ndarray).
When embeddings are the last layers of an encoder concatenated, it may be useful
 to experiment with the average of the last layers instead of the concatenation.
"""

import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-path', dest='dataset_path',
                        help='Path of saved numpy dataset',
                        required=True, type=str, metavar='PATH')

    parser.add_argument('--avg-layers', dest='avg_layers',
                        help='Average last N layers of embeddings',
                        required=True, type=int, metavar='N')

    parser.add_argument('--save-path', dest='save_path',
                        help='File path to save new dataset to',
                        required=True, type=str, metavar='PATH')

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = np.load(args.dataset_path)
    X, Y = dataset['X'], dataset['Y']

    assert X.shape[-1] % args.avg_layers == 0, \
        'Embeddings dimension must be divisible by {}'.format(args.avg_layers)
    new_embeddings_dim = X.shape[-1] // args.avg_layers
    new_dataset_dim = X.shape[:-1] + (new_embeddings_dim,)

    X_new = np.zeros(new_dataset_dim, dtype=X.dtype)
    for i in range(args.avg_layers):
        X_new += X[..., i * new_embeddings_dim : (i+1) * new_embeddings_dim]
    X_new /= args.avg_layers

    np.savez_compressed(
        args.save_path if args.save_path.endswith('.npz') else (args.save_path + '.npz'),
        X=X_new, Y=Y
    )


if __name__ == '__main__':
    main()
