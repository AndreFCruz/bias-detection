"""
Common functions for extracting arguments from the command line ArgParser.
"""

import argparse


def add_train_data_args(parser):
    parser.add_argument('--news-dir', '--train-news-dir', dest='train_news_dir',
                        help='Hyperpartisan news XML directory (TRAIN)',
                        default=None, type=str, metavar='DIR_PATH')
    parser.add_argument('--ground-truth-dir', '--train-ground-truth-dir', dest='train_ground_truth_dir',
                        help='Ground truth XML directory (TRAIN)',
                        default=None, type=str, metavar='DIR_PATH')

    parser.add_argument('--val-news-dir', dest='val_news_dir',
                        help='Hyperpartisan news XML directory (VALIDATION)',
                        default=None, type=str, metavar='DIR_PATH')
    parser.add_argument('--val-ground-truth-dir', dest='val_ground_truth_dir',
                        help='Ground truth XML directory (VALIDATION)',
                        default=None, type=str, metavar='DIR_PATH')
    parser.add_argument('--test-news-dir', dest='test_news_dir',
                        help='Hyperpartisan news XML directory (TEST)',
                        default=None, type=str, metavar='DIR_PATH')
    parser.add_argument('--test-ground-truth-dir', dest='test_ground_truth_dir',
                        help='Ground truth XML directory (TEST)',
                        default=None, type=str, metavar='DIR_PATH')

    add_tensor_dataset_option(parser)

    return parser


def add_tensor_dataset_option(parser):
    parser.add_argument('--tensor-dataset', dest='tensor_dataset',
                        help='Path to a previously serialized dataset (may be multiple datasets; ordered)',
                        type=str, metavar='PATH', action='append')
    return parser


def add_dataset_args(parser):
    parser.add_argument('--granularity', dest='granularity',
                        help='Granularity of embeddings in dataset',
                        type=str, action='append',
                        choices=['token', 'sentence', 'document', 'tokens_grouped_by_sentence'])
    parser.add_argument('--max-seq-len', dest='max_seq_len',
                        help='Maximum tokens to use for training (cutoff)',
                        default=100, type=int, metavar='N')
    parser.add_argument('--max-sent-len', dest='max_sent_len',
                        help='Maximum number of tokens in each sentence (cutoff)',
                        default=50, type=int, metavar='N')
    parser.add_argument('--embeddings-matrix-path', dest='embeddings_matrix_path',
                        help='Path to pre-generated embeddings matrix (needs word2index)',
                        default=None, type=str)
    parser.add_argument('--word2index-path', dest='word2index_path',
                        help='Path to word-to-index mapping (corresponding to the given embeddings matrix)',
                        default=None, type=str)

    parser.add_argument('--embeddings-path', dest='embeddings_path',
                        help='Path to pre-trained embeddings',
                        default=None, type=str)

    ## Embeddings for Flair Dataset
    parser.add_argument('--flair-embeddings', dest='flair_embeddings',
                        help='Type of embeddings to use for Flair-based Dataset',
                        type=str, action='append')
    parser.add_argument('--avg-layers', dest='avg_layers',
                        help='Average last N layers of embeddings, instead concatenating (default)',
                        default=None, type=int, metavar='N')

    parser.add_argument('--dataloader-workers', dest='dataloader_workers',
                        help='Number of workers to use for pytorch\'s Dataloader (0 means using main thread)',
                        default=0, type=int, metavar='N')

    parser.add_argument('--undersampling',
                        help='Whether to use undersampling to balance classes in the dataset',
                        action='store_true')

    return parser

def add_model_args(parser):
    parser.add_argument('--batch-size', dest='batch_size',
                        help='Number of samples per batch (for training)',
                        default=16, type=int, metavar='N')
    parser.add_argument('--seed', dest='seed',
                        help='Random seed for initializing training',
                        default=42, type=int, metavar='N')

    parser.add_argument('--CUDA',
                        help='Whether to use CUDA if available',
                        action='store_true')

    return parser
