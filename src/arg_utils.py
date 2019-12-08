"""
Several functions for constructing datasets, embeddings and other objects from command line arguments.
"""

import torch
from generate_hyperpartisan_dataset import process_and_group_articles

def load_embeddings(args):
    from datasets import PretrainedEmbeddingsWrapper
    if args.embeddings_matrix_path is not None and args.word2index_path is not None:
        print('Loading pre-generated embeddings from matrix / word2index... ', end='')
        return PretrainedEmbeddingsWrapper.from_files(
                        args.embeddings_matrix_path, args.word2index_path)
    elif args.embeddings_path is not None:
        from generate_token_embeddings import load_embeddings_from_path
        print('Loading pretrained embeddings from path: "{}" ... '
                .format(args.embeddings_path), end='')
        return load_embeddings_from_path(args.embeddings_path)
    else:
        print('\n=> **  NOT loading any embeddings :(   **')
        return None


def construct_hyperpartisan_flair_dataset(articles_dir: str, truth_dir: str, embeddings=None, args=None):
    articles = process_and_group_articles(articles_dir, truth_dir)
    return construct_hyperpartisan_flair_dataset_from_articles(articles, args)


def construct_hyperpartisan_flair_dataset_from_articles(articles, args=None):
    from datasets.hyperpartisan import HyperpartisanDatasetFlair
    return  HyperpartisanDatasetFlair(articles) if args is None else \
            HyperpartisanDatasetFlair(
                articles,
                max_seq_len=args.max_seq_len,
                granularity=args.granularity,
                max_sent_len=args.max_sent_len,
                embeddings=args.flair_embeddings,
                avg_layers=args.avg_layers,
                use_cuda=args.CUDA
            )


def construct_hyperpartisan_embeddings_dataset(articles_dir: str, truth_dir: str, embeddings, args):
    from datasets.hyperpartisan import HyperpartisanDatasetEmbeddings
    from generate_token_embeddings import use_token_embedding
    articles = process_and_group_articles(articles_dir, truth_dir)
    return HyperpartisanDatasetEmbeddings(
        articles, embeddings,
        max_seq_len=args.max_seq_len,
        granularity=args.granularity,
        use_token_embedding=use_token_embedding,
        max_sent_len=args.max_sent_len
    )


def construct_hyperpartisan_flair_and_features_dataset(articles_dir: str, truth_dir: str, embeddings=None, args=None):
    articles = process_and_group_articles(articles_dir, truth_dir)

    from datasets.hyperpartisan import HyperpartisanDatasetFlair
    flair_dataset = construct_hyperpartisan_flair_dataset_from_articles(articles, args)

    from generate_hyperpartisan_dataset import generate_featurized_dataset_from_articles
    from encoder import DocumentEncoderSpaCy
    ## TODO wrap featurizer in a torch dataset
    X, Y, _ = generate_featurized_dataset_from_articles(articles, DocumentEncoderSpaCy)
    features_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))

    from datasets import ConcatFeaturesDataset
    concat_dataset = ConcatFeaturesDataset(flair_dataset, features_dataset)
    return concat_dataset


def construct_propaganda_flair_dataset(articles_dir: str, truth_dir: str, embeddings=None, args=None):
    from datasets.propaganda import PropagandaReader, PropagandaDatasetFlair
    reader = PropagandaReader(articles_dir, truth_dir)
    articles = reader.get_articles()
    return PropagandaDatasetFlair(
        articles,
        embeddings=args.flair_embeddings,
        max_seq_len=args.max_seq_len,
        use_cuda=args.CUDA,
    )


def construct_propaganda_features_dataset(articles_dir: str, truth_dir: str, embeddings=None, args=None):
    from datasets.propaganda import PropagandaReader, PropagandaDatasetFeatures
    reader = PropagandaReader(articles_dir, truth_dir)
    articles = reader.get_articles()
    return PropagandaDatasetFeatures(articles)
