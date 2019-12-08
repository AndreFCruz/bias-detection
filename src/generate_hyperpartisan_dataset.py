#!/usr/bin/env python3

"""
Generate/serialize featurized dataset -- for "Hyperpartisan News Detection" @SemEval-2019.
"""

import sys, os
import xml, argparse
import numpy as np
from datasets.hyperpartisan import NewsExtractor, \
                                   GroundTruthExtractor, \
                                   NewsExtractorFeaturizerFromStream
from itertools import chain
from typing import Dict

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

    parser.add_argument('--normalize',
                        help='Whether to normalize dataset features',
                        action='store_true')

    return parser.parse_args()


def extract_data(data_dir, data_extractor_constructor):
    data_extractor = data_extractor_constructor()
    for file in os.listdir(data_dir):
        if file.endswith('xml'):
            print('Extracting data from "{}"'.format(file))
            with open(data_dir + '/' + file, 'r', encoding='utf-8') as in_file:
                parser = xml.sax.make_parser()
                parser.setContentHandler(data_extractor)

                source = xml.sax.xmlreader.InputSource()
                source.setByteStream(in_file)
                source.setEncoding('utf-8')

                parser.parse(source)

    return data_extractor.get_data()


def process_and_group_articles(news_data_dir, ground_truth_dir):
    """
    Extract articles from news_data_dir, and fill with the truth values
     from ground_truth_dir.
    @returns set of articles.
    """
    articles = extract_data(news_data_dir, NewsExtractor)
    if ground_truth_dir is not None:
        ground_truth = extract_data(ground_truth_dir, GroundTruthExtractor)
        for key, val in ground_truth.items():
            articles[key].set_ground_truth(*val)

    return [articles[k] for k in sorted(articles)]


def process_and_group_articles_as_stream(news_data_dir, ground_truth_dir, doc_encoder):
    """
    Process data stream from articles in news_data_dir, and immediately
     featurize articles into data (not needing to keep articles in memory).
    @returns (X, y) featurized dataset
    """
    assert news_data_dir is not None and ground_truth_dir is not None,\
            "news_data_dir or ground_truth_dir not provided"

    ef_constructor = lambda: NewsExtractorFeaturizerFromStream(doc_encoder)
    featurized = extract_data(news_data_dir, ef_constructor)
    ground_truth = extract_data(ground_truth_dir, GroundTruthExtractor)

    print('Moving data to ndarray...')
    X, y = None, None
    for idx, (key, val) in enumerate(featurized.items()):
        if X is None:
            X = np.ndarray((len(featurized), *(val.shape)), dtype=val.dtype)
            y = np.ndarray((len(ground_truth),), dtype=np.bool)

        X[idx] = val
        y[idx] = (ground_truth[key][0] == 'true')

    return X, y


def extract_texts(news_data_dir: str) -> Dict[str, str]:
    """
    Processes articles as a stream and returns mapping {article_id -> article_text}.
    """
    ef_constructor = lambda: NewsExtractorFeaturizerFromStream(lambda x: x.get_text())
    return extract_data(news_data_dir, ef_constructor)


def generate_featurized_dataset(news_data_dir, ground_truth_dir, doc_encoder_constructor, normalize=False):
    articles = process_and_group_articles(news_data_dir, ground_truth_dir) \
        if ground_truth_dir is not None else extract_data(news_data_dir, NewsExtractor)

    return generate_featurized_dataset_from_articles(articles, doc_encoder_constructor, normalize=normalize)


def generate_featurized_dataset_from_articles(articles, doc_encoder_constructor, normalize=False):
    ## Transform data
    print('Featurizing articles...')
    encoder = doc_encoder_constructor(texts=(x.get_text() for x in articles), titles=(x.get_title() for x in articles))
    X = encoder.featurize(articles)
    Y = np.fromiter(map(lambda a: a.is_hyperpartisan(), articles), dtype=np.bool) \
        if articles[0].get_hyperpartisan() is not None else None

    if normalize:
        from sklearn.preprocessing import normalize
        X = normalize(X, axis=0)

    Y = np.reshape(Y, (len(Y), 1))
    return X, Y, encoder


if __name__ == '__main__':
    import numpy as np
    import pickle
    from encoder import DocumentEncoderNLTK, DocumentEncoderSpaCy
    
    ## Command-line arguments
    args = parse_args()

    X, Y, doc_encoder = generate_featurized_dataset(
        args.news_dir, args.ground_truth_dir, DocumentEncoderSpaCy,
        normalize=args.normalize
    )

    base_path = args.save_path[:-4] if args.save_path.endswith('.npz') else args.save_path
    base_path += '_{}'.format(X.shape[-1])

    ## Saving DocumentEncoder
    doc_encoder_path = base_path + '_DocEncoder.pickle'
    print('Savind DocumentEncoder to "{}"'.format(doc_encoder_path))
    pickle.dump(doc_encoder, open(doc_encoder_path, 'wb'))

    ## Saving Dataset
    dataset_path = base_path + '.npz'
    print('Saving generated dataset to "{}"'.format(dataset_path))
    np.savez_compressed(dataset_path, X=X, Y=Y)
