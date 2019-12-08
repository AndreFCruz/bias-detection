#!/usr/bin/env python3

"""
Evaluate supplied news data, and classify articles in hyperpartisan / not hyperpartisan.
"""

import sys, argparse
import numpy as np
from generate_hyperpartisan_dataset import extract_data
from datasets.hyperpartisan import NewsExtractorFeaturizerFromStream
import encoder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('classifier', help='Classifier path')
    parser.add_argument('doc_encoder', help='Document encoder path')
    parser.add_argument('input', help='Input dataset directory')
    parser.add_argument('output', help='Output directory')

    args = vars(parser.parse_args())
    return args['classifier'], args['doc_encoder'], args['input'], args['output']


def process_data_as_stream(news_data_dir, doc_encoder):
    """
    Process data stream from articles in news_data_dir, and immediately
     featurize articles into data (not needing to keep articles in memory).
    @returns (X, y) featurized dataset
    """
    extractor_featurizer = lambda: NewsExtractorFeaturizerFromStream(doc_encoder)
    featurized = extract_data(news_data_dir, extractor_featurizer)

    return featurized


def get_data_from_dir(input_dir, doc_encoder):
    featurized = process_data_as_stream(input_dir, doc_encoder)

    X, idx_to_article_id = None, None
    for i, (art_id, val) in enumerate(featurized.items()):
        if X is None:
            X = np.ndarray((len(featurized), *(val.shape)), dtype=np.float32)
            idx_to_article_id = np.ndarray((len(featurized),), dtype=np.int32)

        X[i] = val
        idx_to_article_id[i] = art_id

    return X, idx_to_article_id


def main():
    import pickle
    classifier_path, doc_encoder_path, input_dir, output_dir = parse_args()

    classifier = pickle.load(open(classifier_path, 'rb'))
    doc_encoder = pickle.load(open(doc_encoder_path, 'rb'))
    doc_encoder._load_resources()

    X, idx_to_article_id = get_data_from_dir(input_dir, doc_encoder)
    np.savez_compressed('./processed_data_checkpoint.npz', X=X, idx_to_article_id=idx_to_article_id)

    predictions = classifier.predict(X)

    with open(output_dir + '/prediction.txt', 'w', encoding='utf-8') as ofile:
        for i in range(len(predictions)):
            print('{:07} {} {}'.format(
                idx_to_article_id[i],
                'true' if predictions[i] > 0.5 else 'false',
                ((predictions[i] - 0.5) * 2) if predictions[i] > 0.5 else ((0.5 - predictions[i]) * 2)),
                file=ofile
            )


if __name__ == '__main__':
    main()
