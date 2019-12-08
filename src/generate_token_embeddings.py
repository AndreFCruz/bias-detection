#!/usr/bin/env python3

"""
Generate token embeddings
Save Tokenizer + word2index + vectors (index2vectors)
"""

import os
import re
import spacy
import gensim
import pickle
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--articles-dir', help='Directory that contains hyperpartisan news articles XML file (may be several)',
                        dest='articles_dir', required=True, type=str, action='append')
    parser.add_argument('--embeddings-path', help='Path to pretrained embeddings file',
                        dest='embeddings_path', required=True, type=str)
    parser.add_argument('--save-dir', help='Where to save generated files',
                        dest='save_dir', default='.', type=str)
    parser.add_argument('--base-name', help='Where to save generated files',
                        dest='base_name', default='', type=str)

    return parser.parse_args()


def save(embeddings_matrix, word2index, args): ## word2index in tokenizer (right?)
    base_name = args.base_name if len(args.base_name) > 0 else os.path.basename(args.embeddings_path)
    base_name = os.path.join(args.save_dir, base_name)

    # Save embeddings
    embeddings_save_path = base_name + '.embeddings.npy'
    print('Saving embeddings to "{}"'.format(embeddings_save_path))
    np.save(embeddings_save_path, embeddings_matrix)

    # Save word2index
    word2index_save_path = base_name + '.word2index.pkl'
    with open(word2index_save_path, 'wb') as f:
        print('Saving tokenizer to "{}"'.format(word2index_save_path))
        pickle.dump(word2index, f)


def load_embeddings_from_path(embeddings_path):
    """
    Loads FastText embeddings from path (either .bin or .vec file), and returns word-vectors index.
    """
    if embeddings_path.endswith('.bin'):
        return gensim.models.fasttext.load_facebook_model(embeddings_path).wv
    elif embeddings_path.endswith('.vec'):
        return gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(embeddings_path)


def extract_texts_and_title(news_data_dir: str):
    """
    Processes articles as a stream and returns mapping {article_id -> article_text}.
    """
    from datasets.hyperpartisan import NewsExtractorFeaturizerFromStream
    from generate_hyperpartisan_dataset import extract_data
    def ef_constructor():
        return NewsExtractorFeaturizerFromStream(lambda x: x.get_title() + ' ' + x.get_text())
    return extract_data(news_data_dir, ef_constructor)


def like_twitter_mention(text: str) -> bool:
    return re.fullmatch(r'^[@#]\w+', text) is not None


def use_token_embedding(token) -> bool:
    """Whether to use/store the embedding for this Token"""
    return token.is_alpha or like_twitter_mention(token.text)


def main():
    args = parse_args()

    ## Load embeddings
    print('Loading pretrained embeddings from "{}"'.format(args.embeddings_path))
    word_vectors = load_embeddings_from_path(args.embeddings_path)

    ## Construct tokenizer
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])

    ## Load articles
    print('Processing files from the following directories:', args.articles_dir)
    texts = list()
    for file_dir in args.articles_dir:
        texts.extend(extract_texts_and_title(file_dir).values())

    ## OOV Embedding
    from nn_utils import generate_random_unit_vec
    oov_embedding = generate_random_unit_vec((word_vectors.vector_size,))

    ## Generate embeddings
    print('Generating embeddings from unique tokens...')
    from typing import Sequence, Dict
    word2index: Dict[str, int] = {'<OOV_TOKEN>': 0}
    index2vectors: Sequence[np.ndarray] = [oov_embedding]

    for x in texts:
        for token in filter(use_token_embedding, nlp(x)):

            token_str = token.text
            if token_str not in word2index:
                try:
                    index2vectors.append(word_vectors.get_vector(token_str))
                    word2index[token_str] = len(index2vectors) - 1
                except KeyError:
                    print('=> Token out-of-vocabulary: "{}"'.format(token_str.encode()))
                    word2index[token_str] = 0   ## OOV Token

    ## Save
    embeddings_matrix = np.zeros((len(index2vectors), word_vectors.vector_size))
    for i, emb in enumerate(index2vectors):
        embeddings_matrix[i] = emb
    save(embeddings_matrix, word2index, args)


if __name__ == '__main__':
    main()
