"""
Document encoder
Transforms documents into numeric features.
"""

from abc import ABC, abstractmethod
import numpy as np
import string
import re

import sklearn.feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import concurrent.futures

class DocumentEncoder(ABC):

    punctuation = set(string.punctuation)

    def __init__(self, texts=None, titles=None, use_counts=False, **kwargs):
        self.use_counts = use_counts

        ## Full dataset of texts (train/test/val), for constructing vocabulary
        texts_cleaned = list(map(DocumentEncoder.clean_text, texts)) if texts is not None else None
        titles_cleaned = list(map(DocumentEncoder.clean_text, titles)) if titles is not None else None

        self._load_resources()

        if self.use_counts and texts_cleaned is not None:
            self.texts_counter = self.fit_counter(texts_cleaned, ngram_range=(1,2), max_features=50, max_df=0.95)
        else:
            self.texts_counter = None

        if self.use_counts and titles_cleaned is not None:
            self.titles_counter = self.fit_counter(titles_cleaned, ngram_range=(1,2), max_features=50, max_df=0.95)
        else:
            self.titles_counter = None

    def featurize(self, X):
        ## NOTE although it seems promising, do not try to parallelize this, python just won't take it :|
        ## NOTE ...  me: watch me :)
        ## NOTE me 24h later: nevermind...
        data = None
        for idx, featurized in enumerate(map(self.featurize_sample, X)):
            if data is None:
                data = np.ndarray((len(X), *featurized.shape), dtype=np.float32)
            data[idx] = featurized

        return data

    def featurize_sample(self, x, **kwargs):
        featurized = np.concatenate(
            (self._featurize_sample_stats(x, **kwargs), self._featurize_sample_metadata(x)))
        if self.texts_counter is not None:
            featurized = np.concatenate((featurized, self._featurize_text_ngrams(self.texts_counter, x.get_text())))
        if self.titles_counter is not None:
            featurized = np.concatenate((featurized, self._featurize_text_ngrams(self.titles_counter, x.get_title())))

        return featurized

    def __call__(self, x):
        return self.featurize_sample(x)

    @classmethod
    @abstractmethod
    def _load_resources(self):
        """
        Loads resources necessary for tokenization or encoding.
        """
        pass

    @abstractmethod
    def _featurize_sample_stats(self, x, **kwargs):
        """
        Featurizes a document into descriptive statistical features.
        """
        pass

    def _featurize_sample_metadata(self, x):
        """
        Featurizes a document's metadata (i.e. non-textual data present in the xml)
        """
        num_internal_links = sum(1 for elem in x.get_links() if elem.get('type') == 'internal')
        num_external_links = sum(1 for elem in x.get_links() if elem.get('type') == 'external')

        return np.array(
            [num_internal_links, num_external_links],
            \
            dtype=np.float32
        )

    @classmethod
    def _featurize_text_ngrams(cls, vectorizer, text):
        """
        Featurizes a document into a vector of n-gram counts, tf-idf, or any other useful metric.
        """
        return vectorizer.transform([DocumentEncoder.clean_text(text)]).toarray().flatten()

    @classmethod
    def top_tfidf_terms(cls, tfidf, text, n=10):
        matrix = tfidf.transform([text])
        feature_array = np.array(tfidf.get_feature_names())
        tfidf_sorting = np.argsort(matrix.toarray()).flatten()[::-1]
        ## NOTE toarray() transforms sparse array to dense array, which kills RAM...

        return feature_array[tfidf_sorting[:n]]

    @classmethod
    def fit_tfidf(cls, texts):
        print('Fitting TF-IDF Vectorizer for {} texts...'.format(len(texts)))
        tfidf = TfidfVectorizer(tokenizer=cls.tokenize, stop_words='english', ngram_range=(1,2))
        ## NOTE can also easily use character ngrams with analyzer='char'
        return tfidf.fit(texts) # returns self

    @classmethod
    def fit_counter(cls, texts, **kwargs):
        print('Fitting CountVectorizer for {} texts...'.format(len(texts)))
        counter = CountVectorizer(
            tokenizer=cls.tokenize,
            stop_words=[cls.tokenize(w)[0] for w in sklearn.feature_extraction.stop_words.ENGLISH_STOP_WORDS],
            **kwargs
        )

        return counter.fit(texts)

    @classmethod
    @abstractmethod
    def tokenize(cls, text):
        pass

    @staticmethod
    def strip_punctuation(text):
        trans_table = dict.fromkeys(map(string.punctuation), None)
        ## or: trans_table = str.maketrans('','',string.punctuation)
        return text.translate(trans_table)

    @staticmethod
    def clean_text(text):
        return re.sub('[^A-Za-z ]', '', text)
