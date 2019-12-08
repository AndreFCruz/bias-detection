"""
Concrete DocumentEncoder implemented with NLTK.
"""

import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

from encoder import DocumentEncoder


class DocumentEncoderNLTK(DocumentEncoder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    ## Overrides super
    @classmethod
    def _load_resources(cls):
        nltk.download('punkt')
        cls.stemmer = PorterStemmer()
        
    ## Overrides super
    @classmethod
    def tokenize(cls, text):
        tokens = nltk.word_tokenize(text)
        return [cls.stemmer.stem(tok) for tok in tokens]

    ## Implements interface
    def _featurize_sample_stats(self, x, **kwargs):
        """
        Featurizes a document into descriptive statistical features.
        """
        return np.concatenate(
            (self._featurize_text_stats(x.get_title()), self._featurize_text_stats(x.get_text()))
        )

    def _featurize_text_stats(self, text):
        
        sentences = sent_tokenize(text)
        num_sentences = len(sentences)

        ## Average sentence length
        ## -> sent. len. in words
        sent_length_words = [len(word_tokenize(s)) for s in sentences]
        sent_length_words = (sum(sent_length_words) / len(sent_length_words)) if len(sent_length_words) > 0 else 0

        ## -> sent. len. in characters
        sent_length_chars = [len(s) for s in sentences]
        var_sent_length_chars = np.var(sent_length_chars)
        sent_length_chars = (sum(sent_length_chars) / len(sent_length_chars)) if len(sent_length_chars) > 0 else 0

        ## Average word length
        words = [w for w in word_tokenize(self.clean_text(text))]
        word_length = [len(w) for w in words]

        word_length_var = np.var(word_length)     ## Variance of word length
        word_length_avg = (sum(word_length) / len(word_length)) if len(word_length) > 0 else 0  ## Average word length

        ## Frequency of punctuation
        punct_freq = [1 if c in DocumentEncoder.punctuation else 0 for c in text if c != ' ']
        punct_freq = sum(punct_freq) / len(punct_freq) if len(punct_freq) > 0 else 0

        ## Frequency of capital case letters
        capital_freq = [1 if c.isupper() else 0 for c in text if c.isalpha()]
        capital_freq = sum(capital_freq) / len(capital_freq) if len(capital_freq) > 0 else 0

        ## Ratio of types / atoms (vocabulary richness)
        previously_seen = set()
        total_types, total_atoms = 0, 0
        for w in words:
            w = self.stemmer.stem(w) ## NOTE use stemmed words?
            if w not in previously_seen:
                previously_seen.add(w)
                total_types += 1
            total_atoms += 1
        type_token_ratio = (total_types / total_atoms) if total_atoms > 0 else 0

        return np.array([
                num_sentences, sent_length_words, sent_length_chars, var_sent_length_chars,
                word_length_avg, word_length_var, punct_freq, capital_freq, type_token_ratio],
                \
                dtype=np.float32
            )
