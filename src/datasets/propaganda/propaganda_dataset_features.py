"""
Dataset for Propaganda Detection based on handcrafted features.
"""

import re
import spacy
import torch
import string
import numpy as np
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from .propaganda_dataset import PropagandaDataset, PropagandaArticle

class PropagandaDatasetFeatures(PropagandaDataset):

    def __init__(self, articles):
        super().__init__(articles)

        self.nlp = spacy.load('en_core_web_sm', disable=['ner'])
        self.vectorizer = self.fit_vectorizer(
            [s for article in self.articles for s in article],
            ngram_range=(1,2), max_features=50, max_df=0.95
        )

        ## Featurize a sample sentence to get sample shape
        self.sample_shape = self[0][0].shape

    @staticmethod
    def clean_text(text):
        text = text.replace("&amp;", "&")
        text = text.replace("&gt;", ">")
        text = text.replace("&lt;", "<")
        text = text.replace("<p>", " ")
        text = text.replace("</p>", " ")
        text = text.replace(" _", " ")
        text = text.replace("–", "-")
        text = text.replace("”", "\"")
        text = text.replace("“", "\"")
        text = text.replace("’", "'")
        text = text.replace("\t", " ")
        text = text.replace("⚪", " ")
        text = text.replace("  ", " ")

        return text

    @staticmethod
    def strip_punctuation(text):
        return re.sub('[^A-Za-z ]', '', text)

    def fit_vectorizer(self, texts, **kwargs):
        vectorizer = TfidfVectorizer(
            tokenizer=self.tokenize_lemmas,
            stop_words='english',
            # stop_words=[self.tokenize_lemmas(w)[0] for w in ENGLISH_STOP_WORDS],
            **kwargs
        )
        print('Fitting sklearn vectorizer...')
        return vectorizer.fit(texts)

    def tokenize_lemmas(self, text):
        # return self.nlp(text, disable=['parser', 'ner'])
        return [w.lemma_ for w in self.nlp(self.strip_punctuation(self.clean_text(text)), disable=['parser', 'ner'])]

    def vectorize_text(self, text):
        return self.vectorizer.transform([text]).toarray().flatten()

    def featurize_sample(self, sent: str, article: PropagandaArticle):
        sent = self.clean_text(sent)

        if len(sent) <= 1:
            return torch.zeros(*self.sample_shape, dtype=torch.float32)

        ## Document-level statistics
        num_sentences = len(article) - 1            ## number of sentences
        
        sent_char_len = [len(s) for s in article]
        avg_sent_char_len = np.mean(sent_char_len)  ## average sentence length
        var_sent_char_len = np.var(sent_char_len)   ## variance of sentence length

        ## Sentence-level statistics
        actual_sent_char_len = len(sent)

        tokens = self.nlp(sent, disable=['parser', 'ner'])
        word_len = [len(tok) for tok in tokens]
        avg_word_len = np.mean(word_len)            ## average word length
        var_word_len = np.var(word_len)             ## variance of word length

        punct_freq = sum(1 if c in string.punctuation else 0 for c in sent) / len(sent)
        capital_freq = sum(1 if c.isupper() else 0 for c in sent) / len(sent)

        ## NOTE Use document-level type-token ratio ?
        type_token_ratio = len({tok.lemma for tok in tokens}) / len(tokens)

        ## tf-idf
        counts = torch.tensor(self.vectorize_text(sent), dtype=torch.float32)

        features = torch.tensor([
            num_sentences, avg_sent_char_len, var_sent_char_len,
            actual_sent_char_len, avg_word_len, var_word_len, punct_freq,
            capital_freq, type_token_ratio],
            dtype=torch.float32
        )

        return torch.cat((features, counts))
        # return features

    def shape(self) -> tuple:
        return self.sample_shape
