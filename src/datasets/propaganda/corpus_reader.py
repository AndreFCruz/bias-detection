"""
Corpus reader for the "Propaganda Detection" dataset,
from NLP4IF 2019.

Reads the Corpus from structured txt files into memory.
"""

import re
import os

class PropagandaReader:

    def __init__(self, articles_dir, labels_dir=None):
        self.articles_dir = articles_dir
        self.labels_dir = labels_dir

        self.articles = None

    def extract_articles(self):
        articles = dict()
        for file_name in os.listdir(self.articles_dir):
            with open(self.articles_dir + '/' + file_name, 'r', encoding='utf-8') as file_in:
                art = PropagandaArticle.from_file(file_in)
                articles[art.get_id()] = art

        self.articles = articles
        return self.articles

    def load_labels(self):
        if self.articles is None:
            print('Must load articles before loading articles\' labels')
            return
        if self.labels_dir is None:
            print('Not loading article labels, as data was not provided')
            return

        for file_name in os.listdir(self.labels_dir):
            with open(self.labels_dir + '/' + file_name, 'r', encoding='utf-8') as file_in:
                id = PropagandaArticle.get_id_from_name(file_name)
                labels = PropagandaArticle.labels_from_file(file_in)
                self.articles[id].set_labels(labels)

    def get_articles(self):
        if self.articles is None:
            self.extract_articles()
            self.load_labels()

        assert self.articles is not None
        return self.articles


class PropagandaArticle:
    """
    Class representing a Propaganda Article extracted from the
     "Propaganda Detection" dataset.
    """

    @classmethod
    def from_file(clazz, file_in):
        # extract article id from file path
        id = PropagandaArticle.get_id_from_name(file_in.name)

        # extract article sentences
        sents = [l.strip() for l in file_in.readlines()]

        return clazz(id, sents)

    @staticmethod
    def labels_from_file(file_in):
        lines = [l.strip().split('\t') for l in file_in.readlines()]

        ## Assert file name is coherent with file content
        assert PropagandaArticle.get_id_from_name(file_in.name) == lines[0][0]

        labels = [l[2] for l in lines]  ## ordered labels for each sentence
        return labels

    @staticmethod
    def get_id_from_name(file_name):
        ## 0th group is full match, 1st group is article id
        return re.fullmatch(r'(?:.*[/])?article([\d]+)[\.](?:txt|task-SLC[\.]labels)', file_name)[1]

    def __init__(self, id, sentences):
        self.id = id
        self.sentences = sentences

        self.labels = None

    def get_id(self):
        return self.id

    def get_title(self):
        return self.sentences[0]

    def get_sentences(self):
        return self.sentences

    def set_labels(self, labels):
        """Binary propaganda labels for each sentence"""
        self.labels = labels
        assert len(self.labels) == len(self.sentences)

    def get_label(self, idx):
        return self.labels[idx] if self.labels is not None else ''

    def get_label_val(self, idx):
        label = self.get_label(idx)
        if label == 'propaganda':
            return 1
        elif label == 'non-propaganda':
            return 0
        else:
            print('.')  ## Trying to access inexistent label
            return 0.5

    def __getitem__(self, idx):
        return self.sentences[idx]

    def __len__(self):
        return len(self.sentences)
