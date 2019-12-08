"""
Concrete DocumentEncoder implemented with SpaCy.
"""

import spacy
import numpy as np
from encoder import DocumentEncoder

class DocumentEncoderSpaCy(DocumentEncoder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.features_shape = None

    ## Overrides super
    @classmethod
    def _load_resources(cls):
        cls.nlp = spacy.load('en_core_web_sm', disable=['ner'])

    ## Overrides super
    @classmethod
    def tokenize(cls, text):
        return [w.lemma_ for w in cls.nlp(text, disable=['parser', 'ner', 'tagger'])]
        ## May need pipe='tagger'; actually works without it and probably it's not that worse

    ## Overrides super
    def featurize(self, X):
        data = None
        ## Parse all articles in parallel
        text_docs = self.nlp.pipe((x.get_text() for x in X), n_threads=-1, batch_size=128)
        title_docs = self.nlp.pipe((x.get_title() for x in X), n_threads=-1, batch_size=128)
        ## NOTE this improves performance only marginally, thanks python!

        for idx, (article, text_doc, title_doc) in enumerate(zip(X, text_docs, title_docs)):
            featurized = self.featurize_sample(article, text_doc=text_doc, title_doc=title_doc)
            if data is None:
                data = np.ndarray((len(X), *featurized.shape), dtype=np.float32)
                self.features_shape = featurized.shape
            data[idx] = featurized
            if idx > 0 and idx % 500 == 0:
                print('Progress: {} / {}'.format(idx, len(X)))

        return data

    ## Overrides super
    def _featurize_sample_stats(self, x, text_doc=None, title_doc=None, **kwargs):
        """
        Featurizes a document into descriptive statistical features.
        May be provided the article's document already processed/parsed,
         to maximize multi-threaded operations.
        """
        return np.concatenate(
            (self._featurize_text_stats(x.get_title(), title_doc), self._featurize_text_stats(x.get_text(), text_doc))
        )

    def _featurize_text_stats(self, text, doc=None):
        if doc is None:
            doc = self.nlp(text)

        ## Number of sentences
        num_sentences = len(list(doc.sents))

        if num_sentences == 0:
            if self.features_shape is not None:
                # NOTE This is somewhat hardcoded, my break if feature structure is altered
                # currently assumes title-features + text-features + 2 meta-data-features
                return np.zeros(((np.array(self.features_shape) - 2) / 2).astype(int), dtype=np.float32)
            else:
                ## This only happens when the first document ever evaluated has no sentences
                return np.zeros((26,), dtype=np.float32)

        ## Average + Variance of sentence length (in words and characters)
        sent_word_len = [len(s) for s in doc.sents]
        avg_sent_word_len = np.mean(sent_word_len)
        # var_sent_word_len = np.var(sent_word_len)

        sent_char_len = [s.end_char - s.start_char for s in doc.sents]
        avg_sent_char_len = np.mean(sent_char_len)
        var_sent_char_len = np.var(sent_char_len)

        ## Average word length
        word_len = [len(t) for t in doc if t.is_alpha]
        avg_word_len = np.mean(word_len)
        var_word_len = np.var(word_len)

        ## Frequency of punctuation characters ## TODO repeated on frequency of tokens per PoS class ?
        punct_freq = [1 if c in DocumentEncoder.punctuation else 0 for c in text if c != ' ']
        punct_freq = np.mean(punct_freq)

        ## Frequency of capital case letters
        capital_freq = [1 if c.isupper() else 0 for c in self.clean_text(text) if c.isalpha()]
        capital_freq = np.mean(capital_freq)

        ## Ratio of types / atoms (vocabulary richness)
        previously_seen = set()
        total_types, total_tokens = 0, 0
        for w in doc:
            if w.lemma not in previously_seen:
                previously_seen.add(w.lemma)
                total_types += 1
            total_tokens += 1
        type_token_ratio = total_types / total_tokens if total_tokens > 0 else 0

        ## Average height of parse tree
        avg_tree_height = np.mean([tree_height(s.root) for s in doc.sents])

        ## Percentage of tokens per type
        percent_pos_tokens = dict()
        for tok in doc:
            percent_pos_tokens[tok.pos] = 1 + \
                (percent_pos_tokens[tok.pos] if tok.pos in percent_pos_tokens else 0)
        
        MIN_POS_ID, MAX_POS_ID = 83, 102    ## 20 PoS classes
        pos_tag_ids = set(range(MIN_POS_ID, MAX_POS_ID + 1))
        pos_tag_ids = pos_tag_ids - {83, 88, 98, 102}   ## These PoS tags are extremely rare
        sum_percent_pos_tokens = sum(percent_pos_tokens.values())
        percent_pos_tokens = [0 if i not in percent_pos_tokens else (percent_pos_tokens[i] / sum_percent_pos_tokens) \
                                for i in pos_tag_ids]

        return np.array([
            num_sentences, avg_sent_word_len, avg_sent_char_len, var_sent_char_len,
            avg_word_len, var_word_len, punct_freq, capital_freq, type_token_ratio,
            avg_tree_height, *percent_pos_tokens],
            \
            dtype=np.float32
        )


def tree_height(root):
    """
    Calculates height of parse tree starting at token root.
    """
    if not list(root.children):
        return 1
    else:
        return 1 + max(tree_height(child) for child in root.children)

