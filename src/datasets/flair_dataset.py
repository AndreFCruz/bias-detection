"""
Generic Flair-based dataset
"""

import torch
import flair
from flair.data import Sentence
from flair.embeddings import BertEmbeddings, FlairEmbeddings, ELMoEmbeddings, WordEmbeddings
from typing import Sequence, List, Tuple, Optional

class FlairDataset:
    """
    Base class for a Flair-based dataset (uses flair to extract word/span embeddings).
    """

    def __init__(self, embeddings: Sequence[str], use_cuda: bool = False, **kwargs):
        super().__init__(**kwargs)  ## kwargs needed for MRO resolution in multiple-inheritance

        if torch.cuda.is_available() and use_cuda:
            flair.device = torch.device('cuda')
        else:
            flair.device = torch.device('cpu')

        self.embeddings = None
        self.embeddings_type = None
        self.bert_tokenizer = None
        self.embeddings_names = embeddings
        self.token_embeddings = self.construct_embeddings(embeddings)

    @staticmethod
    def extract_embedding_configs(string, sep='-', opts_sep='|') -> Tuple[str]:
        """
        Extracts name of embeddings in form
        "<prefix>-<name-of-embeddings>[|<options-file-path>|<weights-file-path>]"

        Returns (embedding-type, embedding-name, options-file, weights-file)
        """
        sep_1 = string.find(sep)
        if sep_1 < 0: return string, None, None, None

        sep_2 = string.find(opts_sep, sep_1 + 1)
        sep_3 = string.find(opts_sep, sep_2 + 1)
        if sep_2 < 0 or sep_3 < 0:
            return string[ : sep_1], string[sep_1 + 1: ], None, None

        return \
            string[ : sep_1],           \
            string[sep_1 + 1: sep_2],   \
            string[sep_2 + 1: sep_3],   \
            string[sep_3 + 1: ]

    def construct_embeddings(self, embeddings: Sequence[str]) -> List[flair.embeddings.TokenEmbeddings]:
        ret = list()
        for name in embeddings:
            emb_type, emb_name, o_file, w_file = FlairDataset.extract_embedding_configs(name)
            self.embeddings_type = emb_type.lower()

            if name.startswith('bert'):
                embs = BertEmbeddings(emb_name)
                self.bert_tokenizer = embs.tokenizer
                ret.append(embs)
            elif name.startswith('flair'):
                ret.append(FlairEmbeddings(emb_name))
            elif name.startswith('elmo'):
                embs = ELMoEmbeddings(emb_name, options_file=o_file, weight_file=w_file)
                ret.append(embs)
            elif name.startswith('word'):
                ret.append(WordEmbeddings(emb_name))
            else:
                raise ValueError('Invalid Embedding Type: "{}"'.format(name))

        return ret

    def _get_tokenwise_embeddings(self, text: str, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Returns token embeddings for the given sentence,
        in shape: (sequence_length, embeddings_dim)

        If no seq_len is passed, uses the length of the sentence.
        """
        if len(text) < 2:
            print('Sentence is too short: "{}"'.format(text))
            return torch.zeros(
                seq_len if seq_len is not None else self.max_seq_len,
                self.get_embeddings_dim(), dtype=torch.float32)

        s = Sentence(text)

        if seq_len is None:
            seq_len = len(s)
        else:
            s.tokens = s.tokens[:seq_len]

        if self.embeddings_type == 'bert' or self.bert_tokenizer is not None:
            sent_len = self.crop_sentence_to_fit_bert(s)
            if sent_len == 0 or len(s) == 0:
                return torch.zeros(seq_len, self.get_embeddings_dim(), dtype=torch.float32)

        embs = torch.zeros(seq_len, self.get_embeddings_dim(), dtype=torch.float32)
        self.embeddings.embed(s)
        for i, tok in enumerate(s):
            if i >= seq_len:
                break
            embs[i] = tok.embedding

        return embs

    def _get_span_embedding(self, text: str, max_seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Returns embeddings for the given sentence's text,
        in shape: (embeddings_dim,)
        """
        tokenwise_embeddings = self._get_tokenwise_embeddings(text, max_seq_len)
        return torch.sum(tokenwise_embeddings, dim=0, keepdim=False) / tokenwise_embeddings.size(0)

    def crop_sentence_to_fit_bert(self, sent, max_seq_len=512, delta=50) -> int:
        """Crop a given sentence to fit the BERT encoder (max. 512 tokens)"""
        assert self.bert_tokenizer is not None

        ## This is a hacky way to trim a sentence down to Bert's 512 length limit :)
        sent_length = len(self.bert_tokenizer.tokenize(sent.to_original_text()))
        while sent_length > (max_seq_len - 2):
            if len(sent.tokens) > delta:
                sent.tokens = sent.tokens[:len(sent) - delta]
            else:
                sent.tokens = sent.tokens[:len(sent.tokens) // 2]
            sent_length = len(self.bert_tokenizer.tokenize(sent.to_original_text()))
            print('.', end='')
        print('#', end='')

        return sent_length

    def get_embeddings_dim(self) -> int:
        return self.embeddings.embedding_length

