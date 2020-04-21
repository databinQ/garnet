# coding: utf-8

"""
@File   : tokenizer.py
@Author : garnet
@Time   : 2020/4/21 11:23
"""

import typing
import unicodedata

from . import StateUnit
from .vocabulary import PAD, UNK, SOS, EOS, CLS, SEP, MASK
from .vocabulary import Vocabulary, BertVocabulary
from ...utils.text import is_space
from ...utils.text import is_punctuation
from ...utils.text import is_cjk_character
from ...utils.text import is_control


class BaseTokenizer(StateUnit):
    def __init__(self, vocab: typing.Optional[Vocabulary], with_sos=True, with_eos=True, max_length=None,
                 token_sos=SOS, token_eos=EOS, token_pad=PAD, token_unk=UNK,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vocab = vocab
        self.with_sos = with_sos
        self.with_eos = with_eos
        self._token_sos = token_sos
        self._token_eos = token_eos
        self._token_pad = token_pad
        self._token_unk = token_unk
        self.max_length = max_length
        self.fitted = True

    def token2id(self, token):
        """
        Transfer token to index
        """
        return self._vocab[token]

    def id2token(self, id_):
        """
        Transfer id to token
        """
        return self._vocab.id2word(id_)

    def _tokenize(self, text):
        raise NotImplementedError

    def tokenize(self, text, max_length=None):
        tokens = self._tokenize(text)
        if self.with_sos:
            tokens.insert(0, self._token_sos)
        if self.with_eos:
            tokens.append(self._token_eos)

        max_length = max_length or self.max_length
        if max_length:
            pass


    def _word_piece_tokenize(self, word):
        """
        Divide normal word into sub-word.
        """
        if word in self._vocab:
            return [word]

        tokens = []
        start, stop = 0, 0
        while start < len(word):
            stop = len(word)
            sub = word[start:stop]
            while stop > start:
                sub = word[start:stop]
                if start > 0:
                    sub = '##' + sub
                if sub in self._vocab:
                    break
                stop -= 1
            if start == stop:
                stop += 1
            tokens.append(sub)
            start = stop

        return tokens


class BertTokenizer(BaseTokenizer):
    def __init__(self, dict_path, ignore_case=False):
        super().__init__(vocab=None)
        self.ignore_case = ignore_case
        self._token_pad = PAD
        self._token_sep = SEP
        self._token_cls = CLS
        self._token_unk = UNK
        self._token_mask = MASK

        self._vocab = BertVocabulary(dict_path=dict_path)
        self.fitted = True

    def _tokenize(self, text):
        if self.ignore_case:  # text clean process
            text = unicodedata.normalize('NFD', text)
            text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
            text = text.lower()

        spaced = ''
        for ch in text:
            if is_punctuation(ch) or is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif is_space(ch):
                spaced += ' '
            elif is_control(ch) or ord(ch) == 0 or ord(ch) == 0xfffd:
                continue
            else:
                spaced += ch  # number and alphabet letter will stick together

        tokens = []
        for word in spaced.strip().split():
            tokens.extend(self._word_piece_tokenize(word))
        return tokens

    def transform(self, input_):
        pass
