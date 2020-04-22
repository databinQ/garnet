# coding: utf-8

"""
@File   : tokenizer.py
@Author : garnet
@Time   : 2020/4/21 11:23
"""

import re
import typing
import unicodedata

from . import StateUnit
from .vocabulary import PAD, UNK, SOS, EOS, CLS, SEP, MASK
from .vocabulary import Vocabulary, BertVocabulary
from ...utils.text import is_space
from ...utils.text import is_punctuation
from ...utils.text import is_cjk_character
from ...utils.text import is_control
from ...utils.text import punctuation


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

    def tokens2ids(self, tokens):
        return [self.token2id(token) for token in tokens]

    def ids2tokens(self, ids):
        return [self.id2token(id_) for id_ in ids]

    def _tokenize(self, text):
        raise NotImplementedError

    def tokenize(self, text, max_length=None, truncate='post'):
        """
        Tokenize text into token sequence, and option can be chosen whether sequence should be truncated

        :param truncate: truncate token sequence when length of sequence is greater than max threshold.
            `pre`, `post` and `None` is available.
        """
        tokens = self._tokenize(text)
        if self.with_sos:
            tokens.insert(0, self._token_sos)
        if self.with_eos:
            tokens.append(self._token_eos)

        max_length = max_length or self.max_length
        if max_length and truncate is not None:
            if truncate == 'pre':
                tokens = self.truncate_sequence(max_length, tokens, pop_index=int(self.with_sos))
            elif truncate == 'post':
                tokens = self.truncate_sequence(max_length, tokens, pop_index=-(int(self.with_eos) + 1))
        return tokens

    @staticmethod
    def truncate_sequence(max_length, tokens, pop_index=-1):
        """
        Truncate token sequence when length of sequence is greater than max threshold

        :param pop_index: 0 or -1, 0 means front tokens will be removed while -1 means latter tokens
        """
        if len(tokens) > max_length:
            delta = len(tokens) - max_length
            return tokens[delta:] if pop_index == 0 else tokens[:-delta]

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
    def __init__(self, dict_path, ignore_case=False, max_length=None):
        super().__init__(vocab=None, with_sos=True, with_eos=True, max_length=max_length)
        self.ignore_case = ignore_case
        self._token_sep = SEP
        self._token_cls = CLS
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

    def tokenize(self, text, max_length=None):
        tokens = self._tokenize(text)
        tokens.insert(0, self._token_cls)
        tokens.append(self._token_sep)

        max_length = max_length or self.max_length
        if max_length:
            tokens, _ = self.truncate_sequence(max_length, tokens, None, -2)
        return tokens

    @staticmethod
    def truncate_sequence(max_length, first_sequence, second_sequence=None, pop_index=-1):
        second_sequence = second_sequence or []
        while True:
            total_length = len(first_sequence) + len(second_sequence)
            if total_length <= max_length:
                break
            elif len(first_sequence) > len(second_sequence):
                first_sequence.pop(pop_index)
            else:
                second_sequence.pop(pop_index)
        return first_sequence, second_sequence or None

    def transform(self, first_text, second_text=None, max_length=None, first_length=None, second_length=None):
        """
        :param first_text: string of first sentence
        :param second_text: string of second sentence. `None` is available
        :param max_length: length of total sequence
        :param first_length: length of first sentence
        :param second_length: length of second sentence
        :return: tuple. First element is token ids sequence, second element is segmentation ids sequence
        """

        first_tokens = self.tokenize(first_text)
        second_tokens = self.tokenize(second_text)[1:] if second_text is not None else None

        max_length = max_length or self.max_length
        if max_length:
            first_tokens, second_tokens = self.truncate_sequence(max_length, first_tokens, second_tokens, -2)

        first_token_ids = self.tokens2ids(first_tokens)
        if first_length is not None:
            first_token_ids = first_token_ids[:first_length]
            first_token_ids.extend([self.token2id(self._token_pad)] * (first_length - len(first_token_ids)))
        first_segment_ids = [0] * len(first_token_ids)

        if second_text is not None:
            second_token_ids = self.tokens2ids(second_tokens)
            if second_length is not None:
                second_token_ids = second_token_ids[:second_length]
                second_token_ids.extend([self.token2id(self._token_pad)] * (second_length - len(second_token_ids)))
            second_segment_ids = [1] * len(second_token_ids)

            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)

        return first_token_ids, first_segment_ids

    def reverse_transform(self, ids, tokens=None):
        tokens = tokens or self.ids2tokens(ids)
        tokens = [token for token in tokens if not self._is_special(token)]

        text = ""
        for i, token in enumerate(tokens):
            if token[:2] == '##':
                text += token[2:]
            elif len(token) == 1 and is_cjk_character(token):
                text += token
            elif len(token) == 1 and is_punctuation(token):
                text += token + ' '
            elif i > 0 and is_cjk_character(token):
                text += token
            else:
                text += ' ' + token

        text = re.sub(r' +', ' ', text)  # eliminate continuous space characters
        text = re.sub(r'\' (re|m|s|t|ve|d|ll) ', '\'\\1 ', text)
        punctuation_pattern = '|'.join([re.escape(p) for p in punctuation])
        punctuation_pattern = '(%s) ' % punctuation_pattern
        text = re.sub(punctuation_pattern, '\\1', text)
        text = re.sub(r'(\d\.) (\d)', '\\1\\2', text)
        return text.strip()

    def rematch(self, text, tokens):
        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self.ignore_case:
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))
        text = normalized_text.lower()

        token_mapping, offset = [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            else:
                token = self._stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping

    @staticmethod
    def _is_special(token):
        return bool(token) and (token[0] == '[') and (token[-1] == ']')

    @staticmethod
    def _stem(token):
        """
        Remove `##` if `token` starts with it.
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token
