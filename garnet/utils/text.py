# coding: utf-8

"""
@File   : text.py
@Author : garnet
@Time   : 2020/4/21 15:03
"""

import unicodedata


def is_space(ch):
    """
    Whether `ch` is a space char.

    :param ch: a char
    """
    return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or unicodedata.category(ch) == 'Zs'


def is_punctuation(ch):
    """
    Whether `ch` is a punctuation char.

    :param ch: a char
    """
    code = ord(ch)
    return 33 <= code <= 47 or \
           58 <= code <= 64 or \
           91 <= code <= 96 or \
           123 <= code <= 126 or \
           unicodedata.category(ch).startswith('P')


def is_cjk_character(ch):
    """
    Whether `ch` is a CJK character. CJK characters contains most common characters used in Chinese, Japanese, Korean
    and Vietnamese characters. See
    [CJK Unified Ideographs](https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block) for more info.

    :param ch: a char
    """
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or \
           0x3400 <= code <= 0x4DBF or \
           0x20000 <= code <= 0x2A6DF or \
           0x2A700 <= code <= 0x2B73F or \
           0x2B740 <= code <= 0x2B81F or \
           0x2B820 <= code <= 0x2CEAF or \
           0xF900 <= code <= 0xFAFF or \
           0x2F800 <= code <= 0x2FA1F


def is_control(ch):
    """
    Whether `ch` is a control character

    :param ch: a char
    """
    return unicodedata.category(ch) in ('Cc', 'Cf')
