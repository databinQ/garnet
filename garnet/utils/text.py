# coding: utf-8

"""
@File   : text.py
@Author : garnet
@Time   : 2020/4/21 15:03
"""

import unicodedata

punctuation = '+-/={(<[' + '\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\xb7\uff01\uff1f\uff61\u3002'


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
