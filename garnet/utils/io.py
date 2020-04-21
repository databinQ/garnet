# coding: utf-8

"""
@File   : io.py
@Author : garnet
@Time   : 2020/4/15 16:38
"""

import json
import codecs
import pathlib

from .path import check_suffix


def safe_save(file_path, text, mode='w', encoding='utf-8', suffix=None):
    """
    Save string or binary bytes to file. Parent directory will be created if not exist.

    :param file_path: File path to save
    :param text: Content
    :param mode: Use `wb` if text is binary bytes, while `w` if text is string
    :param encoding: Encoding
    :param suffix: If `suffix` is not `None`, suffix check will be performed on `file_path`
    """
    path = pathlib.Path(file_path)
    if suffix is not None:
        check_suffix(path, suffix)

    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    with codecs.open(path, mode, encoding=encoding) as f:
        f.write(text)


def safe_save_json(file_path, dict_data):
    """
    Save `dict` object or list of `dict` objects into `.json` file
    """
    path = pathlib.Path(file_path)
    check_suffix(path, 'json')

    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    with codecs.open(path, 'w', encoding='utf-8') as f:
        if isinstance(dict_data, dict):
            json.dump(dict_data, f, indent=4, ensure_ascii=False)
        else:
            f.write('\n'.join([json.dumps(d).replace('\n', '').replace('\r', '') for d in dict_data]))
