# coding: utf-8

"""
@File   : path.py
@Author : garnet
@Time   : 2020/4/15 16:42
"""

import pathlib


def check_suffix(file_path, suffix):
    suffix = suffix if suffix.startswith(".") else "." + suffix
    path = pathlib.Path(file_path)
    assert path.suffix == suffix, "A {} file must be offered, got {}".format(suffix, file_path)
