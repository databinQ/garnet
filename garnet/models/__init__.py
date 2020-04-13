# coding: utf-8

"""
@File   : __init__.py
@Author : garnet
@Time   : 2020/4/13 23:36
"""


class BaseModel(object):
    def __init__(self, *args, **kwargs):
        self.built = False

    def build(self, *args, **kwargs):
        raise NotImplementedError

    def call(self, inputs, *args, **kwargs):
        raise NotImplementedError
