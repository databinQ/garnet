# coding: utf-8
"""
@File   : __init__.py
@Author : garnet
@Time   : 2020/3/25 16:21
"""

import abc


class Unit(metaclass=abc.ABCMeta):
    """
    Data/text process unit without states, in other words it does not need fitting
    """
    @abc.abstractmethod
    def transform(self, input_):
        ...


class StateUnit(Unit, metaclass=abc.ABCMeta):
    """
    Data/text process unit with states. Need to be fitted before transformation. All states will be gathered during
    fitting process.
    """
    def __init__(self, *args, **kwargs):
        self._context = dict()
        self.fitted = False

    def fit(self, input_):
        self.fitted = True

    def reverse_transform(self, input_):
        ...

    @property
    def context(self):
        return self._context
