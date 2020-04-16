# coding: utf-8

"""
@File   : chain.py
@Author : garnet
@Time   : 2020/4/16 10:54
"""

import typing
import functools

from ..units import Unit


def chain_units_and_funcs(units_and_funcs: typing.List[typing.Optional[Unit, typing.Callable]]) -> typing.Callable:
    """
    Compose unit transformations and functions into a single function.

    Functions to be chained must receive only one parameter, and either return one variable. For function receiving
    more than one parameters, use `functools.partial` to wrap the function and fix parameters regarded as
    hyper-parameters.

    :param units_and_funcs: List of :class:`Unit` and functions.
    """

    @functools.wraps(chain_units_and_funcs)
    def wrapper(args):
        for call in units_and_funcs:
            if isinstance(call, Unit):
                args = call.transform(args)
            else:
                args = call(args)
        return args

    chain_name = " => ".join(
        call.__class__.__name__ if isinstance(call, Unit) else
        call.func.__name__ if isinstance(call, functools.partial) else call.__name__ for call in units_and_funcs
    )
    wrapper.__name__ += " of " + chain_name
    return wrapper
