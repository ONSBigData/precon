# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:51:16 2020

@author: edmunm
"""
import inspect
import functools

from natstats._error_handling import _handle_axis, _check_valid_pandas_arg
from natstats.helpers import flip

def get_func_defaults(func):
    """ """
    argspec = inspect.getargspec(func)
    positional_count = len(argspec.args) - len(argspec.defaults)
    return dict(zip(argspec.args[positional_count:], argspec.defaults))


def validate_args(func):
    """ """
    @functools.wraps(func)
    def wrap_checker(*args, **kwargs):
        """ """
        if 'axis' in kwargs:
            kwargs['axis'] = _handle_axis(kwargs['axis'])  
        else:
            kwargs.update(get_func_defaults(func))
        
        arg_names = [
            n for n in inspect.signature(func).parameters.keys()
            if n not in list(kwargs)
        ]
        
        for i, arg in enumerate(args):
            _check_valid_pandas_arg(arg, arg_names[i], flip(kwargs['axis']))
            
        value = func(*args, **kwargs)
        
        return value
    
    return wrap_checker
