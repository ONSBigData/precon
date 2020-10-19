# -*- coding: utf-8 -*-
"""
Common enumerations to be used.

This module provides pre-defined enumerations, as well as functions
for creating new enumerations.

New enumerations can be creaed using the |enumeration| function.

@author: edmunm
"""

from enum import Enum


def enumeration(name, *args, **kwargs):
    """
    Call ``Enum`` with a sequence of (unique) strings to create an
    enumeration object.
    """
    if not (args and all(isinstance(arg, str) and arg for arg in args)):
        raise ValueError(
            f"expected a non-empty sequence of strings, got {args}")

    if len(args) != len(set(args)):
        raise ValueError(f"enumeration items must be unique, got {args}")

    return Enum(name, ' '.join(args))


Adjust = enumeration('forward', 'back')
