# -*- coding: utf-8 -*-

from typing import Sequence

class Params(object):
    """Holds and returns parameters for pytest paramaterisation.
    
    Parameters
    ----------
    name: function name under test
    test_inputs: the inputs injected using fixtures
    """
    name: str = None
    test_inputs: Sequence[str] = None
    
    def __init__(self, **kwargs):
        """Stores any number of kwargs in the instance dict."""
        self.__dict__.update(kwargs)
    
    def get_test_cases(self):
        """Returns a tuple of fixture names that are a combination of
        name, test inputs, and test case.
        """
        return tuple(
            [self.name + "_" + var +"_" + self.case
             for var in self.test_inputs]
        )
    
    def get_remaining_attrs(self):
        """Returns all instance attributes except 'case'."""
        return tuple([v for k,v in self.__dict__.items() if k != 'case'])
    
    
    def to_tuple(self):
        """Returns the test parameters as a tuple."""
        return self.get_test_cases() + self.get_remaining_attrs()
