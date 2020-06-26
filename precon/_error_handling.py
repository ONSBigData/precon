"""
Reusable exceptions for the precon package.
"""

import pandas as pd
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries, ABCDatetimeIndex

### Exceptions
class DateTimeIndexError(Exception):
    def __init__(self, msg):
        self.msg = msg


def assert_columns_equal(df1, df2, message):
    if set(df1.columns) != set(df2.columns):
        raise Exception(message)


def assert_pandas_obj(df, name):
    """ """
    if not (isinstance(df, ABCSeries)
            or isinstance(df, ABCDataFrame)):
        err_msg = (
            f"Argument passed to {name} should be of pandas type "
            "DataFrame or Series."
        )
        raise ValueError(err_msg)


def assert_datetime_index(df, axis, name=None):
    """ """
    if not isinstance(df.axes[axis], ABCDatetimeIndex):
        if not name:
            err_msg = (
                "Expected DatetimeIndex, but got index type "
                f" {type(df.index)} instead."
            )
        else:
            err_msg = (
                f"Expected DatetimeIndex for {name}, but got index type "
                f" {type(df.index)} instead."
            )
        raise DateTimeIndexError(err_msg)
        
        
def assert_monthly_index(df):
    if ((type(df.index.freq) is not pd.tseries.offsets.MonthBegin)
            and (type(df.index.freq) is not pd.tseries.offsets.MonthEnd)):
        raise IndexError("function expected DataFrame with monthly DatetimeIndex")


def assert_argument_is_int(arg, name):
    if not isinstance(arg, int):
        raise TypeError("{} argument must be an integer".format(name))


def check_valid_pandas_arg(df, name, axis):
    """ """
    assert_pandas_obj(df, name)
    assert_datetime_index(df, axis, name)


def _handle_axis(axis):
    if axis not in [0, 1, 'columns', 'index']:
        raise ValueError("axis parameter should be one of"
                         " [0, 1, 'columns', 'index']")
        
    if isinstance(axis, str):
        for word, i in {'index': '0', 'columns': '1'}.items():
            axis = axis.replace(word, i)
        axis = int(axis)
    
    return axis
# def assert_