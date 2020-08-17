"""
Reusable exceptions for the precon package.
"""

import pandas as pd

### Exceptions
class DateTimeIndexError(Exception):
    def __init__(self, msg):
        self.msg = msg
        
class UnequalAxisLabelsError(Exception):
    def __init__(self, msg):
        self.msg = msg

def assert_columns_equal(df1, df2, message):
    if set(df1.columns) != set(df2.columns):
        raise Exception(message)


def assert_pandas_obj(df, name):
    """ """
    if not (isinstance(df, pd.Series)
            or isinstance(df, pd.DataFrame)):
        err_msg = (
            f"Argument passed to {name} should be of pandas type "
            "DataFrame or Series."
        )
        raise ValueError(err_msg)


def assert_datetime_index(df, axis, name=None):
    """ """
    if not isinstance(df.axes[axis], pd.DatetimeIndex):
        if not name:
            err_msg = (
                "Expected DatetimeIndex, but got index type "
                f" {type(df.axes[axis])} instead."
            )
        else:
            err_msg = (
                f"Expected DatetimeIndex for {name}, but got index type "
                f" {type(df.axes[axis])} instead."
            )
        raise DateTimeIndexError(err_msg)
        
        
def assert_monthly_index(df):
    if ((type(df.index.freq) is not pd.tseries.offsets.MonthBegin)
            and (type(df.index.freq) is not pd.tseries.offsets.MonthEnd)):
        raise IndexError("function expected DataFrame with monthly DatetimeIndex")


def assert_argument_is_int(arg, name):
    if not isinstance(arg, int):
        raise TypeError("{} argument must be an integer".format(name))


def _check_valid_pandas_arg(df, name, axis):
    """ """
    assert_pandas_obj(df, name)
    assert_datetime_index(df, axis, name)


def _assert_equal_axis_labels(indices, weights, axis):
    """ """
    labels_in_both = weights.axes[axis].isin(indices.axes[axis])
    if not labels_in_both.all():
        raise UnequalAxisLabelsError(
            "The labels on the given axis need to be equivalent for both"
            " indices and weights in order to aggregate. Weight labels "
            "not in indices are: "
            f"{weights.axes[axis][~labels_in_both].tolist()}."
        )


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