# -*- coding: utf-8 -*-

import pandas as pd

### Exceptions
def assert_columns_equal(df1, df2, message):
    if set(df1.columns) != set(df2.columns):
        raise Exception(message)


def assert_datetime_index(df):
    if type(df.index) is not pd.core.indexes.datetimes.DatetimeIndex:
        raise Exception("function expected DataFrame with DateTimeIndex")
        
        
def assert_monthly_index(df):
    if ((type(df.index.freq) is not pd.tseries.offsets.MonthBegin)
            and (type(df.index.freq) is not pd.tseries.offsets.MonthEnd)):
        raise IndexError("function expected DataFrame with monthly DateTimeIndex")


def assert_argument_is_int(arg, name):
    if not isinstance(arg, int):
        raise TypeError("{} argument must be an integer".format(name))
        