# -*- coding: utf-8 -*-

import pandas as pd

from precon._error_handling import assert_argument_is_int


def reindex_and_fill(df, like_df, first='ffill', axis=0):
    """Reindex and fill the DataFrame or Series by given index and axis."""
    reindexed = df.reindex(like_df.axes[axis], axis=axis)
    
    if first == 'ffill':
        return reindexed.ffill(axis).bfill(axis)
    elif first == 'bfill':
        return reindexed.bfill(axis).ffill(axis)
    

def swap_columns(df, col1, col2):
    """Swaps the two given columns of the DataFrame."""    
    col_list = list(df.columns)
    a, b = col_list.index(col1), col_list.index(col2)
    col_list[b], col_list[a] = col_list[a], col_list[b]
    return df[col_list]


def reduce_cols(df, newcol, cols, reduce_func, drop=False, swap=None):
    """Creates a new column as a reduction of any number of given 
    columns. Options to choose the reduce function (such as mean or
    sum), drop the given columns, and swap the new column inplace
    with one of the reduced columns.
    """   
    if swap:
        assert_argument_is_int(swap, 'swap')
        try:
            cols[swap]
        except IndexError:
            raise IndexError("swap list index is out of range for given cols")
        
    kwargs = {newcol: df[cols].apply(reduce_func, axis=1)}
    df = df.assign(**kwargs)
    
    if swap:
        df = swap_columns(df, newcol, cols[swap])
    
    if drop:
        cols_to_drop = [c for c in cols if c != newcol] 
        df = df.drop(columns=cols_to_drop)

    return df


def map_headings(df, labels, map_from, map_to):
    """Quick rename of the columns of given DataFrame, using the labels
    DataFrame."""
    mapper = dict(zip(labels[map_from], labels[map_to]))
    if isinstance(df, pd.DataFrame):
        return df.rename(columns=mapper)
    elif isinstance(df, pd.Series):
        if df.name in mapper.keys():
            return df.rename(mapper.get(df.name))
        else:
            return df


def axis_slice(value, axis):
    """Creates a slice for pandas indexing along given axis."""    
    return {0: (value, slice(None)), 1: (slice(None), value)}.get(axis)


def axis_flip(axis):
    """Returns the opposite axis value to the one passed."""
    return axis ^ 1


def _selector(slicer, *args):
    """Selects the given args with the given slice and returns a tuple."""
    sliced_args = []
    for arg in args:
        sliced_args.append(arg.loc[slicer])
    return tuple(sliced_args)


def _get_end_year(start_year):
    """Returns the string of the previous year given the start year."""
    return str(int(start_year)-1)

