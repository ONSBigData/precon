# -*- coding: utf-8 -*-

import pandas as pd

from error_handling import assert_argument_is_int


def reindex_and_fill(df, new_idx, first='ffill'):
    """Reindex the  DataFrame or Series by the given index, and fill, by 
    first filling forward and then backwards.
    """
    reindexed = df.reindex(index=new_idx)   
    
    if first == 'ffill':
        reindexed_and_filled = reindexed.ffill().bfill()
    elif first == 'bfill':
        reindexed_and_filled = reindexed.bfill().ffill()
    
    return reindexed_and_filled


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
    if newcol in df.columns:
        raise Exception("the name for newcol cannot be an existing column label.")
    
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
        df = df.drop(columns=cols)

    return df


def map_headings(df, labels, map_from, map_to):
    """Quick rename of the columns of given DataFrame, using the labels
    DataFrame."""
    return df.rename(columns=dict(zip(labels[map_from], labels[map_to])))

