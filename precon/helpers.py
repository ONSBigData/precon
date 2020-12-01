# -*- coding: utf-8 -*-
from typing import Optional, Callable, Union, Sequence

import numpy as np
import pandas as pd
from pandas._typing import Level


def reindex_and_fill(df, other, first='ffill', axis=0):
    """Reindex and fill the DataFrame or Series by given index and axis.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to reindex.
    other : Object of the same data type
        Its row or column indices are used to define the new indices of
        the first parameter, depending on axis parameter.
    first : str
        Direction to fill first.
    axis : {0 or ‘index’, 1 or ‘columns’}
        Whether to compare by the index (0 or ‘index’) or columns
        (1 or ‘columns’). For Series input, axis to match Series index
        on.

    Returns
    -------
    DataFrame
        The reindexed and filled DataFrame.
    """
    reindexed = df.reindex(other.axes[axis], axis=axis)

    if first == 'ffill':
        return reindexed.ffill(axis).bfill(axis)
    elif first == 'bfill':
        return reindexed.bfill(axis).ffill(axis)


def period_window_fill(
        df,
        periods=12,
        freq='MS',
        method='ffill',
        shift_window=0,
        axis=1
        ):
    """Fills NA values in the DataFrame using a rolling period window
    """
    df_out = df.copy()

    starts = df.axes[axis][::periods - 1].shift(shift_window, freq=freq)
    ends = starts.shift(periods - 1, freq=freq)

    period_windows = [slice(start, end) for start, end in zip(starts, ends)]

    for window in period_windows:
        slice_ = axis_slice(window, axis)
        df_out.loc[slice_] = df_out.loc[slice_].fillna(
            method=method, axis=axis
        )

    return df_out


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
    kwargs = {newcol: df[cols].apply(reduce_func, axis=1)}
    df = df.assign(**kwargs)

    if swap:
        try:
            df = swap_columns(df, newcol, cols[swap])

        except IndexError:
            raise IndexError("swap list index is out of range for given cols")

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


def flip(axis):
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
    return str(int(start_year) - 1)


def axis_vals_as_frame(
    df: pd.DataFrame,
    axis: int = 0,
    levels: Optional[Level] = None,
    converter: Callable[[pd.Index], Union[pd.Index, np.ndarray]] = None,
) -> pd.DataFrame:
    """Broadcast axis values across the DataFrame.

    Pick the axis and optional MultIndex level. Optionally
    transform the index object before broadcasting.

    Parameters
    ----------
    df: DataFrame
    axis: {0, 1}, int default 0
        The axis values to broadcast: {0:'index', 1:'columns'}.
    levels: int or str
        Either the integer position or the name of the level.
    converter: callable
        A function to transform an index object.

    Returns
    -------
    DataFrame
        The broadcast axis values with optional transformation.
    """
    vals = {0: df.index, 1: df.columns}.get(axis)

    if levels:
        vals = vals.get_level_values(levels)

    if converter:
        vals = converter(vals)

    # Prepare the values to pass to np.tile which reshapes to the df.
    if not isinstance(vals, np.ndarray):
        vals = vals.values
    reps = (df.shape[axis ^ 1], 1)

    if axis == 0:
        # Reshape vals to a column and reverse reps if axis is 0.
        vals = vals[:, None]
        reps = reps[::-1]

    df_out = df.copy()
    df_out.loc[:, :] = np.tile(vals, reps)

    return df_out


def reduce_to_only_differing_periods(df, axis):
    """Reduces a DataFrame with lots of repeating values over a time
    series to only the periods where the values have changed.
    """
    return df[df.ne(df.shift(1, axis=axis))].dropna()


def subset_shared_axis(
        df: pd.DataFrame,
        other: pd.DataFrame,
        axis: int = 0,
        droplevel: Optional[Union[Level, Sequence[Level]]] = None,
        ) -> pd.DataFrame:
    """Subsets a DataFrame by it's shared axis with the other.

    Optional behaviour to drop levels of the other axis before
    looking for the shared axis.

    Parameters
    ----------
    df : DataFrame
    other : Object of the same data type
        Its indices on the given axis are used to define the subset of
        indices for this object.
    axis : {0, 1} int, default 0
        The axis to subset the shared index.
    droplevel : int, str, or list-like
        If a string is given, must be the name of a level. If
        list-like, elements must be names or positional indexes of
        levels.

    Returns
    -------
    DataFrame
        The subsetted frame.

    """
    if not df.axes[axis].equals(other.axes[axis]):
        
        other_axis = other.axes[axis]
        
        if droplevel:
            other_axis = other_axis.droplevel(droplevel)
        
        shared_axis = df.axes[axis].isin(other_axis)
        return df.loc[axis_slice(shared_axis, axis)]
    
    else:
        return df

