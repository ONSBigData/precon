# -*- coding: utf-8 -*-
import pandas as pd


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


def index_attrs_as_frame(df, attr=None, axis=0):
    """
    Returns a DataFrame of index attributes (or values if not
    specified), the same shape as the given DataFrame.

    # TODO: Get this working for case where axis = 0 and attr = None
    """
    if attr:
        vals = getattr(df.axes[axis], attr).values
    else:
        vals = df.axes[axis].values

    # Create an axis slice so attrs can be cast to any axis
    if axis == 0:
        slice_ = axis_slice(None, flip(axis))
        vals = vals[slice_]

    # Create an empty DataFrame in original shape for casting
    all_attrs = pd.DataFrame().reindex_like(df)

    all_attrs.loc[:, :] = vals

    # Return with the original attribute dtype
    return all_attrs.astype(vals.dtype)


def reduce_to_only_differing_periods(df, axis):
    """Reduces a DataFrame with lots of repeating values over a time
    series to only the periods where the values have changed.
    """
    return df[df.ne(df.shift(1, axis=axis))].dropna()
