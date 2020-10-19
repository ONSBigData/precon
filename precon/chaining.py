# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:49:18 2020

@author: Mitchell Edmunds
@title: Chaining functions
"""
import pandas as pd


def chain(indices, double_link=False, base_periods=None):
    """
    Chain the indices using direct (fixed-base) chaining.

    By default, chains the indices using the direct (fixed-base)
    chaining with a single annual chainlink. Double annual chainlinking
    can be implemented using the `double_link` argument. Alternatively,
    the user can override and specify their own base periods. The
    function can also handle quarterly series if base_periods given.

    Parameters
    ----------
    indices : Series or DataFrame
        The unchained indices to be chained.
    double_link : bool, default 'False', meaning single chainlink
        Boolean switch for annual double chainlinking.
    base_periods : list of int, optional
        The base periods for chainlinking.

    Returns
    -------
    Series or DataFrame
        The chained indices.

    See Also
    --------
    unchain: Unchain the indices using direct (fixed-base) chaining.
    """
    base = indices.copy()

    # # If the initial Jan period is missing then set to 100
    # # Handles indices with different time periods
    # indices = set_first_period_to_100(indices)

    is_base_period = get_base_period_mask(indices, double_link, base_periods)
    # Set base periods to 100
    base[is_base_period] = 100

    # Shift the base by one period to prepare for division
    # Fills first month
    base = base.shift().bfill()

    growth = indices / base
    chained_indices = growth.cumprod() * 100

    return chained_indices.fillna(0)    # Account for zero division


def unchain(indices, double_link=False, base_periods=None):
    """
    Unchain the indices using direct (fixed-base) chaining.

    By default, unchains the indices using the direct (fixed-base)
    chaining with a single annual chainlink. Double annual chainlinking
    can be implemented using the `double_link` argument. Alternatively,
    the user can override and specify their own base months. The
    function can also handle quarterly series if base_periods given.

    Parameters
    ----------
    indices : Series or DataFrame
        The chained indices to be unchained.
    double_link : bool, default 'False', meaning single chainlink
        Boolean switch for annual double chainlinking.
    base_periods : list of int, optional
        The base periods for chainlinking.

    Returns
    -------
    Series or DataFrame
        The unchained indices.

    See Also
    --------
    chain: Chain the indices using direct (fixed-base) chaining.
    """
    # Create a DataFrame of the base values with same index
    base = pd.DataFrame().reindex_like(indices)

    # The values to divide by are the chained indices at base_periods
    is_base_period = get_base_period_mask(indices, double_link, base_periods)

    base.loc[is_base_period] = indices[is_base_period]
    base = base.shift().ffill().bfill()

    unchained_indices = indices / base * 100

    return unchained_indices.fillna(0)  # Account for zero division


def get_base_period_mask(indices, double_link, base_periods):
    """Returns a truthy array if the index of indices is a base_period.
    Works for both quarterly and monthly indices with validation.
    """
    # Ensures base_periods is list of int if given.
    if base_periods:
        base_periods = handle_base_indices_arg(base_periods)

    if check_series_freq(indices, 'M'):
        # Set defaults if not given: either single or double link.
        if not base_periods:
            base_periods = set_monthly_base_periods_defaults(double_link)
        else:
            validate_monthly_base_periods(base_periods)

        return indices.index.month.isin(base_periods)

    elif check_series_freq(indices, 'Q'):
        validate_quarterly_base_periods(base_periods)

        return indices.index.quarter.isin(base_periods)

    else:
        raise ValueError("The frequency of the index cannot be determined. "
                         "Please check the index is a monthly or quarterly "
                         "time series index.")


def set_monthly_base_periods_defaults(double_link):
    """Set default to single link on Jan, or double link on n"""
    base_periods = [1]
    if double_link:
        base_periods.append(12)
    return base_periods


def handle_base_indices_arg(base_periods):
    """Converts arg to list if int, or raises error if not int or list."""
    if not isinstance(base_periods, list):
        if isinstance(base_periods, int):
            return [base_periods]
        else:
            raise ValueError(
                "base_periods must be int or list of ints,"
                f" got type {type(base_periods)}"
            )
    else:
        return base_periods


def validate_quarterly_base_periods(base_periods):
    if not all([base in range(1, 5) for base in base_periods]):
        raise ValueError(
            "Given base periods for a quarterly index must be between"
            " 1 and 4."
        )


def validate_monthly_base_periods(base_periods):
    if not all([base in range(1, 13) for base in base_periods]):
        raise ValueError(
            "Given base periods for a monthly index must be between 1 and 12.")


def check_series_freq(indices, freq):
    """Returns True if the indices have an index with given freq."""
    try:
        indices.index.freq = freq
        return True
    # TODO: Catch the right errors with these except clauses.
    except:
        try:
            indices.index.freq = freq + 'S'
            return True
        except:
            return False


# def set_first_period_to_100(indices):
#     """Handles setting the first first period for both Series
#     and DataFrame.
#     """
#     if isinstance(indices, pd.core.frame.DataFrame):
#         first_row = indices.iloc[0, :]

#         if any((first_row != 100) & (first_row != 0)):
#             indices = indices.apply(set_first_period)

#     elif isinstance(indices, pd.core.series.Series):
#         if (indices.iloc[0] != 100) & (indices.iloc[0] != 100):
#             indices = set_first_period(indices)

#     return indices


# def set_first_period(s):
#     """Sets the first period of the 1st year with data to 100."""
#     s_out = s.copy()

#     if not all(s_out == 0):

#         s_dropped = s.dropna()
#         first_year = str(s_dropped.index.year[0])
#         s_out.loc[first_year].iloc[0] = 100
#         s_out = s_out.sort_index()

#     return s_out
