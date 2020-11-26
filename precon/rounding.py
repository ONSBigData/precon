"""Functions for special rounding methods."""
from typing import Union

import numpy as np
import pandas as pd

from precon._validation import _handle_axis


def round_and_adjust(
        vals: Union[pd.DataFrame, pd.Series],
        decimals: int,
        axis: int = 0,
        ) -> pd.DataFrame:
    """Rounds values while keeping the sum the same by adjusting.

    Ensures rounded values sum to the same value as unrounded values.
    Uses the strategy of picking the values with the greatest
    difference to their rounded values as the ones to adjust.

    Otherwise known as constrained rounding.

    Parameters
    ----------
    vals: DataFrame or Series
        Values to adjust.
    decimals : int
        Number of decimal places to round each column to.
    axis : {0 or ‘index’, 1 or ‘columns’}, default 0
        Axis along which the function is applied.

    Returns
    -------
    DataFrame: The rounded and adjusted values.

    """
    axis = _handle_axis(axis)

    if isinstance(vals, pd.Series):
        adjustments = _get_adjustments(vals, decimals)

    elif isinstance(vals, pd.DataFrame):
        adjustments = vals.apply(_get_adjustments, axis, args=[decimals])

    adjusted_vals = vals + adjustments

    return adjusted_vals.round(decimals)


def _get_adjustments(vals: pd.Series, decimals: int) -> pd.Series:
    """Return a Series of adjustments to make.

    Identifies how many adjustments needed from the rounding errors,
    then identifies which values need to be adjusted, and finally
    returns a Series with the adjustments.
    """
    # Get the rounding factor and adjustment value.
    rounding_factor = 10**decimals
    adjustment = 0.5 / rounding_factor

    # Errors > 0.5 between rounded and unrounded means that adjustment
    # is needed.
    errs = vals.subtract(vals.round(decimals))
    tot_err = errs.sum()

    no_of_adjustments = int(tot_err.round(decimals) * rounding_factor)

    # Create a zeros Series to fill with adjustments.
    adjustments = pd.Series(dtype=float).reindex_like(vals).fillna(0)

    # Fill only those we need to adjust with an adjustment.
    to_adjust = _get_values_to_adjust(errs, decimals, no_of_adjustments)
    adjustments.loc[to_adjust] = adjustment * np.sign(no_of_adjustments)

    return adjustments


def _get_values_to_adjust(errs, decimals, no_of_adjustments):
    """Return index keys where greatest rounding errors occur."""
    # Rank order changes depending on the sign of no_of_adjustments.
    asc = (np.sign(no_of_adjustments) == -1)

    diff_ranked = errs.sort_values(ascending=asc)

    # Select only as many as needed.
    return diff_ranked.index[range(0, abs(no_of_adjustments))]
