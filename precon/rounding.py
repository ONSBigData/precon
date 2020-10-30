"""
Functions for special rounding methods. Includes function to round
and adjust values to keep the sum of values the same.
"""
from typing import Union

import numpy as np
import pandas as pd

from precon.helpers import axis_slice


def round_and_adjust(
        obj: Union[pd.DataFrame, pd.Series],
        decimals: int,
        axis: int = 0,
        ) -> pd.DataFrame:
    """
    Rounds a set of values ensuring the rounded values sum to the same
    total as the unrounded weights.

    Parameters
    ----------
    obj: Object with values to adjust.
    decimals : Number of decimal places to round each column to.
    axis : Axis to adjust on and preserve total.

    Returns
    -------
    The rounded and adjusted values.
    """
    # Choose the iter method based on the given axis
    iter_dict = {
        0: pd.DataFrame.iterrows,
        1: pd.DataFrame.iteritems,
    }
    iter_method = iter_dict.get(axis)

    if isinstance(obj, pd.core.series.Series):

        adjustments = _get_adjustments(obj, decimals)

    elif isinstance(obj, pd.core.frame.DataFrame):

        # Create an empty DataFrame to fill with adjustments
        adjustments = pd.DataFrame().reindex_like(obj)

        for index, row in iter_method(obj):
            # Create a selector based on the axis
            slice_ = axis_slice(index, axis)

            adjustments.loc[slice_] = _get_adjustments(row, decimals)

    adjusted_obj = obj + adjustments
    return adjusted_obj.round(decimals)


def _get_adjustments(obj, decimals):
    """Return a Series of adjustments to make."""
    # Get the rounding factor and adjustment value
    rounding_factor = 10**decimals
    adjustment = 0.5 / rounding_factor

    # Errors > 0.5 between rounded and unrounded means that adjustment
    # is needed
    errs = obj.subtract(obj.round(decimals)).sum()
    no_of_adjustments = int(errs.round(decimals) * rounding_factor)

    # Create a zeros Series to fill with adjustments
    adjustments = pd.Series(dtype=float).reindex_like(obj).fillna(0)

    to_adjust = _get_values_to_adjust(obj, decimals, no_of_adjustments)
    adjustments.loc[to_adjust] = adjustment * np.sign(no_of_adjustments)

    return adjustments


def _get_values_to_adjust(values, decimals, no_of_adjustments):
    """Get the difference of each value from its rounded value and pick
    values to round by rank depending whether adjusting down or up.
    """
    asc = (np.sign(no_of_adjustments) == -1)

    diff_ranked = (
        values
        .subtract(values.round(decimals))
        .sort_values(ascending=asc)
    )

    return diff_ranked.index[range(0, abs(no_of_adjustments))]
