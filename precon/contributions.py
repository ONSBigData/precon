"""
Functions to calculate contributions to growth.
"""

import pandas as pd

from precon.helpers import _selector
from precon.weights import get_weight_shares, reindex_weights_to_indices


def contributions(components, weights, index, double_update=False):
    """
    Calculates the growth contributions of the index components to
    overall growth of a double annual chainlinked index.

    Parameters
    ----------
    components : DataFrame
        The index components.
    weights : DataFrame
        The weights for the index components.
    index : Series
        The unchained aggregated index.

    Returns
    -------
    DataFrame
        The contributions of each component to overall index growth.
    """
    weights = get_weight_shares(weights)
    weights = reindex_weights_to_indices(weights, components)

    # Set equation components

    # Set the January values to 100 for the unchained index
    unchained_index = index.copy()
    unchained_index[unchained_index.index.month == 1] = 100

    # Set components as Ic_y and set Jan = 100
    Ic_y = components.copy()
    Ic_y[Ic_y.index.month == 1] = 100

    # Shift weights to start in Jan rather than Feb
    # _py suffix denotes "previous year" or t-12
    if not double_update:
        w1 = weights.shift(12)
        w2 = w3 = weights
    else:
        w1 = _select_month_reindex(weights, 2).shift(-1).ffill().shift(12)
        w2 = _select_month_reindex(weights, 1).ffill()
        w3 = _select_month_reindex(weights, 2).shift(-1).ffill()

    # Take Jan values from previous Dec=100 index (without Jans set to 100)
    # Dec values can be taken from either, previous year so shift by 12
    Ic_dec = _select_month_reindex(components, 12).bfill().shift(12)
    Ic_jan = _select_month_reindex(components, 1).ffill()
    Ic_py = Ic_y.shift(12)

    IA_dec = _select_month_reindex(index, 12).bfill().shift(12)
    IA_jan = _select_month_reindex(index, 1).ffill()
    IA_py = unchained_index.shift(12)

    # Calculate contributions

    # Calculate each component of the contributions to annual change equation
    # for double-linked index. Zeros are positional args i.e. axis=0
    comp1 = w1.mul((Ic_dec - Ic_py).div(IA_py, 0), 0) * 100
    comp2 = (
        w2.mul((Ic_jan - 100).div(IA_py, 0), 0)
        .mul(IA_dec, 0)
    )
    comp3 = (
        w3.mul((Ic_y - 100).div(IA_py, 0), 0)
        .mul(IA_jan / 100, 0)
        .mul(IA_dec, 0)
    )

    contributions = comp1 + comp2 + comp3

    return contributions.dropna()


def contributions_with_double_update(
        components, weights, index, start_year='2017',
        ):
    """Returns the contributions for time periods that include both
    single update and double update.

    Parameters
    ----------
    components : DataFrame
        The index components.
    weights : DataFrame
        The weights for the index components.
    index : Series
        The unchained aggregated index.
    start_year : str
        The year the double weights update began.

    Returns
    -------
    DataFrame
        The contributions of each component to overall index growth.
    """
    end_year = str(int(start_year) - 1)
    # Slice up to end_year, and from end_year onwards since contributions
    # start the year after the first year (12 month lag)
    pre_slice = slice(end_year)
    post_slice = slice(end_year, None)

    pre_args = _selector(pre_slice, components, weights, index)
    post_args = _selector(post_slice, components, weights, index)

    # Calculate contributions pre double-update, and post double-update
    # Then concatenate them together
    contributions_pre = contributions(*pre_args)
    contributions_post = contributions(*post_args, double_update=True)
    return pd.concat([contributions_pre, contributions_post])


def _select_month_reindex(indices, month):
    """Subsets indices for given month then reindexes to original size."""
    select_month = indices[indices.index.month == month]
    return select_month.reindex(index=indices.index)
