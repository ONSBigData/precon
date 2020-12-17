"""A set of common pipeline functions to create National Statistics."""
from typing import Optional, Union, Sequence

import pandas as pd
from pandas._typing import Axis

from precon._validation import _handle_axis
from precon.base_prices import (
    impute_base_prices,
    get_base_prices,
    ffill_shift,
)
from precon.index_methods import calculate_index
from precon.helpers import flip


def index_calculator(
    prices: pd.DataFrame,
    index_method: str,
    shift_imputed_values: bool = False,
    to_impute: Optional[pd.DataFrame] = None,
    weights: Optional[pd.DataFrame] = None,
    adjustments: Optional[pd.DataFrame] = None,
    exclusions: Optional[pd.DataFrame] = None,
    base_prices: Optional[pd.DataFrame] = None,
    base_period: Union[int, Sequence[int]] = 1,
    axis: Axis = 1,
) -> pd.Series:
    """Calculates an index given prices and an index method, with
    optional arguments for base price imputation.

    Parameters
    ----------
    prices : DataFrame
        The prices with which to calculate the index.
    method : {'jevons', 'dutot', 'carli', 'laspeyres', 'geometric_laspeyres'}
        Method to calculate the index.
    shift_imputed_values: bool, defaults to True
        True if imputed values are shifted onto the following period.
    to_impute : DataFrame, optional
        A boolean mask of where to impute.
    weights : DataFrame, optional
        The weights to use if the index method requires it.
    adjustments : DataFrame, optional
        Adjustment factors to multiply by base prices for quality
        adjustments. A factor of 1 means no adjustment takes place.
    exclusions : DataFrame, optional
        A boolean mask of prices to exclude from the final index
        calculation.
    base_prices : DataFrame, optional
        Base prices to be forward filled and shifted before used in
        index calculation.
    base_period : int, or list of ints, defaults to 1
        Base period/s to select initial base prices from.
    axis : {0 or 'index', 1 or 'columns'}, defaults to 0
        The axis that holds the time series values.

    Returns
    -------
    Series
        The index.

    """
    axis = _handle_axis(axis)

    if exclusions is not None:
        # Set exclusions weights to zero so they are not included in
        # the final index calculation
        weights = weights.mask(exclusions, 0)

    # Impute the base prices if necessary.
    if to_impute is not None:
        base_prices = impute_base_prices(
            prices,
            to_impute,
            index_method=index_method,
            shift_imputed_values=shift_imputed_values,
            base_period=base_period,
            axis=axis,
            weights=weights,
            adjustments=adjustments,
        )
    else:
        if base_prices is not None:
            base_prices = ffill_shift(base_prices, axis)
        else:
            # Get base prices from prices.
            base_prices = get_base_prices(prices, base_period, axis)

        if adjustments is not None:
            base_prices *= adjustments

    return calculate_index(
        prices,
        base_prices,
        weights=weights,
        method=index_method,
        axis=flip(axis),
    )
