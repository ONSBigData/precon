"""
A set of functions for imputation methods.

# TODO: Modify the functions to work with a user defined period.
    * Look at using pd.Grouper with freq argument
"""
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from precon._validation import _handle_axis, _list_convert
from precon.index_methods import calculate_index
from precon.helpers import (
    axis_slice,
    axis_vals_as_frame,
    flip,
    subset_shared_axis,
)
from precon.weights import reindex_weights_to_indices


def impute_base_prices(
        prices: pd.DataFrame,
        to_impute: pd.DataFrame,
        index_method: str,
        shift_imputed_values: bool = True,
        base_period: int = 1,
        axis: pd._typing.Axis = 1,
        weights: Optional[pd.DataFrame] = None,
        # TODO: make index_method an Enum
        adjustments: Optional[pd.DataFrame] = None,
        ) -> pd.DataFrame:
    """
    Imputes base prices where specified by 'to_impute' for given prices.

    Supports:

        * selecting index method for the imputation index
        * shifting imputated values to the period ahead
        * quality adjustment

    Parameters
    ----------
    prices: DataFrame
        Prices to get imputed base prices for.
    to_impute: DataFrame
        A boolean mask of where to impute.
    index_method: {'jevons', 'dutot', 'carli', 'laspeyres', 'geometric_laspeyres'}
        Method to calculate the index to impute with.
    shift_imputed_values: bool, defaults to True
        True if imputed values are shifted onto the following period.
    base_period: int, defaults to 1
        Base period to select initial base prices from.
    axis : {0 or 'index', 1 or 'columns'}, defaults to 0
        The axis that holds the time series values.
    weights: DataFrame, optional
        Weights needed for laspeyres and geometric laspeyres index
        methods.
    adjustments: DataFrame, optional
        Adjustment values to apply to prices for quality adjustment. If
        there is no adjustment for a price then the adjustment value
        should be zero.

    Returns
    -------
    DataFrame
        The imputed base prices.
    """
    axis = _handle_axis(axis)

    # Subset the metadata axis to match those of indices, for quicker
    # handling of function when applied by groupby.
    to_impute = subset_shared_axis(to_impute, prices, flip(axis))
    if weights is not None:
        weights = subset_shared_axis(weights, prices, flip(axis))
    if adjustments is not None:
        adjustments = subset_shared_axis(adjustments, prices, flip(axis))

    # Ensure the weights are in the same shape as the prices and
    # exclude the prices to impute from the imputation index
    # calculation by setting weights to zero.
    if weights is not None:
        weights = reindex_weights_to_indices(weights, prices, axis=axis)
        weights = weights.mask(to_impute, 0)

    # Get the base prices to start with from given base period.
    start_prices = get_base_prices(
        prices,
        base_period,
        axis=axis,
        fill_shift=False,
    )
    base_prices = start_prices.copy()

    if not shift_imputed_values:
        # Shifting because base prices need to apply to the
        # following base period. Shifts later on after base prices have
        # been imputed if shift_imputed_values is True.
        base_prices = base_prices.shift(1, axis=axis)

    # Apply quality adjustment if adjustments are given.
    if adjustments is not None:
        base_prices_filled = base_prices.ffill(axis)
        to_adjust = adjustments.ne(0)

        base_prices[to_adjust] = get_quality_adjusted_prices(
            prices,
            base_prices_filled,
            adjustments,
            axis,
        )

    # Repeat base price imputation method n times where n is the
    # max number of imputations needed in a year. This is because the
    # index needs the newly imputed values at each new period in order
    # to impute correctly.
    # TODO: Does it though? Test whether this loop can be removed
    times_to_impute = get_annual_max_count(to_impute, flip(axis))
    for _ in range(times_to_impute):

        base_prices_filled = base_prices.ffill(axis)

        # If no weights, set base_prices where imputation occurs to NA
        # to get the index excluding those values for imputing
        if weights is None:
            base_prices_filled[to_impute] = np.nan

        # Get imputed base prices by dividing the prices by the index
        # excluding values to impute
        index = calculate_index(
            prices,
            base_prices_filled,
            weights=weights,
            method=index_method,
            axis=flip(axis),
        )

        imputed_values = prices.div(index, axis) * 100
        base_prices = base_prices.mask(to_impute, imputed_values)

    # Groupby year prevents discontinued prices filling beyond the year
    # that they are discontinued
    # TODO: Get this to work for user defined freq
    # TODO: Check this doesn't fail for central collection
    base_prices = (
        base_prices.groupby(lambda x: x.year, axis=axis)
        .fillna(method='ffill', axis=axis)
    )

    if shift_imputed_values:
        # Shift the base prices one period ahead so the price in the
        # next base period uses the previous base price for calculating
        # the index value. Also shifts imputed base prices here.
        base_prices = base_prices.shift(1, axis=axis)

    # Back fill the first Jan
    return base_prices.fillna(start_prices)


def get_base_prices(
    prices: pd.DataFrame,
    base_period: Union[int, Sequence[int]] = 1,
    axis: pd._typing.Axis = 0,
    fill_shift: bool = True,
) -> pd.DataFrame:
    """Return base prices with optional fill and shift.

    Parameters
    ----------
    prices : DataFrame
    base_period : int, or list of ints, defaults to 1
        Base period/s to select base prices from.
    axis : {0, 1} int, defaults to 0
        Fill and shift direction.
    fill_shift : bool, defaults to True
        Switch to forward fill base prices within year and shift one
        period.

    Returns
    -------
    DataFrame
        The base prices.

    Notes
    -----
    When using fill_shift, the base prices are forward filled within
    each year so that base prices are not filled when prices have
    stopped being collected. Base prices are also shifted by one period,
    so for a base period of Jan (int=1) base prices are shifted on to
    the Feb-Jan+1 time delta in which they apply.

    """
    base_period = _list_convert(base_period)

    # Only prices in the base periods are not NaN.
    months = axis_vals_as_frame(prices, axis, converter=lambda x: x.month)
    base_prices = prices.where(months.isin(base_period))

    # Ensure the prices in the first period are taken as base prices
    # even if not a period given by base_period parameter.
    first_period = axis_slice(0, axis)

    if not base_prices.iloc[first_period].isna().all():
        base_prices.iloc[first_period] = prices.iloc[first_period]

    if fill_shift:
        return ffill_shift(base_prices, axis)
    else:
        return base_prices


def ffill_shift(
    base_prices: pd.DataFrame,
    axis: int = 0
) -> pd.DataFrame:
    """Fill forward base prices within year and shift one period.

    Parameters
    ----------
    base_prices : DataFrame
    axis : {0, 1} int, defaults to 0
        Fill and shift direction.

    Returns
    -------
    DataFrame
        The filled and shifted base prices.

    Notes
    -----
    The base prices are forward filled within each year so that base
    prices are not filled when prices have stopped being collected.
    The base prices are shifted by one period on to the Feb-Jan+1
    time delta in which they apply. A base price is needed for the Jan
    period at the start of the series, so the function fills the
    shifted values with the unshifted values to achieve this.

    """
    return (
        base_prices.groupby(lambda x: x.year, axis=axis)
        .fillna(method='ffill')
        .shift(1, axis=axis)
        .fillna(base_prices)
    )


def get_quality_adjusted_prices(
        prices: pd.DataFrame,
        base_prices: pd.DataFrame,
        adjustments: pd.DataFrame,
        axis: pd._typing.Axis = 1,
        ) -> pd.DataFrame:
    """Applies the quality adjustments to get new base prices."""
    adjustment_factor = prices.div(prices - adjustments)

    return base_prices * adjustment_factor.cumprod(axis)


def get_annual_max_count(
        df: pd.DataFrame,
        axis: pd._typing.Axis,
        ) -> int:
    """Counts values present in each year for df, returns max."""
    # TODO: Change this to work with user defined freq
    return int(df.any(axis).groupby(lambda x: x.year).sum().max())
