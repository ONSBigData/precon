"""
A set of functions for imputation methods.
"""
from typing import Optional

import numpy as np
import pandas as pd

from precon.index_methods import calculate_index
from precon.helpers import period_window_fill, flip
from precon.weights import reindex_weights_to_indices


def impute_base_prices(
        prices: pd.DataFrame,
        to_impute: pd.DataFrame,
        shift_imputed_values: bool = True,
        base_period: int = 1,
        axis: int = 1, 
        weights: Optional[pd.DataFrame] = None,
        # TODO: make index_method an Enum
        index_method: str = None,
        adjustments: Optional[pd.DataFrame] = None,
        ) -> pd.DataFrame:
    """
    Selects base prices for the base period and performs base price
    imputation

    This function performs base price imputations depending on the
    parameters given. The `markers` parameter is to pass the function
    a DataFrame of string markers or response indicators that tell the
    function which prices to impute. The `impute_methods` parameter is
    used to specify which imputation function should be used for a given
    marker. The `weights` and `index_method` govern how the index that
    is used for imputation is calculated. The `adjustments` parameter
    is used to optionally pass in quality adjustments.
    """   
    if weights is not None:
        weights = reindex_weights_to_indices(weights, prices, axis=axis)
        weights = weights.mask(to_impute, 0)
    
    # Set the default imputation method if none provided.   
    start_prices = get_base_prices(prices, base_period, axis=axis, ffill=False)
    base_prices = start_prices.copy()
    
    if not shift_imputed_values:
        base_prices = base_prices.shift(1, axis=axis)

    # Apply quality adjustment if adjustments arg given.
    if adjustments is not None:
        base_prices_filled = base_prices.ffill(axis)
        to_adjust = adjustments.ne(0)

        base_prices[to_adjust] = get_quality_adjusted_prices(
            prices,
            base_prices_filled,
            adjustments,
            axis,
        )

    # Get the max times to impute for any year.
    times_to_impute = get_annual_max_count(to_impute, flip(axis))
    
    # Repeat the base price imputation for the number of imputations
    # needed, and for each indicator-method combination.
    for _ in range(times_to_impute):

        base_prices_filled = base_prices.ffill(axis)

        # If no weights, set base_prices where imputation occurs to NA
        # to get the index excluding those values for imputing
        if weights is None:
            base_prices_filled[to_impute] = np.nan

        index = calculate_index(
            prices,
            base_prices_filled,
            weights=weights,
            method=index_method,
            axis=flip(axis),
        )
        
        imputed_values = prices.div(index, axis) * 100
        base_prices = base_prices.mask(to_impute, imputed_values)
    
    # Using period_window_fill prevents discontinued prices filling
    # beyond the year that they are discontinued
    base_prices = period_window_fill(
        base_prices, periods=12, freq='MS',
        axis=axis, method='ffill',
    )
    
    if shift_imputed_values:
        # Shift the base prices onto Feb-Jan+1
        base_prices = base_prices.shift(1, axis=axis)
    
    # Back fill the first Jan
    return base_prices.fillna(start_prices)


def get_base_prices(prices, base_period=1, axis=0, ffill=True):
    """Returns the prices at the base month in the same shape as prices.

    Default behaviour is to fill forward values, but can be changed to
    return NaN where not base_month by setting ffill=False.
    """
    bases = prices.axes[axis].month == base_period

    # Fill all except base months with NA
    base_prices = prices.copy()
    if axis == 0:
       base_prices.iloc[~bases, :] = np.nan
    elif axis == 1:
        base_prices.iloc[:, ~bases] = np.nan
    
    if ffill:
        return base_prices.ffill(axis)
    else:
        return base_prices


def get_quality_adjusted_prices(prices, base_prices, adjustments, axis=1):
    """Applies the quality adjustments to get new base prices."""    
    adjustment_factor = prices.div(prices - adjustments)

    return base_prices * adjustment_factor.cumprod(axis)


def get_annual_max_count(df, axis):
    """Counts values present in each year for df, returns max."""
    return int(df.any(axis).replace(True, 1).resample('A').sum().max())
