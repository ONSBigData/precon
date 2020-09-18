"""
A set of functions for imputation methods.
"""
from typing import Dict, Callable, Optional, Union

import numpy as np
import pandas as pd

from precon.index_methods import calculate_index
from precon.helpers import in_year_fill

ImputationCallable = Callable[
    [pd.DataFrame, pd.DataFrame, pd.DataFrame, str, int], pd.DataFrame
]
ImputationMethods = Union[ImputationCallable, Dict[str, ImputationCallable]]

def base_price_imputation(
        prices: pd.DataFrame,
        imputation_markers: pd.DataFrame,
        impute_method: Optional[ImputationMethods] = None,
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
    # Set the default imputation method if none provided.
    if impute_method is None:
        impute_method = impute_using_index
    
    base_prices = get_base_prices(prices, base_period, axis=axis, ffill=False)

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
    if imputation_markers.dtypes.all() == np.dtype('bool'):
        imputations = imputation_markers
    else:
        # Convert to bool if not already.
        imputations = imputation_markers.notna()

    times_to_impute = get_annual_max_count(imputations, axis^1)
    
    # Repeat the base price imputation for the number of imputations
    # needed, and for each indicator-method combination.
    for _ in range(times_to_impute):
        # If a dict of impute methods is provided, loop through and 
        # apply the method for each response indicator.
        if isinstance(impute_method, dict):
            for indicator, method in impute_method.items():
                to_impute = imputation_markers == indicator
                base_prices[to_impute] = method(
                    base_prices, prices, weights, index_method, axis,
                )
                
        else:
            base_prices[imputations] = impute_method(
                    base_prices, prices, weights, index_method, axis,
            )
    
    # Using in_year_fill prevents discontinued prices filling beyond
    # the year that they are discontinued
    base_prices = in_year_fill(base_prices, axis, method='ffill')

    # Shift the base prices onto Feb-Jan+1
    return base_prices.shift(1, axis=axis)


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


def impute_using_index(
        base_prices: pd.DataFrame,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
        method: str,
        axis: int,
        ) -> pd.DataFrame:
    """
    Imputes base prices using the index value from the previous period

    Parameters
    ==========
    base_prices: Unfilled base prices that only contain values where
        the base price changes

    to_impute: A boolean mask for values to impute

    prices: The price quotes observed

    weights: The weights for each price quote

    method: The index method used

    axis: The axis that is a time series
    """
    base_prices_filled = base_prices.ffill(axis)

    index = calculate_index(
        prices,
        base_prices_filled,
        weights=weights,
        method=method,
        axis=axis^1,
    )

    return prices.div(index, axis) * 100


def get_annual_max_count(df, axis):
    """Counts values present in each year for df, returns max."""
    return int(df.any(axis).replace(True, 1).resample('A').sum().max())
