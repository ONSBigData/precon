"""
A set of functions for imputation methods.
"""
import numpy as np
import pandas as pd

from precon.index_methods import calculate_index
from precon.helpers import in_year_fill

def base_price_imputation(
        prices,
        markers,
        impute_methods,
        base_month=1,
        axis=1, 
        weights=None,
        index_method=None,
        adjustments=None,
        ):
    """Impute the prices marked non-comparable using the index excl.
    non-comparables.
    """   
    base_prices = get_base_prices(prices, base_month, axis=axis, ffill=False)
        
    # Apply quality adjustment if adjustments arg given
    if adjustments is not None:
        base_prices_filled = base_prices.ffill(axis)
        to_adjust = adjustments.ne(0)

        base_prices[to_adjust] = get_quality_adjusted_prices(
            prices,
            base_prices_filled,
            adjustments,
            axis,
        )

    # Fill forward the base prices but set non-comparables and
    # "T" markers to NA
    # to_impute = non_comparables | (markers == 'T')
    # Get the max times to impute for any year.
    to_impute = markers.isin(impute_methods.keys())
    times_to_impute = get_annual_max_count(to_impute, axis^1)

    for _ in range(times_to_impute):
        for ind, method in impute_methods.items():
            to_impute = markers == ind
            base_prices[to_impute] = method(
                base_prices, to_impute, prices, weights, index_method, axis,
            )
    
    base_prices = in_year_fill(base_prices, axis, method='ffill')

    # Shift the base prices onto Feb-Jan+1 then bfill first month
    return base_prices.shift(1, axis=axis)


def get_base_prices(prices, base_month=1, axis=0, ffill=True):
    """Returns the prices at the base month in the same shape as prices.
    
    Default behaviour is to fill forward values, but can be changed to
    return NaN where not base_month by setting ffill=False.
    """
    not_bases = prices.axes[axis].month != base_month
   
    # Fill all except base months with NA
    base_prices = prices.copy()
    if axis == 0:
       base_prices.iloc[not_bases, :] = np.nan
    elif axis == 1:
        base_prices.iloc[:, not_bases] = np.nan
    
    if ffill:
        return base_prices.ffill(axis)
    else:
        return base_prices


def get_quality_adjusted_prices(prices, base_prices, adjustments, axis=1):
    """Applies the quality adjustments to get new base prices."""    
    adjustment_factor = prices.div(prices-adjustments)

    return base_prices * adjustment_factor.cumprod(axis)


def impute_using_previous_index_value(
        base_prices: pd.DataFrame,
        to_impute: pd.DataFrame,
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

    return prices.div(index.shift(1), axis) * 100


def impute_using_non_comparable_index(
        base_prices: pd.DataFrame,
        to_impute: pd.DataFrame,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
        method: str,
        axis: int,
        ) -> pd.DataFrame:
    """
    Imputes base prices using an index calculated by excluding the
    values to impute

    This method is typically used for items marked as non-comparable
    but works for any values marked in the `to_impute` argument

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
    # Set base prices to impute as NaN so they are excluded from calc
    base_prices_filled = base_prices_filled.mask(to_impute, np.nan)

    # Create an index and use to impute non-comparables
    index_excl_nc = calculate_index(
        prices,
        base_prices_filled,
        weights=weights,
        method=method,
        axis=axis^1,
    )

    return prices.div(index_excl_nc, axis) * 100



def get_annual_max_count(df, axis):
    """Counts values present in each year for df, returns max."""
    return int(df.any(axis).replace(True, 1).resample('A').sum().max())
