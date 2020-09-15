"""
A set of functions for imputation methods.
"""
import numpy as np
import pandas as pd

from precon.index_methods import calculate_index

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


def get_annual_max_count(df, axis):
    """Counts values present in each year for df, returns max."""
    return int(df.any(axis).replace(True, 1).resample('A').sum().max())
