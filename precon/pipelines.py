# -*- coding: utf-8 -*-

from precon.imputation import impute_base_prices, get_base_prices
from precon.index_methods import calculate_index
from precon.helpers import flip

def apply_index_calculations(
        prices,
        base_period=1,
        axis=1,
        shift_imputed_values=False,
        to_impute=None,
        weights=None,
        index_method=None,
        adjustments=None,
        exclusions=None,
        ):
    """Returns the final index given the optional parameters.
    
    Parameters
    ----------
    markers: A dataframe of non-comparable markers.
    weights: A dataframe of weights to aggregate with prices in index
        calculation.
    method: {'dutot', 'carli', 'jevons'}
        The index method to apply if not aggregating with weights.
    adjustments: A dataframe of quality adjustments to apply.
    """
    if exclusions is not None:
        # Set exclusions weights to zero so they are not included in
        # the final index calculation
        weights = weights.mask(exclusions, 0)
    
    # Impute the base prices if necessary, if not just take the prices
    # in the base period and fill forward
    if to_impute is not None:
        base_prices = impute_base_prices(
            prices,
            to_impute,
            shift_imputed_values=shift_imputed_values,
            base_period=base_period,
            axis=axis, 
            weights=weights,
            index_method=index_method,
            adjustments=adjustments,
        )
    else:
        base_prices = get_base_prices(prices, base_period, axis, ffill=True)
        # Shift is necessary because the price in the following base
        # period uses the previous base period to calculate the index.
        base_prices = base_prices.shift(1, axis=axis)
    
    return calculate_index(
        prices,
        base_prices,
        weights=weights,
        method=index_method,
        axis=flip(axis),
    )