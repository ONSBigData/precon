"""
Functions for special rounding methods. Includes function to round
and adjust weights to keep the sum of weights the same.
"""

import numpy as np
import pandas as pd

from .helpers import axis_slice

def round_and_adjust_weights(weights, decimals, axis):
    """Rounds a set of weights ensuring the rounded values sum to
    the same total as the unrounded weights.
    
    Parameters
    ----------
    weights : DataFrame or Series
        The weights to be rounded.
    decimals : int
        Number of decimal places to round each column to.
        
    Returns
    -------
    DataFrame or Series
        The rounded and adjusted weights.
    """
    iter_dict = {
        0: pd.DataFrame.iterrows,
        1: pd.DataFrame.iteritems,
    }
    
    iter_method = iter_dict.get(axis)
    
    # Get the rounding factor and adjustment value
    rounding_factor = 10**decimals
    adjustment = 0.5 / rounding_factor
    
    if isinstance(weights, pd.core.series.Series):
        
        adjustments = _get_series_adjustments(
            weights, decimals, rounding_factor, adjustment,
        )
        
    elif isinstance(weights, pd.core.frame.DataFrame):
        
        # Create a zeros DataFrame to fill with adjustments
        adjustments = pd.DataFrame().reindex_like(weights).fillna(0)
        
        for index, row in iter_method(weights):
           # Create a selector based on the axis
           slice_ = axis_slice(index, axis)
           
           adjustments.loc[slice_] = _get_series_adjustments(
               row, decimals, rounding_factor, adjustment,
           )
            
    adjusted_weights = weights + adjustments
    return adjusted_weights.round(decimals)
          

def _get_series_adjustments(weights, dec, factor, adjustment):
    """Return a Series of weight adjustments to make"""
    # Errors > 0.5 between rounded and unrounded means that adjustment
    # is needed
    errs = (weights - weights.round(dec)).sum()
    no_of_adjustments = int((errs.round(dec) * factor))
    
    # Create a zeros Series to fill with adjustments
    adjustments = pd.Series(dtype=float).reindex_like(weights).fillna(0)

    to_adjust = _get_weights_to_adjust(weights, dec, no_of_adjustments)
    adjustments.loc[to_adjust] = adjustment * np.sign(no_of_adjustments)
    
    return adjustments
    

def _get_weights_to_adjust(weights, dec, no_of_adjustments):
    """Get the difference of each value from its rounded value and pick
    weights to round by rank depending whether adjusting down or up.
    """
    asc = True if np.sign(no_of_adjustments) == -1 else False

    diff_ranked = (weights - weights.round(dec)).sort_values(ascending=asc)

    return diff_ranked.index[range(0, abs(no_of_adjustments))]

