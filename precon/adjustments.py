# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:39:08 2020

@author: edmunm
"""
import pandas as pd

def jan_adjustment(indices, direction='forward'):
    """Adjust the January values of the index."""
    if (direction != 'forward') & (direction != 'back'):
        raise ValueError("'direction' must be either 'forward' or 'back'")
        
    jans = (indices.index.month == 1)
    first_year = (indices.index.year == indices.index[0].year)
    
    jan_values = indices[jans & ~first_year]
    dec_values = indices[indices.index.month == 12]
    # Divide Jan values by Dec values. Dec values t-shifted to match
    # the time series. Results in the need to drop NaN at end
    if direction == 'forward':
        adjusted = jan_values/dec_values.tshift(1, freq='MS') * 100
    elif direction == 'back':
        adjusted = jan_values*dec_values.tshift(1, freq='MS') / 100
    
    adjusted = adjusted.dropna()
    
    # Replace Jan values in original DataFrame or Series of indices
    # with adjustment.
    adjusted_indices = indices.copy()  
    if isinstance(indices, pd.core.frame.DataFrame):
        adjusted_indices.loc[jans & ~first_year, :] = adjusted
    else:
        adjusted_indices.loc[jans & ~first_year] = adjusted
        
    return adjusted_indices


def round_and_adjust_weights(weights, decimals):
    """Rounds a set of weights ensuring the rounded values sum to
    the same total as the unrounded weights.
    
    Parameters
    ----------
    weights : DataFrame or Series
        The weights to round and adjust
    decimals : int
        The decimal points to round to
        
    Returns
    -------
    DataFrame or Series
        The rounded and adjusted weights
    """
    # Get the rounding factor and adjustment
    rounding_factor = 10**decimals
    adjustment = 0.5 / rounding_factor
    
    if isinstance(weights, pd.core.series.Series):
        
        adjustments = _get_series_adjustments(
            weights, dec, rounding_factor, adjustment,
        )
        
    elif isinstance(weights, pd.core.frame.DataFrame):
        
        # Create a zeros DataFrame to fill with adjustments
        adjustments = pd.DataFrame().reindex_like(weights).fillna(0)
        
        for index, row in weights.iterrows():
            adjustments.loc[index] = _get_series_adjustments(
                row, decimals, rounding_factor, adjustment,
            )   
            
    adjusted_weights = weights + adjustments
    return adjusted_weights.round(decimals)


def _get_series_adjustments(weights, dec, factor, adjustment):
    """Get the adjustments for the weight series."""
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
    """Return the weights closest to the rounding decision point to
    minimise differences from the adjustment.
    """
    # Get the difference of each value from its rounded value and
    # rank depending whether adjusting down or up
    if np.sign(no_of_adjustments) == -1:
        asc = True
    else:
        asc = False

    diff_ranked = (weights - weights.round(dec)).sort_values(ascending=asc)

    return diff_ranked.index[range(0, abs(no_of_adjustments))]
