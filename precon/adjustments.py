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


def round_and_adjust_weights(weights, dec):
    """Rounds a set of weights while ensuring the rounded values sum to
    the same total as the unrounded weights.
    """
    # Get the rounded weights and rounding factor
    rounded_weights = weights.round(dec)
    rounding_factor = 10**dec
    
    # Errors > 0.5 between rounded and unrounded means that adjustment
    # is needed
    errs = (weights - rounded_weights).sum(1)
    need_adjusting = abs(errs) > (0.5 / rounding_factor)
    no_of_adjustments = (errs[need_adjusting].round(dec) * rounding_factor)
    no_of_adjustments = no_of_adjustments.astype(int)
    
    # Create a zeros DataFrame to fill with adjustments
    adjustments = pd.DataFrame().reindex_like(weights).fillna(0)
    
    for index, row in weights[need_adjusting].iterrows():
        
        # Get number of adjustments for this row
        adjust = no_of_adjustments.loc[index]
        
        # Get the difference of each value from its rounded value and
        # rank depending whether adjusting down or up
        if np.sign(adjust) == -1:
            diff_ranked = (row - row.round(dec)).sort_values(ascending=True)
        else:
            diff_ranked = (row - row.round(dec)).sort_values(ascending=False)
        
        # For each adjusment to be made, set the appropriate value
        # in the adjustments df        
        for i in range(0, abs(adjust)):
            adjustments.at[index, diff_ranked.index[i]] = (
                (0.5 / rounding_factor) * np.sign(adjust)
            )
            
    adjusted_weights = weights + adjustments
    return adjusted_weights.round(dec)