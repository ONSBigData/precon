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


#def adjust_weights(weights):
#    # Errors > 0.5 means that adjustment is needed
#    errs = abs((weights - weights.round()).sum(1))
#    need_adjusting = errs > 0.5
#    no_of_adjustments = errs[need_adjusting].round(1)
#    
#    adjustments = pd.DataFrame().reindex_like(weights[need_adjusting])
#    for index, row in weights[need_adjusting].iterrows():
#        for _ in range(1, abs(number_of_adjustments[index])):
#    # NEEDS DEV WORK