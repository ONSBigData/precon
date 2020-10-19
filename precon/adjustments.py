"""
Function for the jan_adjustment.
"""
import pandas as pd


def jan_adjustment(indices, direction='forward'):
    """Adjust the January values of the index."""
    if direction not in ['forward', 'back']:
        raise ValueError("'direction' must be either 'forward' or 'back'")

    jans = (indices.index.month == 1)
    first_year = (indices.index.year == indices.index[0].year)

    jan_values = indices[jans & ~first_year]
    dec_values = indices[indices.index.month == 12]
    # Divide Jan values by Dec values. Dec values t-shifted to match
    # the time series. Results in the need to drop NaN at end
    if direction == 'forward':
        adjusted = jan_values / dec_values.tshift(1, freq='MS') * 100
    elif direction == 'back':
        adjusted = jan_values * dec_values.tshift(1, freq='MS') / 100

    adjusted = adjusted.dropna()

    # Replace Jan values in original DataFrame or Series of indices
    # with adjustment.
    adjusted_indices = indices.copy()
    if isinstance(indices, pd.core.frame.DataFrame):
        adjusted_indices.loc[jans & ~first_year, :] = adjusted
    else:
        adjusted_indices.loc[jans & ~first_year] = adjusted

    return adjusted_indices
