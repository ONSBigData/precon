# -*- coding: utf-8 -*-

import pandas as pd

def set_ref_period(df, period):
    """ A function to re-reference an index series on a given period."""
    base_mean = df[period].mean() # Mean of values at the base period.
    reref = df.div(base_mean) * 100 # Normalise
    
    # Fill NaNs from division with zeros
    reref.fillna(0, inplace=True)
       
    return reref


def set_index_range(df, start=None, end=None):
    """Edit this function to take a start and an end year."""   
    if start:
        if start not in df.index.year.unique().astype(str).to_numpy():
            raise Exception("start needs to be a year in the index, as a string.")
    
    subset = df.loc[start:end]
    
    if type(df) == type(pd.Series()):
        subset.iloc[0] = 100  
    else:
        subset.iloc[0, :] = 100
    return subset


def prices_to_index(prices): #TODO add base
    """Returns an index from a DataFrame or Series of aggregated prices."""
    if isinstance(prices, pd.Series):
        return prices/prices.iloc[0] * 100
    else:
        return prices.div(prices.iloc[0, :]) * 100


def full_index_to_in_year_indices(full_index):
    """Break index down into Jan-Jan+1 segments, rebased at 100 each year.
    Returns a dictionary of the in-year indices with years as keys.
    """
    # Get a list of the years present in the index
    index_years = full_index.resample('A').first().index.to_timestamp()

    # Set the index range for each year (base=100, runs through to Jan+1)
    pi_yearly = {}
    for year in index_years:
        end = year + pd.DateOffset(years=1)
        pi_yearly[year.year] = set_index_range(full_index, start=year, end=end)
    
    return pi_yearly


def in_year_indices_to_full_index(in_year_indices):
    """Converts a dictionary of in-year indices into a single 
    unchained index.
    """
    full_index = pd.concat(in_year_indices).fillna(0).droplevel(0)
    
    # Take out any Jan=100 that are not the first in the full index
    condition = ((full_index.index.month == 1)
                 & (full_index.index.year != full_index.index[0].year)
                 & (full_index == 100).any(axis=1))
    full_index = full_index[~condition]
    
    return full_index
