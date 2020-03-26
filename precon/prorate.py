# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:58:49 2020

@author: edmunm
"""

def prorate(df, factor, exclusions=None):
    """
    Prorates all columns in the DataFrame by the given factor.
    
    Parameters
    ----------
    df : DataFrame
        The values to be prorated.
    factor : float
        The prorating factor to multiply by, between 0 and 1.
    exclusions : iterable
        A list or similar of columns to be excluded from prorating.
    
    Returns
    -------
    DataFrame
        The prorated values.
    """
    if exclusions:
        prorate_cols = ~df.columns.isin(exclusions)
        df.loc[:, prorate_cols] = df.loc[:, prorate_cols].mul(factor, axis=0)
        return df
    else:
        return df.mul(factor, axis=0)