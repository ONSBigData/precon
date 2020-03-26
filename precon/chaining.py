# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:49:18 2020

@author: Mitchell Edmunds
@title: Chaining functions
"""
import pandas as pd

def chain(indices, double_link=False, base_months=None):
    """
    Chain the indices using direct (fixed-base) chaining.
    
    By default, chains the indices using the direct (fixed-base) 
    chaining with a single annual chainlink. Double annual chainlinking
    can be implemented using the `double_link` argument. Alternatively, 
    the user can override and specify their own base months.
    
    Parameters
    ----------
    indices : Series or DataFrame
        The unchained indices to be chained.
    double_link : bool, default 'False', meaning single chainlink
        Boolean switch for annual double chainlinking.
    base_months : list of int, optional
        The base months for chainlinking.
        
    Returns
    -------
    Series or DataFrame
        The chained indices.
    
    See Also
    --------
    unchain: Unchain the indices using direct (fixed-base) chaining.
    """
    if not base_months:
        base_months = [1]
        if double_link:
            base_months.append(12) 
    
    base = indices.copy()
    base[base.index.month.isin(base_months)] = 100
    base = base.shift().bfill()    # Fills first month
    
    growth = indices / base
    chained_indices = growth.cumprod() * 100
    
    return chained_indices.fillna(0)    # Account for zero division



def unchain(indices, double_link=False, base_months=None):
    """
    Unchain the indices using direct (fixed-base) chaining.
    
    By default, unchains the indices using the direct (fixed-base) 
    chaining with a single annual chainlink. Double annual chainlinking
    can be implemented using the `double_link` argument. Alternatively, 
    the user can override and specify their own base months.
    
    Parameters
    ----------
    indices : Series or DataFrame
        The chained indices to be unchained.
    double_link : bool, default 'False', meaning single chainlink
        Boolean switch for annual double chainlinking.
    base_months : list of int, optional
        The base months for chainlinking.
        
    Returns
    -------
    Series or DataFrame
        The unchained indices.
    
    See Also
    --------
    chain: Chain the indices using direct (fixed-base) chaining.
    """
    if not base_months:
        base_months = [1]
        if double_link:
            base_months.append(12) 
    
    # Create a DataFrame of the base values with same index
    base = pd.DataFrame().reindex_like(indices)
    mask = base.index.month.isin(base_months)
    base.loc[mask] = indices[mask]
    base = base.shift().ffill().bfill()
    
    unchained_indices = indices/base * 100
    
    return unchained_indices.fillna(0)  #   Account for zero division
