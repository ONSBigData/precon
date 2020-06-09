""" prorate.py

This module contains a function for prorating.
"""

def prorate(df, factor, axis=0):
    """
    Prorates all columns in the DataFrame by the given factor.
    
    Parameters
    ----------
    df : DataFrame
        The values to be prorated.
    factor : float
        The prorating factor to multiply by, between 0 and 1.
    axis : int {0, 1}, default 0
        Axis along which the function is applied.
    
    Returns
    -------
    DataFrame
        The prorated values.
    """
    return df.mul(factor, axis=axis)