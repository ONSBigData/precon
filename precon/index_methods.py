"""
A set of index methods and index helper functions.
"""
import numpy as np
import pandas as pd

from precon.aggregation import aggregate


def calculate_index(
        prices: pd.DataFrame,
        base_prices: pd.DataFrame,
        weights: pd.DataFrame = None,
        method: str = None,
        axis: int = 1,
        ) -> pd.Dataframe:
    """Calculates the index according to weights or methods parameters
    using given prices and base_prices.
    """        
    indices = prices.div(base_prices) * 100
    
    if weights is not None:
        return aggregate(indices, weights, axis)
    
    elif method == "dutot":
        return prices.mean(axis).div(base_prices.mean(axis)) * 100
    
    elif method == "carli":
        return indices.mean(axis)
    
    elif method == "jevons":
        return geo_mean(indices, axis=axis)
    

def geo_mean(indices: pd.Dataframe, axis: int = 1) -> pd.DataFrame:
    """Calculates the geometric mean, accounting for missing values."""
    if isinstance(indices, pd.DataFrame):
        return np.exp(np.log(indices.prod(axis)) / indices.notna().sum(axis))
    else:
        return np.exp(np.log(indices.prod()) / indices.notna().sum())