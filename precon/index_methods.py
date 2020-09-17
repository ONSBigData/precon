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
        ) -> pd.DataFrame:
    """Calculates the index according to weights or methods parameters
    using given prices and base_prices.
    """        
    price_relatives = prices.div(base_prices)

    if weights is not None:
        
        if method == "jevons":
            indices = aggregate(price_relatives, weights, 'geomean', axis)
        
        else:
            # "laspeyres"?
            indices = aggregate(price_relatives, weights, axis)

    elif method == "dutot":
        indices = prices.mean(axis).div(base_prices.mean(axis))

    elif method == "carli":
        indices = price_relatives.mean(axis)

    elif method == "jevons":
        indices = geo_mean(price_relatives, axis=axis)
        
    return indices * 100
    

def geo_mean(indices: pd.DataFrame, axis: int = 1) -> pd.DataFrame:
    """Calculates the geometric mean, accounting for missing values."""
    if isinstance(indices, pd.DataFrame):
        return np.exp(np.log(indices.prod(axis)) / indices.notna().sum(axis))
    else:
        return np.exp(np.log(indices.prod()) / indices.notna().sum())