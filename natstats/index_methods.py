"""
A set of index methods and index helper functions.
"""
from typing import Optional

import numpy as np
import pandas as pd

from natstats._validation import _handle_axis
from natstats.aggregation import aggregate


def calculate_index(
        prices: pd.DataFrame,
        base_prices: pd.DataFrame,
        method: str,
        axis: pd._typing.Axis = 1,
        weights: Optional[pd.DataFrame] = None,
        ) -> pd.Series:
    """Calculates the index with the given method.

    Parameters
    ----------
    prices: DataFrame
        The prices with which to calculate the index.
    base_prices: DataFrame
        The pre-calculated base prices for the index calculation.
    method: {'jevons', 'dutot', 'carli', 'laspeyres', 'geometric_laspeyres'}, str or callable, defaults to 'jevons'
        Method to calculate the index.
    axis : {0 or 'index', 1 or 'columns'}, defaults to 0
        The axis that holds the time series values.
    weights: DataFrame, optional
        The weights to use if the index method requires it.
    
    Returns
    -------
    Series
        The index.
    """
    axis = _handle_axis(axis)

    index_method_mapper = {
        'jevons': jevons_index,
        'carli': carli_index,
        'dutot': dutot_index,
        'laspeyres': laspeyres_index,
        'geometric_laspeyres': geometric_laspeyres_index,
    }
    index_method = index_method_mapper.get(method)
    
    if method in ['laspeyres', 'geometric_laspeyres']:
        indices = index_method(prices, base_prices, weights, axis) 
    else:
        indices = index_method(prices, base_prices, axis)
        
    return indices * 100
    

def jevons_index(prices, base_prices, axis):
    """Calculates an index using the Jevons method which takes the
    geometric mean of price relatives.
    """
    price_relatives = prices.div(base_prices)
    return geo_mean(price_relatives, axis)


def carli_index(prices, base_prices, axis):
    """Calculates an index using the Carli method which takes the mean
    of price relatives.
    """
    price_relatives = prices.div(base_prices)
    return price_relatives.mean(axis)


def dutot_index(prices, base_prices, axis):
    """Calculates an index using the Dutot method which divides the
    mean of the prices by the mean of the base prices.
    """
    return prices.mean(axis).div(base_prices.mean(axis))


def laspeyres_index(prices, base_prices, weights, axis):
    """Calculates an index using the Laspeyres method which takes a
    sum of the product of the price relatives and weight shares.
    """
    price_relatives = prices.div(base_prices)
    return aggregate(price_relatives, weights, axis=axis)


def geometric_laspeyres_index(prices, base_prices, weights, axis):
    """Calculates an index using the geometric Laspeyres method which
    takes the geometric mean of the price relatives multiplied by weight
    shares.
    """
    price_relatives = prices.div(base_prices)
    return aggregate(price_relatives, weights, method='geomean', axis=axis)


def geo_mean(indices: pd.DataFrame, axis: int = 1) -> pd.DataFrame:
    """Calculates the geometric mean, accounting for missing values."""
    if isinstance(indices, pd.DataFrame):
        return np.exp(np.log(indices.prod(axis)) / indices.notna().sum(axis))
    else:
        return np.exp(np.log(indices.prod()) / indices.notna().sum())
