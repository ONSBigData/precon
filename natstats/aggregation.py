"""
Common aggregation functions.    
"""
from typing import Union

import numpy as np
import pandas as pd

from natstats.weights import get_weight_shares, reindex_weights_to_indices
from natstats.helpers import reduce_cols, flip
from natstats._validation import _handle_axis

PandasObj = Union[pd.DataFrame, pd.Series]
Axis = Union[int, str]

def aggregate(
        indices: pd.DataFrame,
        weights: Union[pd.DataFrame, pd.Series],
        method: str = 'mean',
        axis: Axis = 1,
        ) -> pd.Series:
    """
    Aggregate unchained indices with weights to get sum product.
    
    Parameters
    ----------
    axis : {0 or ‘index’, 1 or ‘columns’}, default 1
        Axis along which the function is applied:
            * 0 or ‘index’: apply function to each column.
            * 1 or ‘columns’: apply function to each row.
    """
    axis = _handle_axis(axis)    
    
    methods_lib = {
        'mean': _mean_aggregate,
        'geomean': _geo_mean_aggregate,
    }
    agg_method = methods_lib.get(method)
    
    # Make sure that the indices and weights have the same time series
    # axis before aggregating
    if isinstance(weights, pd.Series):
        weights = weights.to_frame()
        if axis == 1:
            weights = weights.T

    weights = reindex_weights_to_indices(weights, indices, flip(axis))
    
    # Ensure zero or NA indices have zero weight
    weights = weights.mask(indices.isna() | indices.eq(0), 0)
    
    weight_shares = get_weight_shares(weights, axis)
    return agg_method(indices, weight_shares, axis)


def _mean_aggregate(indices, weight_shares, axis):
    """Aggregates indices and weight shares using sum product."""
    return indices.mul(weight_shares).sum(axis=axis, min_count=1)
  
    
def _geo_mean_aggregate(indices, weight_shares, axis):
    """Aggregates indices and weight shares using geo mean method."""
    return (
        np.exp(
            np.log(indices).mul(weight_shares)
            .sum(axis=axis, min_count=1)
        )
    )
