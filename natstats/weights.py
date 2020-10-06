# -*- coding: utf-8 -*-

import pandas as pd

from natstats.helpers import reindex_and_fill, flip
from natstats._validation import _handle_axis


def get_weight_shares(weights, axis=1):
    """If not weight shares already, calculates weight shares."""
    axis = _handle_axis(axis)
    
    # TODO: test precision
    if not weights.sum(axis).round(5).eq(1).all():
        return weights.div(weights.sum(axis), axis=flip(axis))
    
    else:   # It is already weight shares so return input
        return weights


def reindex_weights_to_indices(weights, indices, axis=0):
    """If not already indexed like indices, reindexes weights."""
    axis = _handle_axis(axis)
    
    # Convert to a DataFrame is weight is a Series, transpose if needed
    if isinstance(weights, pd.Series):
        weights = weights.to_frame()
        if axis == 0:
            weights = weights.T
            
    if not weights.axes[axis].equals(indices.axes[axis]):
        return reindex_and_fill(weights, indices, 'ffill', axis)
    else:
        return weights
