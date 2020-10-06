# -*- coding: utf-8 -*-

import pandas as pd

from natstats.helpers import reindex_and_fill, flip, _get_end_year
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


def reindex_to_update_periods(weights):
    """Returns only months where weights are updated.
    
    Useful for reversing a reindex and fill operation where the weights
    repeat monthly. Takes Feb values pre 2017 and Jan & Feb values post
    2017 for the double update.
    """
    # TODO: Rewrite using shift
    pre_weights = weights.loc[:'2016']
    post_weights = weights.loc['2017':]
    
    to_concat = [
        pre_weights.loc[pre_weights.index.month == 2],
        post_weights.loc[post_weights.index.month.isin([1, 2])],
    ]
    
    return pd.concat(to_concat)


def jan_adjust_weights(weights, direction='back'):
    """Adjust Feb weights by one month so that weights start in Jan."""
    if direction == 'back':
        return weights.tshift(-1, freq='MS')
    elif direction == 'forward':
        return weights.tshift(1, freq='MS')


def adjust_pre_doublelink(weights, start_year='2017', direction='back'): 
    """Jan adjusts only the weights up to the end year."""
    # Double update (Jan & Feb) starts in 2017
    return pd.concat([
        jan_adjust_weights(weights[:_get_end_year(start_year)], direction),
        weights[start_year:],
    ])

