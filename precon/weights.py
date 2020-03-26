# -*- coding: utf-8 -*-

import pandas as pd
from precon.helpers import reindex_and_fill

def get_weight_shares(weights):
    """If not weight shares already, calculates weight shares."""
    if not weights.sum(axis=1).round().eq(1).all():
        # It is already weight shares so return input
        return weights.div(weights.sum(axis=1), axis=0)
    else:
        return weights


def reindex_weights_to_indices(weights, indices):
    """If not already indexed like indices, reindexes weights."""
    if not (weights.index).equals(indices.index):
        return reindex_and_fill(weights, indices.index, 'ffill')
    else:
        return weights


def jan_adjust_weights(weights, direction='back'):
    """Adjust Feb weights by one month so that weights start in Jan."""
    if direction == 'back':
        return weights.tshift(-1, freq='MS')
    elif direction == 'forward':
        return weights.tshift(1, freq='MS')


def adjust_pre_doublelink(weights, end_year, direction='back'):
    """Jan adjusts only the weights up to the end year."""
    # Double update (Jan & Feb) starts in 2017
    adjusted_weights = pd.concat([
            jan_adjust_weights(weights[:end_year], direction),
            weights[str(int(end_year)+1):],
    ])
    return adjusted_weights