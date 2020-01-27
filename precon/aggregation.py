# -*- coding: utf-8 -*-
import pandas as pd

from weights import get_weight_shares, reindex_weights_to_indices

def aggregate(indices, weights):
    """
    Takes a set of unchained indices and corresponding weights with matching
    time series index, and produces the weighted aggregate unchained index.
    """
    weight_shares = get_weight_shares(weights)
    weight_shares = reindex_weights_to_indices(weight_shares, indices)
    
    return indices.mul(weight_shares).sum(axis=1)

