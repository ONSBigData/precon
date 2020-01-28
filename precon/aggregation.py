# -*- coding: utf-8 -*-
import pandas as pd

from precon.weights import get_weight_shares, reindex_weights_to_indices

def aggregate(indices, weights):
    """
    Takes a set of unchained indices and corresponding weights with matching
    time series index, and produces the weighted aggregate unchained index.
    """
    weight_shares = get_weight_shares(weights)
    weight_shares = reindex_weights_to_indices(weight_shares, indices)
    
    return indices.mul(weight_shares).sum(axis=1)


def reaggregate_index(indices, weights, subs):
    """Returns a reaggregated index after substituting.
    
    Parameters
    ----------
    indices : DataFrame
        A set of unchained component indices.
    weights : DataFrame
        A set of component weights.
    subs : dict of {str : Series}
        A dictionary of components to be substituted where key is
        column name and value is the Series to substitute.
        
    Returns
    -------
    Series
        The unchained aggregate index after substituting.
    """
    subbed_indices = substitute_indices(indices, subs)
    return aggregate(subbed_indices, weights)


def substitute_indices(indices, subs):
    """Substitutes the indices at given keys for given Series.
    
    Parameters
    ----------
    indices : DataFrame
        A set of unchained component indices.
    subs : dict of {str : Series}
        A dictionary of components to be substituted where key is
        column name and value is the Series to substitute.
        
    Returns
    -------
    DataFrame
        The original indices including the substitutes.
    """
    subbed_indices = indices.copy()
    for key, sub in subs.items():
        subbed_indices[key] = sub
    
    return subbed_indices
