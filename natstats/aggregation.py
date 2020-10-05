"""
Common aggregation functions.    
"""
from typing import Union

import numpy as np
import pandas as pd

from natstats.weights import get_weight_shares, reindex_weights_to_indices
from natstats.helpers import reduce_cols, flip
# from natstats._error_handling import _check_valid_pandas_arg
# from natstats._error_handling import _assert_equal_axis_labels
from natstats._error_handling import _handle_axis

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
    
    
    if isinstance(weights, pd.core.series.Series):
        weights = weights.to_frame()
        if axis == 1:
            weights = weights.T
    
    # _check_valid_pandas_arg(indices, 'indices', flip(axis))
    # _check_valid_pandas_arg(weights, 'weights', flip(axis))
    
    # _assert_equal_axis_labels(indices, weights, axis)

    weights = reindex_weights_to_indices(
        weights,
        indices,
        flip(axis),
    )
    
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


def sum_up_hierarchy(df, labels, tree):
    """Expand to the full structure then aggregate sum up hierarchy."""
    df_expanded = expand_full_structure(df, labels)
    return tree.aggregate_sum(df_expanded)


def expand_full_structure(df, headers):
    """
    Reindexes the columns of the given DataFrame to the full
    classification structure. Resulting NaNs filled with zeros.
    
    Parameters
    ----------
    df : DataFrame
        The leaf values, either weights or indices.
    headers : list or Series
        The list of column labels representing the full class structure
        to reindex by.
        
    Returns
    -------
    DataFrame
        The indices or weights reindexed to the full class structure
        given by the headers parameter.
    """
    # Check existing DataFrame columns are in headers.
    if not all(df.columns.isin(headers)):
        raise Exception("Not all columns of given DataFrame are in headers.")
        
    return df.reindex(headers, axis=1).fillna(0)


def create_special_aggregation(indices, weights, name, agg_cols):
    """Create a special aggregation by aggregating indices and 
    summing the weights.
    """
    other_agg = aggregate(indices[agg_cols], weights[agg_cols])
    other_agg = other_agg.rename(name)

    to_concat = [indices.loc[:, ~indices.columns.isin(agg_cols)], other_agg]
    pub_subs = pd.concat(to_concat, axis=1)
    
    pub_weights = reduce_cols(
            weights,
            newcol=name,
            cols=agg_cols,
            reduce_func=sum,
            drop=True,
            swap=4,
    )
    
    return pub_subs, pub_weights

