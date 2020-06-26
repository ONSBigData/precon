"""
Common aggregation functions.    
"""
import pandas as pd

from precon.weights import get_weight_shares, reindex_weights_to_indices
from precon.helpers import reduce_cols, axis_flip
from precon._error_handling import _check_valid_pandas_arg, _handle_axis


def aggregate(indices, weights, axis=1, ignore_na_indices=False):
    """
    Aggregate unchained indices with weights to get sum product.
    
    
    Parameters
    ----------
    indices : Series or DataFrame
        The unchained indices to aggregate.
    weights : Series or DataFrame
        The weights to aggregate.
    axis : {0 or ‘index’, 1 or ‘columns’}, default 1
        Axis along which the function is applied:
            * 0 or ‘index’: apply function to each column.
            * 1 or ‘columns’: apply function to each row.
    ignore_na_indices : bool, default False
        Remove NA indices from aggregation step.
    
    Returns
    -------
    Series
        The aggregated unchained index.
    """
    
    axis = _handle_axis(axis)    
    _check_valid_pandas_arg(indices, 'indices', axis_flip(axis))
    _check_valid_pandas_arg(weights, 'weights', axis_flip(axis))
    
    if (isinstance(weights, pd.core.series.Series)
            and isinstance(indices, pd.core.frame.Frame)):
        weights = weights.to_frame()
    
    # Step through by each period to ignore NAs
    if ignore_na_indices and indices.isna().any().any() == True:
        return (
            indices.apply(
                _aggregate_ignore_na,
                weights=weights,
                axis=axis,
            )
        )
    
    else:
        weight_shares = get_weight_shares(weights, axis)
        weight_shares = reindex_weights_to_indices(
            weight_shares,
            indices,
            axis_flip(axis)
        )
        
        return indices.mul(weight_shares).sum(axis=axis)


def _aggregate_ignore_na(indices, weights):
    """
    Returns the aggregate of an indices and weights Series,
    ignoring the weight of any NaN indices.
    """
    na_indices = indices.isna()
 
    weight_shares = get_weight_shares(weights.loc[~na_indices])
    
    return indices.loc[~na_indices].mul(weight_shares).sum()
    

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

