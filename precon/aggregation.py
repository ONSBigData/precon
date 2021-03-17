"""
Common aggregation functions.
"""
from typing import Sequence, Optional, Dict, Union

import numpy as np
import pandas as pd
from pandas._typing import (
    Axis,
    FrameOrSeriesUnion,
    Level,
)

from precon.weights import get_weight_shares, reindex_weights_to_indices
from precon.helpers import (
    flip,
    subset_shared_axis,
    axis_slice,
)
from precon._validation import _handle_axis


def aggregate(
        indices: pd.DataFrame,
        weights: FrameOrSeriesUnion,
        method: str = 'mean',
        axis: Axis = 1,
        ) -> pd.Series:
    """Aggregate indices or price relatives with weights.

    Supports mean aggregation by default, where the weight shares are
    calculated from the weights, and the sum product of indices and
    weight shares calculated to get the mean.

    Also supports geometric mean aggregation, which takes the exponent
    of the sum product of the natural log of the indices and the weight
    shares.

    If the weights are not the same shape as the indices, then they
    will be reshaped by the function. This allows the user to pass in
    a DataFrame of weights, with only the time periods where the
    weights are updated, as weights are usually fixed over a certain
    number of periods.

    Parameters
    ----------
    indices: DataFrame
        The indices or price relatives to aggregate.
    weights: DataFrame or Series
        The weights to be aggregated with the indices. Can be a Series
        when there is only one base period in the dataset.
    method: {'mean', 'geomean'}, str, defaults to 'mean'
        The aggregation method.
    axis : {0 or ‘index’, 1 or ‘columns’}, default 1
        Axis along which the function is applied:
            * 0 or ‘index’: apply function to each column.
            * 1 or ‘columns’: apply function to each row.

    Returns
    -------
    Series:
        The aggregated index.
    """
    axis = _handle_axis(axis)

    methods_lib = {
        'mean': _mean_aggregate,
        'geomean': _geo_mean_aggregate,
    }
    agg_method = methods_lib.get(method)
    
    # Subset the metadata axis to match those of indices, for quicker
    # handling of function when applied by groupby.
    weights = subset_shared_axis(weights, indices, axis)

    # Make sure that the indices and weights have the same time series
    # axis before aggregating.
    weights = reindex_weights_to_indices(weights, indices, flip(axis))

    # Ensure zero, NA and inf indices have zero weight so weight shares
    # calculation reflects the indices being excluded.
    zero_weights_mask = indices.isin([0, np.nan, np.inf])
    masked_weights = weights.mask(zero_weights_mask, 0)

    # Except where all indices are zero, NA and inf.
    slice_ = axis_slice(zero_weights_mask.all(axis), flip(axis))
    masked_weights.loc[slice_] = np.nan

    weight_shares = get_weight_shares(masked_weights, axis)
    return agg_method(indices, weight_shares, axis)


def aggregate_level(
        indices: pd.DataFrame,
        weights: FrameOrSeriesUnion,
        aggregate_on: Union[Level, Sequence[Level]],
        method: str = 'mean',
        axis: Axis = 1,
        ) -> pd.DataFrame:
    """Aggregates the indices or price relatives in each grouping.
    
    The grouping is given by the `aggregate_on` parameter which should
    be a level label or list of level labels to pass to a pandas
    groupby. Passing in hierarchical levels higher than the one to be
    aggregated will mean the data in those levels are retained.
    
    Parameters
    ----------
    indices: DataFrame
        The indices or price relatives to aggregate.
    weights: DataFrame or Series
        The weights to be aggregated with the indices. Can be a Series
        when there is only one base period in the dataset.
    aggregate_on: list of level labels
        The level labels to groupby when aggregating.        
    method: {'mean', 'geomean'}, str, defaults to 'mean'
        The aggregation method.
    axis : {0 or ‘index’, 1 or ‘columns’}, default 1
        Axis along which the function is applied:
            * 0 or ‘index’: apply function to each column.
            * 1 or ‘columns’: apply function to each row.

    Returns
    -------
    DataFrame:
        The aggregated index.

    Examples
    --------

    Example of how to aggregate from price quotes to item level indices using published price quotes data. ONS makes price quote
    data available at `Consumer price inflation item indices and price quotes <https://www.ons.gov.uk/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes>`_
    
    Dummy data below is an example using dummy data. Replace dummy data with price quotes data taken from link provided above. 
    Files are available as csv labelled as Price quotes <month> <year>, e.g Price quotes January 2020

    >>> dummy_data = pd.DataFrame ({
            "price": [0.8, 6.19, 79.0, 0.10, 10.0, 1715.0, 0.78, 29.0, 1.9, 2.09],
            "shop_weight": [1, 3, 3, 3, 1, 1, 1, 1, 1, 1],
            "stratum_weight": [76.49, 37.7, 9.41, 14.71, 2.56, 10.31, 72.65, 4.56, 38.18, 72.65],
            "item_id": [520106, 520106, 520106, 630313, 630313, 630313, 630313, 440129, 440129, 440129],
            "stratum_cell": [1, 2, 3, 1, 2, 3, 4, 1, 2, 3],
            "quote_date": [202001, 202001, 202001, 202001, 202001, 202001, 202001, 202001, 202001, 202001]
        })

    >>> hierarchy = ['item_id', 'stratum_cell']

    Data needs to be in the form of a `MultiIndex <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html>`_.

    >>> price_quotes = dummy_data.pivot_table(
            index=hierarchy,
            columns=['quote_date'],
            values=['price', 'shop_weight', 'stratum_weight']
        )

    Extract the required variables

    >>> prices = price_quotes.xs('price', axis=1)
    >>> shop_weights = price_quotes.xs('shop_weight', axis=1)
    >>> stratum_weights = price_quotes.xs('stratum_weight', axis=1)

    
    >>> stratum_indices = precon.aggregate_level(prices, shop_weights, aggregate_on=hierarchy, axis=0)

    >>> item_indices = precon.aggregate_level(stratum_indices, stratum_weights, aggregate_on='item_id', axis=0)

    """  
    return indices.groupby(aggregate_on, axis).apply(
        aggregate, weights, method, axis,
    )


def aggregate_up_hierarchy(
        indices: pd.DataFrame, 
        weights: pd.DataFrame,
        class_levels: Optional[Sequence[Level]] = None,
        method: str = 'mean',
        axis: Axis = 1,
        ) -> Dict[Level, FrameOrSeriesUnion]:
    """Aggregate up each level of a hierarchy.
    
    For each level given by the `class_levels`, calculates the
    aggregate when grouping on that level. Returns a dict with key
    value pairs of level: aggregated indices.
    
    Requires a MultiIndex. Default behaviour sets class_levels to the 
    MultiIndex level names.
    
    Parameters
    ----------
    indices: DataFrame
        The indices or price relatives to aggregate. Requires a
        MultiIndex on the chosen axis.
    weights: DataFrame or Series
        The weights to be aggregated with the indices. Can be a Series
        when there is only one base period in the dataset. Requires a
        MultiIndex on the chosen axis.
    class_levels: list of level labels
        The class level labels that define the hierarchy aggregate up. 
        Default behaviour sets `class_levels` to the MultiIndex level
        names.
    method: {'mean', 'geomean'}, str, defaults to 'mean'
        The aggregation method.
    axis : {0 or ‘index’, 1 or ‘columns’}, default 1
        Axis along which the function is applied:
            * 0 or ‘index’: apply function to each column.
            * 1 or ‘columns’: apply function to each row.

    Returns
    -------
    dict of DataFrame or Series:
        Returns a dict with key value pairs of level: aggregated
        indices.
        
        There is no key for the lowest level since they aren't
        aggregated. At the highest level, the aggregated indices will
        be a Series.
    """
    if not class_levels:
        class_levels = indices.axes[axis].names
    
    aggregate_on = list(class_levels)
    hierarchy = {}
    
    for i in range(len(class_levels) - 1):
        
        # Pop the first level to get the levels to aggregate_on for
        # this run, then set the key as the next level up.
        aggregate_on.pop(-1)
        key = aggregate_on[-1]
        
        aggregated = aggregate_level(
            indices, weights, aggregate_on, method, axis,
        )
        hierarchy[key] = aggregated
        
        # Set indices and weights for the next loop.
        indices = aggregated
        weights = weights.groupby(aggregate_on, axis).sum()
        
    return hierarchy


def _mean_aggregate(
        indices: pd.DataFrame,
        weight_shares: FrameOrSeriesUnion,
        axis: int = 1,
        ) -> pd.Series:
    """Aggregates indices and weight shares using sum product."""
    # min_count set to 1 to prevent function returning 0 when all
    # values being summed are NA
    return indices.mul(weight_shares).sum(axis=axis, min_count=1)


def _geo_mean_aggregate(
        indices: pd.DataFrame,
        weight_shares: FrameOrSeriesUnion,
        axis: int = 1,
        ) -> pd.Series:
    """Aggregates indices and weight shares using geo mean method."""
    # min_count set to 1 to prevent function returning 0 when all
    # values being summed are NA
    return np.exp(
        np.log(indices)
        .mul(weight_shares)
        .sum(axis=axis, min_count=1)
    )
