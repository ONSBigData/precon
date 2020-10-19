"""
Common aggregation functions.
"""
import numpy as np
import pandas as pd
from pandas._typing import Axis, FrameOrSeriesUnion

from precon.weights import get_weight_shares, reindex_weights_to_indices
from precon.helpers import flip
from precon._validation import _handle_axis


def aggregate(
        indices: pd.DataFrame,
        weights: FrameOrSeriesUnion,
        method: str = 'mean',
        axis: Axis = 1,
        ) -> pd.Series:
    """
    Aggregate unchained indices with weights using the given method.

    Supports mean aggregation by default, where the weight shares are
    calculated from the weights, and the sum product of indices and
    weight shares calculated to get the mean.

    Also supports geometric mean aggregation, which takes the exponent
    of the sum product of the natural log of the indices and the weight
    shares.

    If the weights are not the same shape as the indices, then they
    will be reshaped by the function. This allows the user to pass in
    a DataFrame of weights, with only the time periods where the
    weights are updated, as weights are usually fixed over different
    periods.

    Parameters
    ----------
    indices: DataFrame
        The indices or price relatives to aggregate.
    weights: DataFrame or Series
        The weights to be aggregated with the indices. Can be a Series
        when there is only one base period in the dataset.
    method: {'mean', 'geomean'}, str defaults to mean
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
        'mean': mean_aggregate,
        'geomean': geo_mean_aggregate,
    }
    agg_method = methods_lib.get(method)

    # Make sure that the indices and weights have the same time series
    # axis before aggregating.
    weights = reindex_weights_to_indices(weights, indices, flip(axis))

    # Ensure zero or NA indices have zero weight
    weights = weights.mask(indices.isna() | indices.eq(0), 0)

    weight_shares = get_weight_shares(weights, axis)
    return agg_method(indices, weight_shares, axis)


def mean_aggregate(
        indices: pd.DataFrame,
        weight_shares: FrameOrSeriesUnion,
        axis: int = 1,
        ) -> pd.Series:
    """Aggregates indices and weight shares using sum product."""
    # min_count set to 1 to prevent function returning 0 when all
    # values being summed are NA
    return indices.mul(weight_shares).sum(axis=axis, min_count=1)


def geo_mean_aggregate(
        indices: pd.DataFrame,
        weight_shares: FrameOrSeriesUnion,
        axis: int = 1,
        ) -> pd.Series:
    """Aggregates indices and weight shares using geo mean method."""
    # min_count set to 1 to prevent function returning 0 when all
    # values being summed are NA
    return np.exp(
        np.log(indices).mul(weight_shares)
        .sum(axis=axis, min_count=1)
    )
