"""Functions to manipulate index weights."""
import pandas as pd
from pandas._typing import Axis, FrameOrSeries, FrameOrSeriesUnion

from precon.helpers import reindex_and_fill, flip
from precon._validation import _handle_axis


def get_weight_shares(
        weights: FrameOrSeries,
        axis: Axis = 1,
        ) -> FrameOrSeries:
    """If not weight shares already, calculates weight shares."""
    axis = _handle_axis(axis)
    # TODO: test precision
    if not weights.sum(axis).round(5).eq(1).all():
        return weights.div(weights.sum(axis), axis=flip(axis))

    else:   # It is already weight shares so return input
        return weights


def reindex_weights_to_indices(
        weights: FrameOrSeriesUnion,
        indices: pd.DataFrame,
        axis: Axis = 0,
        ) -> pd.DataFrame:
    """Reshapes and reindexes weights to indices, if they do not
    already have the same shape and share the same index frequency.
    """
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
