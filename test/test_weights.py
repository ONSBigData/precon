"""
Tests for `weights` module.
"""
import numpy as np
import pandas as pd
import pytest
from pandas import Timestamp
from precon import get_weight_shares, reindex_weights_to_indices
from pandas._testing import assert_frame_equal, assert_series_equal


def test_get_weight_shares_with_weight_shares(
        get_weight_shares_weight_shares_3years,
        ):
    """Tests that the function does nothing if already given weight
    shares that sum to 1."""
    res = get_weight_shares(get_weight_shares_weight_shares_3years)
    
    assert_frame_equal(res, get_weight_shares_weight_shares_3years)


def test_get_weight_shares_with_weights(
        get_weight_shares_weights_3years,
        get_weight_shares_weight_shares_3years,
        ):
    """Test that weights are correctly converted to weight shares. As
    an extra check, test that the results all sum to 1."""
    res = get_weight_shares(get_weight_shares_weights_3years)
    
    assert_frame_equal(res, get_weight_shares_weight_shares_3years)
    assert np.allclose(res.sum(1), np.ones((3, 1)))


if __name__ == "__main__":
    
    
    weight_shares = pd.DataFrame.from_records(
        [
            (Timestamp('2012-02-01 00:00:00'), 0.489537029, 0.21362007800000002, 0.29684289199999997),
            (Timestamp('2013-02-01 00:00:00'), 0.535477885, 0.147572705, 0.31694941),
            (Timestamp('2014-02-01 00:00:00'), 0.512055362, 0.1940439, 0.293900738),
        ],
    ).set_index(0, drop=True)
    
    print(get_weight_shares(weight_shares))
    
    