"""
Tests for `weights` module.
"""
import numpy as np
import pandas as pd

from pandas import Timestamp
from precon import get_weight_shares, reindex_weights_to_indices
from pandas.testing import assert_frame_equal


def test_get_weight_shares_with_weight_shares(weight_shares_3years):
    """Tests that the function does nothing if already given weight
    shares that sum to 1."""
    result = get_weight_shares(weight_shares_3years)
    
    assert_frame_equal(result, weight_shares_3years)


def test_get_weight_shares_with_weights(
        weights_3years, weight_shares_3years,
        ):
    """Test that weights are correctly converted to weight shares. As
    an extra check, test that the results all sum to 1."""
    result = get_weight_shares(weights_3years)
    
    assert_frame_equal(result, weight_shares_3years)
    assert np.allclose(result.sum(1), np.ones((3, 1)))


def test_reindex_weights_to_indices_jan_weights(
        weights_3years,
        indices_3years,
        reindex_weights_to_indices_outcome_start_jan,
        ):
    """ """
    result = reindex_weights_to_indices(weights_3years, indices_3years)
    assert_frame_equal(result, reindex_weights_to_indices_outcome_start_jan)


def test_reindex_weights_to_indices_feb_weights(
        weights_3years_start_feb,
        indices_3years,
        reindex_weights_to_indices_outcome_start_feb,
        ):
    """ """
    result = reindex_weights_to_indices(
        weights_3years_start_feb,
        indices_3years,
    )
    assert_frame_equal(result, reindex_weights_to_indices_outcome_start_feb)

if __name__ == "__main__":
    
    
    weight_shares = pd.DataFrame.from_records(
        [
            (Timestamp('2012-02-01 00:00:00'), 0.489537029, 0.21362007800000002, 0.29684289199999997),
            (Timestamp('2013-02-01 00:00:00'), 0.535477885, 0.147572705, 0.31694941),
            (Timestamp('2014-02-01 00:00:00'), 0.512055362, 0.1940439, 0.293900738),
        ],
    ).set_index(0, drop=True)
    
    print(get_weight_shares(weight_shares))
    
    indices = pd.DataFrame.from_records(
        [
            (Timestamp('2012-01-01 00:00:00'), 100.0, 100.0, 100.0),
            (Timestamp('2012-02-01 00:00:00'), 101.239553643, 96.60525323799999, 97.776838217),
            (Timestamp('2012-03-01 00:00:00'), 102.03030533, 101.450821724, 96.59101862),
            (Timestamp('2012-04-01 00:00:00'), 104.432402661, 98.000263617, 94.491213369),
            (Timestamp('2012-05-01 00:00:00'), 105.122830333, 95.946873831, 93.731891785),
            (Timestamp('2012-06-01 00:00:00'), 103.976692567, 97.45914568100001, 90.131064035),
            (Timestamp('2012-07-01 00:00:00'), 106.56768678200001, 94.788761174, 94.53487522),
            (Timestamp('2012-08-01 00:00:00'), 106.652151036, 98.478217946, 92.56165627700001),
            (Timestamp('2012-09-01 00:00:00'), 108.97290730799999, 99.986521241, 89.647230903),
            (Timestamp('2012-10-01 00:00:00'), 106.20124385700001, 99.237117891, 92.27819603799999),
            (Timestamp('2012-11-01 00:00:00'), 104.11913898700001, 100.993436318, 95.758970985),
            (Timestamp('2012-12-01 00:00:00'), 107.76600978, 99.60424011299999, 95.697091336),
            (Timestamp('2013-01-01 00:00:00'), 98.74350698299999, 100.357120656, 100.24073830200001),
            (Timestamp('2013-02-01 00:00:00'), 100.46305431100001, 99.98213513200001, 99.499007278),
            (Timestamp('2013-03-01 00:00:00'), 101.943121499, 102.034291064, 96.043392231),
            (Timestamp('2013-04-01 00:00:00'), 99.358987741, 106.513055039, 97.332012817),
            (Timestamp('2013-05-01 00:00:00'), 97.128074038, 106.132168479, 96.799806436),
            (Timestamp('2013-06-01 00:00:00'), 94.42944162, 106.615734964, 93.72086654600001),
            (Timestamp('2013-07-01 00:00:00'), 94.872365481, 103.069773446, 94.490515359),
            (Timestamp('2013-08-01 00:00:00'), 98.239415397, 105.458081805, 93.57271149299999),
            (Timestamp('2013-09-01 00:00:00'), 100.36774827100001, 106.144579258, 90.314524375),
            (Timestamp('2013-10-01 00:00:00'), 100.660205114, 101.844838294, 88.35136848399999),
            (Timestamp('2013-11-01 00:00:00'), 101.33948384799999, 100.592230114, 93.02874928899999),
            (Timestamp('2013-12-01 00:00:00'), 101.74876982299999, 102.709038791, 93.38277933200001),
            (Timestamp('2014-01-01 00:00:00'), 101.73439491, 99.579700011, 104.755837919),
            (Timestamp('2014-02-01 00:00:00'), 100.247760523, 100.76732961, 100.197855834),
            (Timestamp('2014-03-01 00:00:00'), 102.82080245600001, 99.763171909, 100.252537549),
            (Timestamp('2014-04-01 00:00:00'), 104.469889684, 96.207920184, 98.719797067),
            (Timestamp('2014-05-01 00:00:00'), 105.268899775, 99.357641836, 99.99786671),
            (Timestamp('2014-06-01 00:00:00'), 107.41649204299999, 100.844974811, 96.463821506),
            (Timestamp('2014-07-01 00:00:00'), 110.146087435, 102.01075029799999, 94.332755083),
            (Timestamp('2014-08-01 00:00:00'), 109.17068484100001, 101.562418115, 91.15410351700001),
            (Timestamp('2014-09-01 00:00:00'), 109.872892919, 101.471759564, 90.502291475),
            (Timestamp('2014-10-01 00:00:00'), 108.508436998, 98.801947543, 93.97423224399999),
            (Timestamp('2014-11-01 00:00:00'), 109.91248118, 97.730489099, 90.50638234200001),
            (Timestamp('2014-12-01 00:00:00'), 111.19756703600001, 99.734704555, 90.470418612),
        ],
    ).set_index(0, drop=True)
    
    result = reindex_weights_to_indices(weight_shares, indices,)
    