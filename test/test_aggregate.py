"""
Tests for `aggregate` module.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from precon import aggregate
from pandas import Timestamp
from pandas.testing import assert_series_equal
from pandas._typing import Axis


@dataclass
class AggTestCase:
    name: str
    indices: str
    weights: str
    outcome: str
    axis: Axis = 1


@pytest.fixture(
    params=[
        AggTestCase(
            name="3years",
            indices="indices_3years",
            weights="weights_3years",
            outcome="aggregate_outcome_3years",
        ),
        AggTestCase(
            name="1year",
            indices="indices_1year",
            weights="weights_1year",
            outcome="aggregate_outcome_1year",
        ),
        AggTestCase(
            name="6months",
            indices="indices_6months",
            weights="weights_6months",
            outcome="aggregate_outcome_6months",
        ),
        AggTestCase(
            name="transposed",
            indices="indices_transposed",
            weights="weights_transposed",
            outcome="aggregate_outcome_3years",
            axis=0,
        ),
        AggTestCase(
            name="missing",
            indices="indices_missing",
            weights="weights_3years",
            outcome="aggregate_outcome_missing",
        ),
        AggTestCase(
            name="missing_transposed",
            indices="indices_missing_transposed",
            weights="weights_transposed",
            outcome="aggregate_outcome_missing",
            axis=0,
        ),
    ],
    ids=lambda v: v.name,
)
def aggregate_combinator(request):
    """ """
    indices = request.getfixturevalue(request.param.indices)
    weights = request.getfixturevalue(request.param.weights)
    outcome = request.getfixturevalue(request.param.outcome)

    axis = request.param.axis
    
    return indices, weights, outcome, axis


def test_aggregate(aggregate_combinator):
    """Functional test to test for expected output.
    
    Tests the following scenarios:
        * 3 years, 3 indices, 3 weights
        * 1 year, 3 indices, 3 weights
        * 6 months, 3 indices, 3 weights
        * 3 years, 3 indices (transposed), 3 weights (transposed)
        * 3 years, 3 indices (missing values), 3 weights
        * 3 years, 3 indices (missing values + transposed),
            3 weights (transposed)
    """
    # GIVEN indices and weights
    # AND the outcome
    # WHEN indices and weights are aggregated together
    # THEN they should equal the outcome
    indices, weights, outcome, axis = aggregate_combinator
    
    aggregated = aggregate(indices, weights, axis=axis)

    assert_series_equal(aggregated, outcome, check_names=False)


def test_aggregate_output_type_weight_dataframe(
        indices_3years, weights_3years,
        ):
    """The aggregate function should return a pandas Series."""
    # GIVEN indices and weights pandas DataFrames
    # WHEN they are aggregated together
    # THEN returned aggregated is of type pandas Series
    aggregated = aggregate(indices_3years, weights_3years)
    assert isinstance(aggregated, pd.core.series.Series)


def test_aggregate_output_type_weight_series(
        indices_6months, weights_6months,
        ):
    """The aggregate function should return a pandas Series."""
    # GIVEN indices DataFrame and weights Series
    # WHEN they are aggregated together
    # THEN returned aggregated is of type pandas Series
    aggregated = aggregate(indices_6months, weights_6months)
    assert isinstance(aggregated, pd.Series)
    

@pytest.mark.parametrize('axis', ['toes', 5]) # True])
def test_aggregate_handles_axis(axis):
    with pytest.raises(ValueError):
        aggregate(
            pd.DataFrame(index=pd.DatetimeIndex(['2017-01'])),
            pd.DataFrame(index=pd.DatetimeIndex(['2017-01'])),
            axis=axis,
        ) 
    

# def test_aggregate_output_not_na(indices, weights):
#     """Test that there are no additional NAs in the output."""
#     aggregated = aggregate(indices, weights)
#     assert aggregated.isna().sum() == indices.isna().sum()

if __name__ == "__main__":
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
    
    weights = pd.DataFrame.from_records(
        [
            (Timestamp('2012-01-01 00:00:00'), 5.1869643839999995, 2.263444179, 3.145244219),
            (Timestamp('2013-01-01 00:00:00'), 6.74500585, 1.8588606330000002, 3.992369584),
            (Timestamp('2014-01-01 00:00:00'), 6.23115844, 2.361303832, 3.5764532489999996),
        ],
    ).set_index(0, drop=True)
    
    # weights.columns = [0, 5, 3]
    # weights = weights['2012']
    # indices = indices['2012-06']
    
    # aggregated = aggregate(indices, weights)
    

    indices2 = pd.DataFrame.from_records(
        [
            (Timestamp('2013-01-01 00:00:00'), 100.0, 100.0, 100.0),
            (Timestamp('2013-02-01 00:00:00'), 97.02107546, 96.33756279, 96.85624492),
            (Timestamp('2013-03-01 00:00:00'), 100.2030568, 99.38630881, np.nan),
            (Timestamp('2013-04-01 00:00:00'), 103.9939693, 103.707301, 102.8423593),
            (Timestamp('2013-05-01 00:00:00'), 107.55573159999999, np.nan, 106.4030102),
            (Timestamp('2013-06-01 00:00:00'), 102.15950079999999, 102.1210296, 103.6548477),
            (Timestamp('2013-07-01 00:00:00'), 106.83231370000001, 105.71647240000001, 106.23764979999999),
        ],
    ).set_index(0, drop=True)


    weights2 = pd.DataFrame.from_records(
       [(Timestamp('2013-01-01 00:00:00'), 13.1223919, 7.90254844, 3.20531341),]
    ).set_index(0, drop=True).squeeze()


    aggregated2 = aggregate(indices2.T, weights2.T, axis=0, ignore_na_indices=True)
    aggregated3 = aggregate(indices2, weights2, axis=1, ignore_na_indices=True)
    
    
    def aggregate_indices_missing(aggregate_indices_3years):
        """ """
        aggregate_indices_missing = aggregate_indices_3years.copy()
        
        change_to_nans = [
            ('2012-06', 2),
            ('2012-12', 3),
            ('2013-10', 2),
            ('2014-07', 1),
        ]
        
        for sl in change_to_nans:
            aggregate_indices_missing.loc[sl] = np.nan
        
        return aggregate_indices_missing
    
    indices_missing = aggregate_indices_missing(indices)
    aggregated3 = aggregate(indices_missing, weights, ignore_na_indices=True)
    