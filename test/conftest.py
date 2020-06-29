# -*- coding: utf-8 -*-

import pytest
import pandas as pd
from pandas import Timestamp


### AGGREGATION FIXTURES ---------------------------------------------------

@pytest.fixture()
def aggregate_indices_3years():
    return pd.DataFrame.from_records(
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

@pytest.fixture()
def aggregate_outcome_3years():
    return pd.DataFrame.from_records(
        [
            (Timestamp('2012-01-01 00:00:00'), 100.0),
            (Timestamp('2012-02-01 00:00:00'), 99.22169156),
            (Timestamp('2012-03-01 00:00:00'), 100.29190240000001),
            (Timestamp('2012-04-01 00:00:00'), 100.10739720000001),
            (Timestamp('2012-05-01 00:00:00'), 99.78134264),
            (Timestamp('2012-06-01 00:00:00'), 98.47443727),
            (Timestamp('2012-07-01 00:00:00'), 100.4796172),
            (Timestamp('2012-08-01 00:00:00'), 100.7233716),
            (Timestamp('2012-09-01 00:00:00'), 101.31654509999998),
            (Timestamp('2012-10-01 00:00:00'), 100.5806089),
            (Timestamp('2012-11-01 00:00:00'), 100.9697697),
            (Timestamp('2012-12-01 00:00:00'), 102.4399192),
            (Timestamp('2013-01-01 00:00:00'), 99.45617890000001),
            (Timestamp('2013-02-01 00:00:00'), 100.08652959999999),
            (Timestamp('2013-03-01 00:00:00'), 100.0866599),
            (Timestamp('2013-04-01 00:00:00'), 99.7722843),
            (Timestamp('2013-05-01 00:00:00'), 98.35278839),
            (Timestamp('2013-06-01 00:00:00'), 96.00322344),
            (Timestamp('2013-07-01 00:00:00'), 95.96105198),
            (Timestamp('2013-08-01 00:00:00'), 97.82558448),
            (Timestamp('2013-09-01 00:00:00'), 98.03388747),
            (Timestamp('2013-10-01 00:00:00'), 96.93374613),
            (Timestamp('2013-11-01 00:00:00'), 98.59512718),
            (Timestamp('2013-12-01 00:00:00'), 99.23888357),
            (Timestamp('2014-01-01 00:00:00'), 102.2042938),
            (Timestamp('2014-02-01 00:00:00'), 100.3339127),
            (Timestamp('2014-03-01 00:00:00'), 101.4726729),
            (Timestamp('2014-04-01 00:00:00'), 101.17674840000001),
            (Timestamp('2014-05-01 00:00:00'), 102.57269570000001),
            (Timestamp('2014-06-01 00:00:00'), 102.9223313),
            (Timestamp('2014-07-01 00:00:00'), 103.9199248),
            (Timestamp('2014-08-01 00:00:00'), 102.3992605),
            (Timestamp('2014-09-01 00:00:00'), 102.54967020000001),
            (Timestamp('2014-10-01 00:00:00'), 102.35333840000001),
            (Timestamp('2014-11-01 00:00:00'), 101.8451732),
            (Timestamp('2014-12-01 00:00:00'), 102.8815443),
        ],
    ).set_index(0, drop=True).squeeze()


@pytest.fixture()
def aggregate_weights_3years():
    return pd.DataFrame.from_records(
        [
            (Timestamp('2012-01-01 00:00:00'), 5.1869643839999995, 2.263444179, 3.145244219),
            (Timestamp('2013-01-01 00:00:00'), 6.74500585, 1.8588606330000002, 3.992369584),
            (Timestamp('2014-01-01 00:00:00'), 6.23115844, 2.361303832, 3.5764532489999996),
        ],
    ).set_index(0, drop=True)


@pytest.fixture()
def aggregate_weight_shares_3years():
    return pd.DataFrame.from_records(
        [
            (Timestamp('2012-02-01 00:00:00'), 0.489537029, 0.21362007800000002, 0.29684289199999997),
            (Timestamp('2013-02-01 00:00:00'), 0.535477885, 0.147572705, 0.31694941),
            (Timestamp('2014-02-01 00:00:00'), 0.512055362, 0.1940439, 0.293900738),
        ],
    ).set_index(0, drop=True)

@pytest.fixture()
def aggregate_indices_1year(aggregate_indices_3years):
    return aggregate_indices_3years.loc['2012', :]


@pytest.fixture()
def aggregate_outcome_1year(aggregate_outcome_3years):
    return aggregate_outcome_3years.loc['2012']


@pytest.fixture()
def aggregate_weights_1year(aggregate_weights_3years):
    return aggregate_weights_3years.loc['2012',  :]


@pytest.fixture()
def aggregate_indices_6months(aggregate_indices_3years):
    return aggregate_indices_3years.loc['2012-Jan':'2012-Jun', :]


@pytest.fixture()
def aggregate_outcome_6months(aggregate_outcome_3years):
    return aggregate_outcome_3years.loc['2012-Jan':'2012-Jun']


@pytest.fixture()
def aggregate_weights_6months(aggregate_weights_3years):
    return aggregate_weights_3years.loc['2012',  :]


@pytest.fixture()
def aggregate_indices_transposed(aggregate_indices_3years):
    return aggregate_indices_3years.T


@pytest.fixture()
def aggregate_weights_transposed(aggregate_weights_3years):
    return aggregate_weights_3years.T


@pytest.fixture()
def aggregate_outcome_transposed(aggregate_outcome_3years):
    return aggregate_outcome_3years


test_inputs = ["indices", "outcome", "weights"]
names = ["3years", "1year", "6months", "transposed"]
axis = [1, 1, 1, 0]

agg_params = [
    tuple(["aggregate_" + var + "_" + name for var in test_inputs])
    for name in names
]

agg_params = [params + (axis[i],) for i, params in enumerate(agg_params)]

@pytest.fixture(
    params=[*agg_params],
    ids=names,
)
def aggregate_combinator(request):
    """ """
    indices = request.getfixturevalue(request.param[0])
    outcome = request.getfixturevalue(request.param[1])
    weights = request.getfixturevalue(request.param[2])
    axis = request.param[3]
    
    return indices, outcome, weights, axis


### WEIGHTS FIXTURES ------------------------------------------------------
@pytest.fixture()
def get_weight_shares_weight_shares_3years():
    return pd.DataFrame.from_records(
        [
            (Timestamp('2012-02-01 00:00:00'), 0.489537029, 0.21362007800000002, 0.29684289199999997),
            (Timestamp('2013-02-01 00:00:00'), 0.535477885, 0.147572705, 0.31694941),
            (Timestamp('2014-02-01 00:00:00'), 0.512055362, 0.1940439, 0.293900738),
        ],
    ).set_index(0, drop=True)


@pytest.fixture()
def get_weight_shares_weights_3years():
    return pd.DataFrame.from_records(
        [
            (Timestamp('2012-02-01 00:00:00'), 5.1869643839999995, 2.263444179, 3.145244219),
            (Timestamp('2013-02-01 00:00:00'), 6.74500585, 1.8588606330000002, 3.992369584),
            (Timestamp('2014-02-01 00:00:00'), 6.23115844, 2.361303832, 3.5764532489999996),
        ],
    ).set_index(0, drop=True)

