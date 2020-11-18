"""Script to generate index data."""
import collections.abc
import os
from typing import Tuple, Union, Sequence, Mapping

import numpy as np
import pandas as pd
from pandas._typing import Label

IndexLabels = Union[Label, Sequence[Label]]
NestedHierarchy = Union[
    Mapping[str, Sequence[str]],
    Mapping[str, Mapping[str, Sequence[str]]],
]

# The numpy random generator.
RNG = np.random.default_rng(34)

NO_OF_YEARS = 2
BASE_PERIODS = [1]
FREQ = 'M'
YEAR_BEGIN = 2017
PERIODS = 13
INDEXES = 3
HEADERS = range(3)
HIERARCHY = {
    'top': {
        'cheese': ['A', 'B', 'C'],
        'beer': ['D', 'E', ],
        'wine': ['F', 'G'],
    }
}
OUT_DIR = r"..\test_data\aggregate"
INDICES_FILE_NAME = "aggregate_indices_hierarchy.csv"
WEIGHTS_FILE_NAME = "aggregate_weights_hierarchy.csv"


def create_hiearchy(
    rng: np.random.Generator,
    hierarchy: NestedHierarchy,
    func,
    **kwargs,
) -> pd.DataFrame:
    """ """
    dfs = []
    for _, values in hierarchy.items():

        if isinstance(values, collections.abc.Sequence):

            dfs.append(func(rng, headers=values, **kwargs,))
        else:
            # If calues is not a list, call the function recursively
            # with values being the new hierarchy.
            dfs = create_hiearchy(rng, values, func, **kwargs)

    # Convert dfs to list for concat, if not already.
    dfs = [dfs] if not isinstance(dfs, list) else dfs

    return pd.concat(dfs, axis=1, keys=hierarchy.keys())


def get_index_growth(
    rng: np.random.Generator,
    size: Tuple[int, int],
) -> np.ndarray:
    """Calculates cumulative growth to mimic index growth."""
    # For indices, they are only decreasing a fifth of the time
    signs = np.sign(rng.random(size) - 0.2)

    pois = rng.poisson(1, size)
    noise = rng.random(size)

    growth = (pois + noise) * signs
    growth[0, :] = 0

    return growth.cumsum(axis=0)


def generate_index(
    rng: np.random.Generator,
    size: Tuple[int, int],
) -> np.ndarray:
    """Generates fake indices.

    Works by taking a matrix of all 100 values and adds simulated
    cumulative growth to each value. The size argument controls the
    number of periods (size[0]) and the number of indices (size[1]).
    """
    idx = np.ones(size) * 100
    growth = get_index_growth(rng, size)

    return idx + growth


def generate_weights(
    rng: np.random.Generator,
    year_begin: int,
    base_periods: Sequence[int],
    no_of_years: int,
    headers: IndexLabels,
) -> np.ndarray:
    """ """
    x = no_of_years * len(base_periods)
    y = len(headers)

    first_year_weights = rng.integers(1, 20, (1, y))
    # Rearrange to length needed.
    weights = np.tile(first_year_weights, (x, 1))

    change = RNG.integers(-2, 2, (x, y), endpoint=True)
    change[0, :] = 0

    change = change.cumsum(axis=0)

    # Add change to weights and ensure weights stay >= 1.
    return np.clip(weights + change, 1, None)


def create_weights_dataframe(
    rng: np.random.Generator,
    year_begin: int,
    base_periods: Sequence[int],
    no_of_years: int,
    headers: IndexLabels,
) -> pd.DataFrame:
    """Creates a DataFrame of weights given by size."""
    ts_idx = pd.to_datetime([
        join_year_month(year_begin + i, base_period)
        for i in range(no_of_years)
        for base_period in base_periods
    ])

    weights = generate_weights(
        rng, year_begin, base_periods, no_of_years, headers,
    )

    df = pd.DataFrame(weights, index=ts_idx, columns=headers)

    return df.shift(freq='MS')


def join_year_month(year, month):
    return str(year) + '-' + str(month)


def create_period_index(
    year_begin: int,
    base_period: int,
    periods: int,
    freq: str,
) -> pd.PeriodIndex:
    """Creates a period index."""
    # Build the index start period.
    start = join_year_month(year_begin, base_period)
    return pd.period_range(start=start, periods=periods, freq=freq)


def create_index_dataframe(
    rng: np.random.Generator,
    year_begin: int,
    base_period: int,
    periods: int,
    freq: str,
    headers: IndexLabels,
) -> pd.DataFrame:
    """Creates a DataFrame of indices given by size."""
    # Can't be more than 13 periods.
    if (base_period + periods - 1) > 13:
        periods = (base_period + periods) % 13

    period_idx = create_period_index(year_begin, base_period, periods, freq)
    ts_idx = period_idx.to_timestamp()

    size = (periods, len(headers))
    indices = generate_index(rng, size)

    return pd.DataFrame(indices, index=ts_idx, columns=headers)


def create_multi_year_index(
    rng: np.random.Generator,
    year_begin: int,
    base_periods: Sequence[int],
    periods: int,
    freq: str,
    no_of_years: int,
    headers: IndexLabels,
) -> pd.DataFrame:
    """ """
    df = pd.concat([
        create_index_dataframe(
            rng,
            year_begin + i,
            base_period,
            periods,
            freq,
            headers,
        )
        for i in range(no_of_years)
        for base_period in base_periods
    ])

    return df.groupby(level=0).first()


def add_suffix(fname, suffix):
    """ """
    name, extension = fname.split(".")
    return name + "_" + suffix + "." + extension


if __name__ == "__main__":
    indices = create_hiearchy(
        RNG,
        HIERARCHY,
        create_multi_year_index,
        year_begin=YEAR_BEGIN,
        base_periods=BASE_PERIODS,
        periods=PERIODS,
        freq=FREQ,
        no_of_years=NO_OF_YEARS,
    )

    print(indices)

    weights = create_hiearchy(
        RNG,
        HIERARCHY,
        create_weights_dataframe,
        year_begin=YEAR_BEGIN,
        base_periods=BASE_PERIODS,
        no_of_years=NO_OF_YEARS,
    )

    print(weights)
