"""Script to generate index data."""
import collections.abc
import os
from typing import Tuple, Union, Sequence, Mapping, Callable

import numpy as np
import pandas as pd
from pandas._typing import Label

# Some additional types.
IndexLabels = Union[Label, Sequence[Label]]
NestedHierarchy = Union[
    Mapping[str, Sequence[str]],
    Mapping[str, Mapping[str, Sequence[str]]],
]

# The numpy random generator.
RNG = np.random.default_rng(34)

# Settings...
NO_OF_YEARS = 2
YEAR_BEGIN = 2017
BASE_PERIODS = [1]  # The base months
PERIODS = 24     # The number of periods in the index
FREQ = 'M'
HEADERS = range(3)
HIERARCHY = {
    'top': {
        'cheese': ['A', 'B', 'C'],
        'beer': ['D', 'E'],
        'wine': ['F', 'G'],
    }
}
OUT_DIR = r"..\test_data\aggregate"
INDICES_FILE_NAME = "aggregate_indices_hierarchy.csv"
WEIGHTS_FILE_NAME = "aggregate_weights_hierarchy.csv"


def create_hierarchy(
    rng: np.random.Generator,
    hierarchy: NestedHierarchy,
    func: Callable[..., pd.DataFrame],
    **kwargs,
) -> pd.DataFrame:
    """Creates a hierarchical DataFrame of values.

    Uses a given function and kwargs to create a DataFrame with the
    given hierarchy as the columns MultiIndex. The value are generated
    with the passed numpy random generator.

    Parameters
    ----------
    rng: Generator
        Numpy generator for generating random numbers.
    hierarchy: dict of str -> list of ints, or dict of str -> dict
        A nested dictionary containing the column MultiIndex hierarchy.
    func:
        A function that creates a DataFrame.

        The function needs to accept a numpy random generator as the
        first positional argument, and it also needs a keyword argument
        called "headers" that accepts a list of column names.

    Returns
    -------
    DataFrame:
        Randomly generated values according to func, with hierarchical
        MultiIndex in the columns.

    """
    dfs = []
    for _, values in hierarchy.items():

        if isinstance(values, collections.abc.Sequence):
            dfs.append(func(rng, headers=values, **kwargs))
        else:
            # If values is not a sequence, call the function
            # recursively with values being the new hierarchy.
            dfs = create_hierarchy(rng, values, func, **kwargs)

    # Convert dfs to list for concat, if not already.
    dfs = [dfs] if not isinstance(dfs, list) else dfs

    return pd.concat(dfs, axis=1, keys=hierarchy.keys())


def get_index_growth(
    rng: np.random.Generator,
    size: Tuple[int, int],
) -> np.ndarray:
    """Calculates cumulative growth to mimic index growth."""
    # Subtracts 0.2 from values [0, 1] so that 1/5 have negative sign.
    # This is arbitrary, and results in the index increasing in 4 out
    # of 5 months.
    signs = np.sign(rng.random(size) - 0.2)

    # Takes a poisson dist with lambda = 1 and adds some random noise.
    # Multiply by signs to apply increasing / decreasing.
    pois = rng.poisson(1, size)
    noise = rng.random(size)

    growth = (pois + noise) * signs
    growth[0, :] = 0    # No growth at index start.

    return growth.cumsum(axis=0)


def generate_indices(
    rng: np.random.Generator,
    size: Tuple[int, int],
) -> np.ndarray:
    """Generates fake indices.

    Works by taking a matrix of all 100 values and adds random
    cumulative growth to each value.

    Parameters
    ----------
    rng: Generator
        Numpy generator for generating random numbers.
    size: tuple of (int, int)
        The x, y size of the resulting index matrix.

        size[0] is the number of periods
        size[1] is the number of indices

    Returns
    -------
    ndarray:
        A matrix of time series indices.

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
    """Generates fake weights.

    Selects random ints between 1 and 19 for weights in the first
    period. Then adds a random int between -2 and 2 for each subsequent
    weights update. Returns a numpy matrix of weights, clipped so that
    no weight can be lower than 1.

    Parameters
    ----------
    rng: Generator
        Numpy generator for generating random numbers.
    year_begin: int
        The start year for the first set of weights.
    base_periods: sequence of int
        A list of months, given by the int equivalent, for a weights
        update each year.
    no_of_years: int
        The number of years to generate weights for.
    headers: label, or sequence of labels
        A label or list of labels for each time series column name.

    Returns
    -------
    ndarray:
        A matrix of time series weights.

    """
    x = no_of_years * len(base_periods)
    y = len(headers)

    # Weights randomly initiated as an int between 1 and 19.
    first_year_weights = rng.integers(1, 20, (1, y))
    # Rearrange to length needed.
    weights = np.tile(first_year_weights, (x, 1))

    # Assumes that weights increase or decrease by no more than an
    # increment of 2 each base price refresh.
    change = RNG.integers(-2, 2, (x, y), endpoint=True)
    change[0, :] = 0    # No change at weights start.

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
    """Creates a DataFrame of weights for given size.

    Generates weights for each base period in each year on the x axis,
    and each header on the y axis. Shifts the weights by one month
    since they come into effect the month after the base period.

    Parameters
    ----------
    rng: Generator
        Numpy generator for generating random numbers.
    year_begin: int
        The start year for the first set of weights.
    base_periods: sequence of int
        A list of months, given by the int equivalent, for a weights
        update each year.
    no_of_years: int
        The number of years to generate weights for.
    headers: label, or sequence of labels
        A label or list of labels for each time series column name.

    Returns
    -------
    DataFrame
        The weights for each base period in each year, shifted by one
        month.

    """
    # Create DateTime values for each combination of year and base
    # month.
    ts_idx = pd.to_datetime([
        join_year_month(year_begin + i, base_period)
        for i in range(no_of_years)
        for base_period in base_periods
    ])

    weights = generate_weights(
        rng, year_begin, base_periods, no_of_years, headers,
    )

    df = pd.DataFrame(weights, index=ts_idx, columns=headers)

    # Shift weights by a month since they come into effect one month
    # after base price refresh.
    return df.shift(freq='MS')


def join_year_month(year, month):
    """Joins year and month for parsing with pd.to_datetime."""
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
    chained: bool = False,
) -> pd.DataFrame:
    """Creates a DataFrame of indices for given size.

    Generates indices for the given periods starting from a base period
    in a given year. Generates a index for each column label given by
    headers.

    Parameters
    ----------
    rng: Generator
        Numpy generator for generating random numbers.
    year_begin: int
        The start year for the first set of weights.
    base_period: int
        A base period to begin the index, given by the int equivalent.
    periods: int
        The number of periods to generate a growing index for.
    freq: str
        A freq value to pass to pandas period_range function.
    headers: label, or sequence of labels
        A label or list of labels for each time series column name.

    Returns
    -------
    DataFrame
        The indices for the given periods from the base period in the
        given start year.

    """
    # Can't be more than 13 periods. If it is, truncate periods by
    # taking the modulus.
    if not chained:
        if (base_period + periods - 1) > 13:
            periods = (base_period + periods) % 13

    period_idx = create_period_index(year_begin, base_period, periods, freq)
    ts_idx = period_idx.to_timestamp()

    size = (periods, len(headers))
    indices = generate_indices(rng, size)

    return pd.DataFrame(indices, index=ts_idx, columns=headers)


def create_multi_within_year_indices(
    rng: np.random.Generator,
    year_begin: int,
    base_periods: Sequence[int],
    periods: int,
    freq: str,
    no_of_years: int,
    headers: IndexLabels,
) -> pd.DataFrame:
    """Creates a DataFrame with multiple within-year indices.

    Concatenates indices DataFrames created for each year combination
    of years and base periods for base price refresh.

    Parameters
    ----------
    rng: Generator
        Numpy generator for generating random numbers.
    year_begin: int
        The start year for the first set of weights.
    base_periods: sequence of int
        A list of months, given by the int equivalent, for a base
        price refresh each year.
    periods: int
        The number of periods to generate a growing index for.
    freq: str
        A freq value to pass to pandas period_range function.
    no_of_years: int
        The number of years to generate within-year indices for.
    headers: label, or sequence of labels
        A label or list of labels for each time series column name.

    Returns
    -------
    DataFrame
        The indices for the given periods from the base period in the
        given start year.

    """
    # Concatenate within year indices for each
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

    # Drop any duplicated crossover periods. This should drop the
    # period where the values = 100.
    # TODO: Maybe make this optional?
    return df.groupby(level=0).first()


def reindex_and_fill(df, other, first='ffill', group_on=None, axis=0):
    """Reindex an axis of a DataFrame or Series by another and fill.

    Reindexes one axis of a DataFrame or a Series to another and then
    fills forward and backwards. By default fills forward first. Can
    optionally fill within groups by passing a grouping to `group_on`.

    Parameters
    ----------
    df : DataFrame or Series
        The DataFrame to reindex.
    other : Object of the same data type
        Its row or column indices are used to define the new indices of
        the first parameter, depending on axis parameter.
    first : str
        Direction to fill first.
    group_on:
        Argument to pass to a pandas groupby before filling,
        i.e. passing lambda x: x.year will fill for each years worth of
        data.
    axis : {0 or ‘index’, 1 or ‘columns’}
        Whether to compare by the index (0 or ‘index’) or columns
        (1 or ‘columns’). For Series input, axis to match Series index
        on.

    Returns
    -------
    DataFrame
        The reindexed and filled DataFrame.
    """
    reindexed = df.reindex(other.axes[axis], axis=axis)

    if first == 'ffill':
        def fill_func(x): return x.ffill(axis).bfill(axis)
    elif first == 'bfill':
        def fill_func(x): return x.bfill(axis).ffill(axis)

    if group_on:
        return reindexed.groupby(group_on, axis=axis).apply(fill_func)

    return fill_func(reindexed)


def add_suffix(fname, suffix):
    """Adds a suffix to a file name."""
    name, extension = fname.split(".")
    return name + "_" + suffix + "." + extension


def created_chained_monthly_indices(
    rng: np.random.Generator,
    year_begin: int,
    base_period: int,
    periods: int,
    headers: IndexLabels,
) -> pd.DataFrame:
    """
    Creates chained monthly indices for the given number of periods,
    starting at the base period in year begin.
    """
    return create_index_dataframe(
        rng,
        year_begin,
        base_period,
        periods,
        freq='M',
        headers=headers,
        chained=True,
        )


if __name__ == "__main__":
    # Create hierarchy of within-year indices and save.
    indices = create_hierarchy(
        RNG,
        HIERARCHY,
        create_multi_within_year_indices,
        year_begin=YEAR_BEGIN,
        base_periods=BASE_PERIODS,
        periods=PERIODS,
        freq=FREQ,
        no_of_years=NO_OF_YEARS,
    )

    print(indices)
    indices.to_csv(os.path.join(OUT_DIR, INDICES_FILE_NAME))

    # Create hierarchy of weights and save.
    weights = create_hierarchy(
        RNG,
        HIERARCHY,
        create_weights_dataframe,
        year_begin=YEAR_BEGIN,
        base_periods=BASE_PERIODS,
        no_of_years=NO_OF_YEARS,
    )

    print(weights)
    weights.to_csv(os.path.join(OUT_DIR, WEIGHTS_FILE_NAME))

    # Save weights reindexed to indices.
    if PERIODS != 13:
        def group_on(x): return x.year

    long_weights = reindex_and_fill(weights, indices, 'ffill', group_on)
    long_weights.to_csv(
        os.path.join(
            OUT_DIR,
            add_suffix(WEIGHTS_FILE_NAME, 'long'),
        )
    )
