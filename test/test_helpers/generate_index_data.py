"""Script to generate index data."""
import collections.abc
from typing import Tuple, Optional, Union, Sequence, Mapping
from warnings import warn

import numpy as np
import pandas as pd
from pandas._typing import Label

IndexLabel = Union[Label, Sequence[Label]]
NestedHierarchy = Union[
    Mapping[str, Sequence[str]],
    Mapping[str, Mapping[str, Sequence[str]]],
]

# The numpy random generator.
RNG = np.random.default_rng(34)

YEARS = 2
BASE_PERIOD = 1
FREQ = 'M'
YEAR_BEGIN = 2017
PERIODS = 12
INDEXES = 3
# COLUMN_HEADERS = range(3)
HIERARCHY = {
    'cpi': {
        'cheese': ['A', 'B', 'C'],
        'beer': ['D', 'E', 'F'],
        'wine': ['G', 'H', 'I'],
    }
}


def create_hiearchy_of_indices(
    rng: np.random.Generator,
    hierarchy: NestedHierarchy,
    year_begin: int,
    base_period: int,
    periods: int,
    freq: str,
    no_of_years: int,
) -> pd.DataFrame:
    """ """      
    dfs = []
    for _, values in hierarchy.items():
        
        if isinstance(values, collections.abc.Sequence):
            
            dfs.append(
                create_multi_year_index(
                    rng,
                    year_begin,
                    base_period,
                    periods,
                    freq,
                    no_of_years,
                    column_headers=values,
                )
            )
        else:
             # Call the function recursively.
            dfs = create_hiearchy_of_indices(
                rng,
                values,     # Values is the new hierarchy
                year_begin,
                base_period,
                periods,
                freq,
                no_of_years,
            )
    
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


def create_period_index(
    year_begin: int,
    base_period: int,
    periods: int,
    freq: str,
) -> pd.PeriodIndex:
    """Creates a period index."""
    # Build the index start period.
    start = str(year_begin) + '-' + str(base_period)
    return pd.period_range(start=start, periods=periods, freq=freq)


def create_index_dataframe(
    rng: np.random.Generator,
    year_begin: int,
    base_period: int,
    periods: int,
    freq: str,
    column_headers: Optional[IndexLabel] = None,
) -> pd.DataFrame:
    """Creates a DataFrame of indices given by size."""
    period_idx = create_period_index(year_begin, base_period, periods, freq)
    ts_idx = period_idx.to_timestamp()
    
    size = (periods, len(column_headers))
    indices = generate_index(rng, size)
    
    return pd.DataFrame(indices, index=ts_idx, columns=column_headers)


def create_multi_year_index(
    rng: np.random.Generator,
    year_begin: int,
    base_period: int,
    periods: int,
    freq: str,
    no_of_years: int,
    column_headers: Optional[IndexLabel] = None,
) -> pd.DataFrame:
    """ """
    return pd.concat([
        create_index_dataframe(
            rng,
            year_begin + i,
            base_period,
            periods,
            freq,
            column_headers,
        )
        for i in range(no_of_years)
    ])
    


if __name__ == "__main__":    
    output = create_hiearchy_of_indices(
        RNG,
        HIERARCHY,
        YEAR_BEGIN,
        BASE_PERIOD,
        PERIODS,
        FREQ,
        YEARS,
    )
        

    print(output)
    
