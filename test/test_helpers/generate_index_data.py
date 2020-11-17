"""Script to generate index data."""
from typing import Tuple, Optional, Union, Sequence
from warnings import warn

import numpy as np
import pandas as pd
from pandas._typing import Label

IndexLabel = Union[Label, Sequence[Label]]

YEARS = 2
BASE_PERIOD = 3
FREQ = 'M'
YEAR_BEGIN = 2017
PERIODS = 12
INDEXES = 3
GROUPS = ['A', 'B', 'C']
HIERARCHY = {
    'cpi': {
        'cheese': ['A', 'B', 'C'],
        'beer': ['D', 'E', 'F'],
        'wine': ['G', 'H', 'I'],
    }
}

if (PERIODS > 12) & (FREQ == 'M'):
    warn("The settings given will create duplicate time periods.")


def get_index_growth(
    rng: np.random.Generator, 
    size: Tuple[int, int],
) -> np.ndarray:
    """ """
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
    """ """
    idx = np.ones(size) * 100
    growth = get_index_growth(rng, size)
    
    return idx + growth


def create_period_index(
    year_begin: int,
    base_period: int,
    periods: int,
    freq: str,
) -> pd.PeriodIndex:
    """ """
    # Build the index start period.
    start = str(year_begin) + '-' + str(base_period)
    return pd.period_range(start=start, periods=periods, freq=freq)


def create_index_dataframe(
    rng: np.random.Generator,
    size: Tuple[int, int],
    year_begin: int,
    base_period: int,
    periods: int,
    freq: str,
    column_headers: Optional[IndexLabel] = None,
) -> pd.DataFrame:
    """ """
    period_idx = create_period_index(year_begin, base_period, periods, freq)
    ts_idx = period_idx.to_timestamp()
    
    indices = generate_index(rng, size)
    
    return pd.DataFrame(indices, index=ts_idx, columns=column_headers)


if __name__ == "__main__":
    rng = np.random.default_rng(34)
    
    group_dfs = []
    for _ in range(len(GROUPS)):
        
        size = (PERIODS, INDEXES)
            
        index_dfs = [
            create_index_dataframe(
                rng,
                size,
                YEAR_BEGIN + i,
                BASE_PERIOD,
                PERIODS,
                FREQ,
            )
            for i in range(YEARS)
        ]
        
    
        # Concat the DataFrames to get our output.
        ts_dfs = pd.concat(index_dfs)
        
        group_dfs.append(ts_dfs)
        
    output = pd.concat(group_dfs, axis=1, keys=GROUPS)
        

    print(output)
    
