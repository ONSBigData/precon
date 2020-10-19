"""Functions to help handle the double update present in the CPI."""

import pandas as pd

from precon.helpers import _get_end_year


def jan_adjust_weights(weights, direction='back'):
    """Adjust Feb weights by one month so that weights start in Jan."""
    if direction == 'back':
        return weights.tshift(-1, freq='MS')
    elif direction == 'forward':
        return weights.tshift(1, freq='MS')


def adjust_pre_doublelink(weights, start_year='2017', direction='back'):
    """Jan adjusts only the weights up to the end year."""
    # Double update (Jan & Feb) starts in 2017
    return pd.concat([
        jan_adjust_weights(weights[:_get_end_year(start_year)], direction),
        weights[start_year:],
    ])
