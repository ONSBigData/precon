"""Functions to compile publication statistics."""

import pandas as pd
from pandas._typing import Dict, FrameOrSeries

from precon.chaining import chain
from precon.re_reference import set_reference_period
from precon.contributions import contributions, contributions_with_double_update


def get_index_and_growth_stats(
        index: FrameOrSeries,
        reference_period: str,
        double_link: bool = False,
        prefix: str = '',
        ) -> Dict[str, FrameOrSeries]:
    """Returns the monthly chained index referenced to the given period
    and the index growth as a dictionary.

    Parameters
    ----------
    index : Series
        The unchained index.
    reference_period : str
        The reference period as a string. Must work with pandas
        datetime indexing.
    double_link : bool
        Boolean switch for annual chainlinking.
    prefix : str
        A prefix to add to the keys of the output dict.

    Returns
    -------
    dict of {str : Series}
        The monthly index and growth stats for publication.
    """
    stats = dict()

    stats['idx'] = index.copy()

    # Chain the index and reference to given reference period.
    chained_index = chain(index, double_link=double_link)
    stats[f'idx_r{reference_period}'] = set_reference_period(
        chained_index,
        reference_period,
    )

    # Get the annual MoM growth and drop the first year of NaNs
    stats['idx_growth'] = chained_index.pct_change(12) * 100
    stats['idx_growth'].dropna(inplace=True)

    # Adds prefix to the keys of the output dict
    stats = {prefix + k: v for k, v in stats.items()}

    return stats


def get_reference_table_stats(
        sub_indices: pd.DataFrame,
        headline_index: pd.Series,
        weight_shares: pd.DataFrame,
        reference_period: str,
        double_link: bool = False,
        parts_per: int = 100,
        ) -> Dict[str, FrameOrSeries]:
    """
    Produces the unrounded statistics for the reference tables.

    Given the parameters, returns a dictionary of the unrounded
    statistics with the following keys:
        * subidx : Chained and referenced sub-indices.
        * weights : Weights corresponding to sub-indices as {parts_per}
        * idx : Unchained aggregate headline index.
        * idx_r{ref_period} : The chained headline index referenced to
                              {ref_period}
        * idx_growth : The chained headline index year-on-year growth.
        * contributions : The contributions to growth for each
                          sub-index.

    Parameters
    ----------
    sub_indices : DataFrame
        The unchained sub-indices for publication.
    headline_index : Series
        The unchained aggregated index calculated from the sub-indices.
    weight_shares : DataFrame
        The weight shares for the published sub-indices.
    reference_period : str
        The reference period as a string. Must work with pandas
        datetime indexing.
    double_link : bool
        Boolean switch for annual chainlinking. True for double link.
    parts_per : int, default 100
        The multiplier for published weights.

    Returns
    -------
    dict of {str : Series or DataFrame}
        The statistics for the reference tables accessed via the keys.
    """
    pub_stats = dict()

    pub_stats['weights'] = weight_shares * parts_per

    # Get the chained index and growth stats for the sub indices and
    # the headline index.
    pub_stats.update(
        get_index_and_growth_stats(
            sub_indices, reference_period, prefix='sub'
        )
    )
    pub_stats.update(
        get_index_and_growth_stats(headline_index, reference_period)
    )

    # Select the contributions method based on the double_link
    # parameter and then calculate
    if double_link:
        contributions_method = contributions_with_double_update
    else:
        contributions_method = contributions

    pub_stats['contributions'] = contributions_method(
        components=sub_indices,
        weights=weight_shares,
        index=headline_index,
    )

    return pub_stats
