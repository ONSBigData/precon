# -*- coding: utf-8 -*-

from precon.chaining import chain
from precon.re_reference import set_ref_period
from .contributions import contributions_with_double_update


def get_index_and_growth_stats(index, ref_period, double_link=True, prefix=''):
    """Returns the monthly chained index referenced to the given period
    and the index growth as a dictionary.
    
    Parameters
    ----------
    index : Series
        The unchained index.
    ref_period : str
        The reference period as a string. Must work with pandas
        datetime indexing.
        
    Returns
    -------
    dict of {str : Series}
        The monthly index and growth stats for publication.
    """
    chained_index = chain(index, double_link=double_link)
    
    stats=dict()
    stats['idx'] = index.copy()
    stats[f'idx_r{ref_period}'] = set_ref_period(chained_index, ref_period)
    stats['idx_growth'] = chained_index.pct_change(12) * 100
    stats['idx_growth'].dropna(inplace=True)
    
    stats = {prefix + k: v for k, v in stats.items()}
    
    return stats


def get_reference_table_stats(
        sub_indices, headline_index, weight_shares,
        ref_period, double_link, parts_per=100,
        ):
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
    ref_period : str
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
    pub_stats['subidx_unchained'] = sub_indices # TODO: DELETE
    pub_subs_chained = chain(sub_indices, double_link)
    pub_stats['subidx_ch'] = pub_subs_chained
    pub_stats['subidx'] = set_ref_period(pub_subs_chained, ref_period)
    
    pub_stats['weights'] = weight_shares * parts_per
    
    pub_stats.update(get_index_and_growth_stats(headline_index, ref_period))
    
    pub_stats['contributions'] = contributions_with_double_update(
            components=sub_indices,
            weights=weight_shares,
            index=headline_index,
            start_year='2017',
    )
    
    return pub_stats

