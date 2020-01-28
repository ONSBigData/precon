# -*- coding: utf-8 -*-

from precon.chaining import chain
from precon.re_reference import set_ref_period


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