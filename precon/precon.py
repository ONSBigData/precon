# -*- coding: utf-8 -*-
"""
@author: Mitchell Edmunds
@title: Prices Economic Functions
"""
# Third party imports
import pandas as pd



### CHAINING FUNCTIONS



### AGGREGATION FUNCTIONS






### RE-REFERENCING FUNCTIONS 



### OTHER FUNCTIONS

def jan_adjustment(indices):
    """Adjust the January values of the index."""
    jans = (indices.index.month == 1)
    first_year = (indices.index.year == indices.index[0].year)
    
    jan_values = indices[jans & ~first_year]
    dec_values = indices[indices.index.month == 12]
    adjusted = jan_values/dec_values.values * 100
    
    adjusted_indices = indices.copy()  
    if isinstance(indices, pd.core.frame.DataFrame):
        adjusted_indices.loc[jans & ~first_year, :] = adjusted
    else:
        adjusted_indices.loc[jans & ~first_year] = adjusted
        
    return adjusted_indices    





    



















def sum_up_hierarchy(df, labels, tree):
    """Expand to the full structure then aggregate sum up hierarchy."""
    df_expanded = expand_full_structure(df, labels)
    return tree.aggregate_sum(df_expanded)








def expand_full_structure(df, headers):
    """
    Reindexes the columns of the given DataFrame to the full
    classification structure. Resulting NaNs filled with zeros.
    
    Parameters
    ----------
    df : DataFrame
        The leaf values, either weights or indices.
    headers : list or Series
        The list of column labels representing the full class structure
        to reindex by.
        
    Returns
    -------
    DataFrame
        The indices or weights reindexed to the full class structure
        given by the headers parameter.
    """
    # Check existing DataFrame columns are in headers.
    if not all(df.columns.isin(headers)):
        raise Exception("Not all columns of given DataFrame are in headers.")
        
    return df.reindex(headers, axis=1).fillna(0)





def prorate(df, factor, exclusions=None):
    """
    Prorates all columns in the DataFrame by the given factor.
    
    Parameters
    ----------
    df : DataFrame
        The values to be prorated.
    factor : float
        The prorating factor to multiply by, between 0 and 1.
    exclusions : iterable
        A list or similar of columns to be excluded from prorating.
    
    Returns
    -------
    DataFrame
        The prorated values.
    """
    if exclusions:
        prorate_cols = ~df.columns.isin(exclusions)
        df.loc[:, prorate_cols] = df.loc[:, prorate_cols].mul(factor, axis=0)
        return df
    else:
        return df.mul(factor, axis=0)


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

    pub_subs_chained = chain(sub_indices, double_link)
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

#def adjust_weights(weights):
#    # Errors > 0.5 means that adjustment is needed
#    errs = abs((weights - weights.round()).sum(1))
#    need_adjusting = errs > 0.5
#    no_of_adjustments = errs[need_adjusting].round(1)
#    
#    adjustments = pd.DataFrame().reindex_like(weights[need_adjusting])
#    for index, row in weights[need_adjusting].iterrows():
#        for _ in range(1, abs(number_of_adjustments[index])):
#    # NEEDS DEV WORK
#
#def apply_rule(row, adjustments):
#    
# Example Data
#62.3651                   0                 14.87        20.3484              2.4166               0


#def get_double_chainlinked_index(indices, weights):
#    """Calculates the double chain-linked index from a set of leaf indices
#    and weights."""
#    return (
#            indices.pipe(precon.jan_adjustment)
#                   .pipe(precon.aggregate, weights=weights)
#                   .pipe(precon.chain, double_link=True)
#    )
#    
#def get_double_chainlinked_subindices(indices):
#    """ """
#    return (
#            indices.pipe(precon.jan_adjustment)
#                   .pipe(precon.chain, double_link=True) 
#    )