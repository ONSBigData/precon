# -*- coding: utf-8 -*-
"""
@author: Mitchell Edmunds
@title: Prices Economic Functions
"""
# Third party imports
import pandas as pd



### CHAINING FUNCTIONS



### AGGREGATION FUNCTIONS



def aggregate(indices, weights):
    """
    Takes a set of unchained indices and corresponding weights with matching
    time series index, and produces the weighted aggregate unchained index.
    """
    weight_shares = get_weight_shares(weights)
    weight_shares = reindex_weights_to_indices(weight_shares, indices)
    
    return indices.mul(weight_shares).sum(axis=1)


### RE-REFERENCING FUNCTIONS 
def set_ref_period(df, period):
    """ A function to re-reference an index series on a given period."""
    base_mean = df[period].mean() # Mean of values at the base period.
    reref = df.div(base_mean) * 100 # Normalise
    
    # Fill NaNs from division with zeros
    reref.fillna(0, inplace=True)
       
    return reref


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


def _select_month_reindex(indices, month):
    """Subsets indices for given month then reindexes to original size."""
    select_month = indices[indices.index.month == month]
    return select_month.reindex(index=indices.index)

def contributions(components, weights, index, double_update=False):
    """
    Calculates the growth contributions of the index components to 
    overall growth of a double annual chainlinked index.
    
    Parameters
    ----------
    components : DataFrame
        The index components.
    weights : DataFrame
        The weights for the index components.
    index : Series
        The unchained aggregated index.
        
    Returns
    -------
    DataFrame
        The contributions of each component to overall index growth.        
    """
    weights = get_weight_shares(weights)
    weights = reindex_weights_to_indices(weights, components)
    
    ### Set equation components
    
    # Set the January values to 100 for the unchained index
    unchained_index = index.copy()
    unchained_index[unchained_index.index.month == 1] = 100
    
    # Set components as Ic_y and set Jan = 100
    Ic_y = components.copy()
    Ic_y[Ic_y.index.month == 1] = 100
    
    # Shift weights to start in Jan rather than Feb
    # _py suffix denotes "previous year" or t-12
    if not double_update:
        w1 = weights.shift(12)
        w2 = w3 = weights
    else:
        w1 = _select_month_reindex(weights, 2).shift(-1).ffill().shift(12)
        w2 = _select_month_reindex(weights, 1).ffill()
        w3 = _select_month_reindex(weights, 2).shift(-1).ffill()
    
    # Take Jan values from previous Dec=100 index (without Jans set to 100)
    # Dec values can be taken from either, previous year so shift by 12
    Ic_dec = _select_month_reindex(components, 12).bfill().shift(12)
    Ic_jan = _select_month_reindex(components, 1).ffill()
    Ic_py = Ic_y.shift(12)
    
    IA_dec = _select_month_reindex(index, 12).bfill().shift(12)
    IA_jan = _select_month_reindex(index, 1).ffill()
    IA_py = unchained_index.shift(12)
    
    ### Calculate contributions
    
    # Calculate each component of the contributions to annual change equation
    # for double-linked index. Zeros are positional args i.e. axis=0
    comp1 = w1.mul((Ic_dec-Ic_py).div(IA_py, 0), 0) * 100
    comp2 = (
                w2.mul((Ic_jan-100).div(IA_py, 0), 0)
                   .mul(IA_dec, 0)
    )
    comp3 = (
                w3.mul((Ic_y-100).div(IA_py, 0), 0)
                   .mul(IA_jan/100, 0)
                   .mul(IA_dec, 0)
    )
    
    contributions = comp1 + comp2 + comp3
    
    return contributions.dropna()


def _selector(slicer, *args):
    """Selects the given args with the given slice and returns a tuple."""
    sliced_args = []
    for arg in args:
        sliced_args.append(arg.loc[slicer])
    return tuple(sliced_args)


def contributions_with_double_update(
        components, weights, index, start_year
        ):
    """Returns the contributions for time periods that include both
    single update and double update.
    
    Parameters
    ----------
    components : DataFrame
        The index components.
    weights : DataFrame
        The weights for the index components.
    index : Series
        The unchained aggregated index.
    start_year : str
        The year the double weights update began.
        
    Returns
    -------
    DataFrame
        The contributions of each component to overall index growth.
    """
    end_year = str(int(start_year) - 1)
    # Slice up to end_year, and from end_year onwards since contributions
    # start the year after the first year (12 month lag)
    pre_slice = slice(end_year)
    post_slice = slice(end_year, None)
    
    pre_args = _selector(pre_slice, components, weights, index)
    post_args = _selector(post_slice, components, weights, index)
    
    # Calculate contributions pre double-update, and post double-update
    # Then concatenate them together
    contributions_pre = contributions(*pre_args)
    contributions_post = contributions(*post_args, double_update=True)
    return pd.concat([contributions_pre, contributions_post])
    

def set_index_range(df, start=None, end=None):
    """Edit this function to take a start and an end year."""   
    if start:
        if start not in df.index.year.unique().astype(str).to_numpy():
            raise Exception("start needs to be a year in the index, as a string.")
    
    subset = df.loc[start:end]
    
    if type(df) == type(pd.Series()):
        subset.iloc[0] = 100  
    else:
        subset.iloc[0, :] = 100
    return subset


def prices_to_index(prices): #TODO add base
    """Returns an index from a DataFrame or Series of aggregated prices."""
    if isinstance(prices, pd.Series):
        return prices/prices.iloc[0] * 100
    else:
        return prices.div(prices.iloc[0, :]) * 100


def full_index_to_in_year_indices(full_index):
    """Break index down into Jan-Jan+1 segments, rebased at 100 each year.
    Returns a dictionary of the in-year indices with years as keys.
    """
    # Get a list of the years present in the index
    index_years = full_index.resample('A').first().index.to_timestamp()

    # Set the index range for each year (base=100, runs through to Jan+1)
    pi_yearly = {}
    for year in index_years:
        end = year + pd.DateOffset(years=1)
        pi_yearly[year.year] = set_index_range(full_index, start=year, end=end)
    
    return pi_yearly


def in_year_indices_to_full_index(in_year_indices):
    """Converts a dictionary of in-year indices into a single 
    unchained index.
    """
    full_index = pd.concat(in_year_indices).fillna(0).droplevel(0)
    
    # Take out any Jan=100 that are not the first in the full index
    condition = ((full_index.index.month == 1)
                 & (full_index.index.year != full_index.index[0].year)
                 & (full_index == 100).any(axis=1))
    full_index = full_index[~condition]
    
    return full_index








def get_pub_stats(indices, ref_period):
    """Given a set of monthly indices and a reference period, output
    a set of publication ready statistics.
    
    These statistics are the re-referenced monthly indices, the quarterly 
    resampled indices and the growth rates from the same period in the 
    previous year for both quarterly and monthly.
    """
    stats = dict()
    idx_m = 'idx_m_r{}'.format(ref_period)
    idx_q = 'idx_q_r{}'.format(ref_period)
    
    stats[idx_m] = set_ref_period(indices, period=ref_period)
    stats[idx_q] = stats[idx_m].resample('Q').mean()
    
    stats['yoy_growth_q'] = (stats[idx_q].pct_change(4)*100)
    stats['yoy_growth_m'] = (stats[idx_m].pct_change(12)*100)
    
    for key in stats.keys():
        stats[key] = stats[key].dropna(1, 'all').dropna(0, 'all')
    
    return stats








def sum_up_hierarchy(df, labels, tree):
    """Expand to the full structure then aggregate sum up hierarchy."""
    df_expanded = expand_full_structure(df, labels)
    return tree.aggregate_sum(df_expanded)


def reaggregate_index(indices, weights, subs):
    """Returns a reaggregated index after substituting.
    
    Parameters
    ----------
    indices : DataFrame
        A set of unchained component indices.
    weights : DataFrame
        A set of component weights.
    subs : dict of {str : Series}
        A dictionary of components to be substituted where key is
        column name and value is the Series to substitute.
        
    Returns
    -------
    Series
        The unchained aggregate index after substituting.
    """
    subbed_indices = substitute_indices(indices, subs)
    return aggregate(subbed_indices, weights)


def substitute_indices(indices, subs):
    """Substitutes the indices at given keys for given Series.
    
    Parameters
    ----------
    indices : DataFrame
        A set of unchained component indices.
    subs : dict of {str : Series}
        A dictionary of components to be substituted where key is
        column name and value is the Series to substitute.
        
    Returns
    -------
    DataFrame
        The original indices including the substitutes.
    """
    subbed_indices = indices.copy()
    for key, sub in subs.items():
        subbed_indices[key] = sub
    
    return subbed_indices


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