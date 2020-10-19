"""
A set of functions for price uprating.

* uprate - Returns uprated expenditures given indices and base month.
* get_uprating_factor - Return the expenditure uprating factors for a
        given base month.
"""


def uprate(expenditures, indices, base_month, method=None):
    """
    Returns uprated expenditures given indices and base month.

    Parameters
    ----------
    expenditures : DataFrame
        The expenditure totals to be uprated.
    indices : DataFrame
        The corresponding price indices for expenditure totals.
    base_month : int
        The number of the calendar month.
    method : {'backfill', 'bfill', None} or callable, default None
        The method to fill by:
        * backfill / bfill: use NEXT valid observation to fill gap.
        * callable : callable with input DataFrame and returning a
            DataFrame

    Returns
    -------
    DataFrame
        The uprated expenditures.
    """
    uprating_factor = get_uprating_factor(indices, base_month, method)

    return expenditures * uprating_factor


def get_uprating_factor(indices, base_month, method=None):
    """
    Return the expenditure uprating factors for a given base month.

    Parameters
    ----------
    indices : DataFrame
        The corresponding price indices for expenditure totals.
    base_month : int
        The number of the calendar month (either Jan or Dec).
    method : {'backfill', 'bfill', None} or callable, default None
        The method to fill by:
        * backfill / bfill: use NEXT valid observation to fill gap
        * callable : callable with input DataFrame and returning a
            DataFrame

    Returns
    -------
    DataFrame
        The uprating factors.
    """
    if base_month not in [1, 12]:
        raise ValueError(
            "Base month can currently only be 1 or 12. If you need "
            "other base months raise a new feature request.")

    annual_mean = indices.resample('AS').mean()
    month_values = indices[indices.index.month == base_month]

    if base_month == 12:
        month_values = (
            month_values.tshift(1, freq='MS')
            .reindex_like(annual_mean)
        )

    uprating_factor = month_values.div(annual_mean.shift(2))

    return _apply_uprating_fill_method(uprating_factor, method)


def _apply_uprating_fill_method(uprating_factor, method):
    """Applys the uprating factor fill method."""
    if method is not None:
        if method == 'bfill' or method == 'backfill':
            uprating_factor = uprating_factor.bfill()

        elif callable(method):
            uprating_factor = method(uprating_factor)

        else:
            raise ValueError(
                "method must be either 'bfill', or a callable, "
                f"'{method}' was supplied"
            )

    return uprating_factor
