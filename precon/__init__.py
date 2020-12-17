# flake8: noqa

from precon.adjustments import jan_adjustment
from precon.aggregation import (
    aggregate,
    aggregate_level,
    aggregate_up_hierarchy,
)
from precon.base_prices import (
    impute_base_prices,
    get_base_prices,
    get_quality_adjusted_prices,
    ffill_shift,

)
from precon.chaining import chain, unchain
from precon.contributions import contributions, contributions_with_double_update
from precon.double_update_methods import jan_adjust_weights, adjust_pre_doublelink
from precon.helpers import (
    reindex_and_fill,
    period_window_fill,
    swap_columns,
    reduce_cols,
    map_headings,
    axis_slice,
    axis_vals_as_frame,
)
from precon.index_methods import calculate_index
from precon.pipelines import index_calculator
from precon.re_reference import (
    set_reference_period,
    set_index_range,
    full_index_to_in_year_indices,
    in_year_indices_to_full_index,
)
from precon.rounding import round_and_adjust
from precon.stat_compilers import (
    get_index_and_growth_stats,
    get_reference_table_stats,
)
from precon.uprate import uprate
from precon.weights import get_weight_shares, reindex_weights_to_indices


__author__ = 'Mitchell Edmunds'
__email__ = 'mitchell.edmunds@ext.ons.gov.uk'
__version__ = '0.7.0-alpha.5'
__all__ = [
    'adjustments',
    'aggregation',
    'chaining',
    'contributions',
    'helpers',
    'index_methods',
    'imputation',
    'pipelines',
    're_reference',
    'rounding',
    'stat_compilers',
    'uprate',
    'weights',
]


__doc__ = """
============================================================
precon: Python functions for Price Index production
============================================================

What is it?
-----------

**precon** is a Python package that provides a suite of speedy, vectorised
functions for implementing common methods in the production of Price Indices.
It aims to provide the high-level building blocks for building statistical
systems at National Statistical Institutes (NSIs) and other research
institutions concerned with creating indices. It has been developed in-house
at the Office for National Statistics (ONS) and aims to become the standard
library for price index production. This can only be achieved with help from
the community, so all contributions are welcome!


Installation
------------

.. code-block:: bash

    pip install precon


Use
---

.. code-block:: python

    import precon


API
---

Many functions in the **precon** package are designed to work with **pandas**
DataFrames or Series that contain only one type of value, with any categorical
or descriptive metadata contained within either the index or columns axis.
Each component of a statistical operation or equation will usually be within
it's own DataFrame, i.e. prices in one Frame and weights in another. When
dealing with time series data, the functions expect one axis to contain
only the datetime index. Where a function accepts more than one input
DataFrame, they will need to share the same index values so that **pandas**
can match up the components that the programmer wants to process together.
Processing values using this matrix format approach allows the functions to
take advantage of powerful **pandas**/**numpy**  vectorised methods.

It is not always necessary that the time series period frequencies match up if
the values in one DataFrame do not change over the given period frequency in
another DataFrame, as the functions will resample to the smaller period
frequency and fill forward the values.

Check the docs for detailed guidance on each function and its parameters.


Features
--------

* Calculate fixed-base price indices using common index methods.
* Combine or aggregate lower-level indices to create higher-level indices.
* Chain fixed-base indices together for a continuous time series.
* Re-reference indices to start from a different time period.
* Calculate contributions to higher-level indices from each of the component indices.
* Impute new base prices over a time series.
* Uprating values by index movements.
* Rounding weight values with adjustment to ensure the sum doesn't change.
* Stat compiler functions to quickly produce common sets of statistics.


.. * Calculate contributions or aggregate up a hierarchy present in a **pandas**
..    MultiIndex.


Dependencies
------------

* `pandas <https://github.com/pandas-dev/pandas>`_
* `NumPy <https://numpy.org/>`_


Contributing to precon
------------------------

See CONTRIBUTING.rst

"""
