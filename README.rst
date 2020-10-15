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