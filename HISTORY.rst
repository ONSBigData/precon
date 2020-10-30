.. :changelog:
.. role:: python(code)
   :language: python

History
-------

0.6.2   (2020-10-30)
++++++++++++++++++++

* Bug fix: fixed an issue with the :python:`round_and_adjust` function.

0.6.1   (2020-10-15)
++++++++++++++++++++

* Bug fix: fixed broken API definition.
* Updated README to reflect new installation instructions.



0.6.0   (2020-10-14)
++++++++++++++++++++

* | Added functionality for :python:`base_price_imputation` function accepting
  | the :python:`to_impute` argument.
* | Aggregation function now works with mean or geometric mean depending
  | on :python:`method` argument.
* | The function :python:`calculate_index` introduced offering various
  | different index methods.
* | The :python:`index_calculator` pipeline function offers an end-to-end
  | pipeline for calculating indices with optional base price imputation.


0.5.1   (2020-06-09)
++++++++++++++++++++

* Bug fix in uprate function occuring in Q4 periods.

0.5.0   (2020-06-09)
++++++++++++++++++++

* Removed the prorate function.

0.4.0   (2020-06-05)
++++++++++++++++++++

* Introduced new function uprate and get_uprating_factors for price uprating.

0.3.5   (2020-05-22)
++++++++++++++++++++

* Bug fix

0.3.4   (2020-05-22)
++++++++++++++++++++

* | Introduced improvements to round_and_adjust_weights to work with Series
  | and on any axis of a DataFrame with the axis option.

0.3.3   (2020-05-15)
++++++++++++++++++++

* Rolled back set_first_period in chaining as it introduced a bug.

0.3.2   (2020-05-15)
++++++++++++++++++++

* Bug fix: included flip_axis function in helpers.

0.3.1   (2020-05-15)
++++++++++++++++++++

* Modified aggregation function to work with weight Series and different axes.
* | Changed set_jans in chaining to set_first_period_to_100 to work with 
  | quarterly series.

0.3.0   (2020-05-14)
++++++++++++++++++++

* Added round_and_adjust_weights function in rounding.py.
* | Add set_jans function and improved time series validation in chaining to
  | make functions more robust.

0.2.0   (2020-03-31)
++++++++++++++++++++

* Added create_special_aggregation function.

0.1.1   (2020-03-31)
++++++++++++++++++++

* Fixed bug in importing functions in get_stats module.

0.1.0   (2020-01-27)
++++++++++++++++++++

* First installable version.

