"""A set of unit tests for the helper functions."""
import pandas as pd
from pandas._testing import assert_frame_equal
import pytest

from precon.helpers import axis_vals_as_frame
from test.conftest import create_dataframe


class TestAxisValsAsFrame:
    """Tests for the axis_vals_as_frame function.

    Uses one input dataset to test the following test cases, when:

    * axis = 1
    * axis = 0
    * axis = 1 and converter = lambda x: x.month
    * axis = 0, levels = 1 and conveter = lambda x: x.str.upper()
    """

    @pytest.fixture
    def input_data(self):
        """Return the input data for axis_vals_as_frame."""
        df = create_dataframe(
            [   # A and B cols are set to the index
                ('A', 'B', '2017-01-01', '2017-02-01', '2017-03-01', '2017-04-01'),
                (0, 'foo', None, None, None, None),
                (1, 'bar', None, None, None, None),
                (2, 'baz', None, None, None, None),
                (3, 'qux', None, None, None, None),
            ],
        )
        df = df.set_index(['A', 'B'])
        df.columns = pd.to_datetime(df.columns)
        return df

    @pytest.fixture
    def expout_column_values(self):
        """Return the exp output for axis = 1 case."""
        df = create_dataframe(
            [   # A and B cols are set to the index
                ('A', 'B', '2017-01-01', '2017-02-01', '2017-03-01', '2017-04-01'),
                (0, 'foo', '2017-01-01', '2017-02-01', '2017-03-01', '2017-04-01'),
                (1, 'bar', '2017-01-01', '2017-02-01', '2017-03-01', '2017-04-01'),
                (2, 'baz', '2017-01-01', '2017-02-01', '2017-03-01', '2017-04-01'),
                (3, 'qux', '2017-01-01', '2017-02-01', '2017-03-01', '2017-04-01'),
            ],
        )
        df = df.set_index(['A', 'B'])
        df.columns = pd.to_datetime(df.columns)
        # Convert all the df values to datetime
        return df.apply(pd.to_datetime)

    def test_that_col_values_broadcast_across_all_rows_in_df(
        self,
        input_data,
        expout_column_values,
    ):
        """Unit test for axis = 1 case."""
        # GIVEN a DataFrame and axis argument = 1 for columns
        # WHEN axis_vals_as_frame function returns
        # THEN returns a DataFrame with the column values broadcast across each row.
        true_output = axis_vals_as_frame(input_data, axis=1)

        assert_frame_equal(true_output, expout_column_values)

    @pytest.fixture
    def expout_index_values(self):
        """Return the exp output for axis = 0 case."""
        df = create_dataframe(
            [   # A and B cols are set to the index
                ('A', 'B', '2017-01-01', '2017-02-01', '2017-03-01', '2017-04-01'),
                (0, 'foo', (0, 'foo'), (0, 'foo'), (0, 'foo'), (0, 'foo')),
                (1, 'bar', (1, 'bar'), (1, 'bar'), (1, 'bar'), (1, 'bar')),
                (2, 'baz', (2, 'baz'), (2, 'baz'), (2, 'baz'), (2, 'baz')),
                (3, 'qux', (3, 'qux'), (3, 'qux'), (3, 'qux'), (3, 'qux')),
            ],
        )
        df = df.set_index(['A', 'B'])
        df.columns = pd.to_datetime(df.columns)
        return df

    def test_that_index_values_broadcast_across_all_columns_in_df(
        self,
        input_data,
        expout_index_values,
    ):
        """Unit test for axis = 0 case."""
        # GIVEN a DataFrame and axis argument = 0 for index
        # WHEN axis_vals_as_frame function returns
        # THEN returns a DataFrame with the index values broadcast across each col.
        true_output = axis_vals_as_frame(input_data, axis=0)

        assert_frame_equal(true_output, expout_index_values)

    @pytest.fixture
    def expout_months_from_cols(self):
        """Return exp output for axis=1 and converter=lambda x: x.month case."""
        df = create_dataframe(
            [   # A and B cols are set to the index
                ('A', 'B', '2017-01-01', '2017-02-01', '2017-03-01', '2017-04-01'),
                (0, 'foo', 1, 2, 3, 4),
                (1, 'bar', 1, 2, 3, 4),
                (2, 'baz', 1, 2, 3, 4),
                (3, 'qux', 1, 2, 3, 4),
            ],
        )
        df = df.set_index(['A', 'B'])
        df.columns = pd.to_datetime(df.columns)
        return df

    def test_that_broadcasts_col_vals_across_rows_with_converter(
        self,
        input_data,
        expout_months_from_cols,
    ):
        """Unit test for axis=1 and converter=lambda x: x.month case."""
        # GIVEN a DataFrame, axis = 1 argument and a lambda function to get the months attr
        # WHEN axis_vals_as_frame function returns
        # THEN returns a DataFrame with the months broadcast across each row.
        true_output = axis_vals_as_frame(
            input_data,
            axis=1,
            converter=lambda x: x.month,
        )

        assert_frame_equal(true_output, expout_months_from_cols)

    @pytest.fixture
    def expout_index_level_1_all_caps(self):
        """Return exp output for axis=0, levels=1 and converter = lambda x: x.upper() case."""
        df = create_dataframe(
            [   # A and B cols are set to the index
                ('A', 'B', '2017-01-01', '2017-02-01', '2017-03-01', '2017-04-01'),
                (0, 'foo', 'FOO', 'FOO', 'FOO', 'FOO'),
                (1, 'bar', 'BAR', 'BAR', 'BAR', 'BAR'),
                (2, 'baz', 'BAZ', 'BAZ', 'BAZ', 'BAZ'),
                (3, 'qux', 'QUX', 'QUX', 'QUX', 'QUX'),
            ],
        )
        df = df.set_index(['A', 'B'])
        df.columns = pd.to_datetime(df.columns)
        return df

    def test_that_broadcasts_index_level_1_vals_to_columns_with_converter(
        self,
        input_data,
        expout_index_level_1_all_caps,
    ):
        """Unit test for axis=0, levels=1 and converter = lambda x: x.upper() case."""
        # GIVEN a DataFrame, axis=0, levels=1 and converter=lambda x: x.upper() as args
        # WHEN axis_vals_as_frame function returns
        # THEN returns a DataFrame with level 1 in all caps broadcast across each col.
        true_output = axis_vals_as_frame(
            input_data,
            axis=0,
            levels=1,
            converter=lambda x: x.str.upper(),
        )

        assert_frame_equal(true_output, expout_index_level_1_all_caps)
