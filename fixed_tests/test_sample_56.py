import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_56 import get_grouped_df


class TestGetGroupedDF(unittest.TestCase):
    def test_empty_dataframe(self):
        """
        Test grouping with an empty DataFrame.
        Ensures that both 'x' and 'value' are float dtypes so that the grouped
        result has a Float64Index and float column dtype (matching what
        the function returns).
        """
        # Create an empty DataFrame specifying float dtype for 'x' and 'value'
        df = pd.DataFrame({
            'x': pd.Series(dtype=float),
            'value': pd.Series(dtype=float)
        })

        # Get the grouped DataFrame
        result = get_grouped_df(df)

        # Expect an empty DataFrame with the correct float64 index and float64 column
        expected = pd.DataFrame(
            {'value': pd.Series(dtype=float)},
            index=pd.Float64Index([], name='x')
        )

        pd.testing.assert_frame_equal(result, expected)

    def test_single_group(self):
        """Test grouping when there's only one group."""
        df = pd.DataFrame({'x': [1, 1, 1], 'value': [10, 20, 30]})
        result = get_grouped_df(df)
        expected = pd.DataFrame(
            {'value': [20]},
            index=pd.Index([1], name='x')
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_multiple_groups(self):
        """Test grouping with multiple distinct groups."""
        df = pd.DataFrame({'x': [1, 2, 1, 2], 'value': [10, 20, 30, 40]})
        result = get_grouped_df(df)
        expected = pd.DataFrame(
            {'value': [20, 30]},
            index=pd.Index([1, 2], name='x')
        )
        pd.testing.assert_frame_equal(result, expected)


if __name__ == "__main__":
    unittest.main()