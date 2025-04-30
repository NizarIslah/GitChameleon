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
        """Test grouping with an empty DataFrame."""
        # Create an empty DataFrame with the required columns
        df = pd.DataFrame({'x': [], 'value': []})

        # Get the grouped DataFrame
        result = get_grouped_df(df)

        # Expected result is an empty DataFrame with the correct structure:
        # One column 'value' and an index named 'x'
        # Ensure the dtype matches what the grouping function returns (float64)
        expected = pd.DataFrame(
            columns=['value'],
            index=pd.Float64Index([], name='x'),
            dtype=float
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