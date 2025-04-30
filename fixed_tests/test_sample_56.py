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
        # - one column 'value'
        # - an index named 'x'
        expected = pd.DataFrame(
            columns=['value'],
            index=pd.Index([], name='x')
        )

        pd.testing.assert_frame_equal(result, expected)

    def test_single_group(self):
        """Test grouping when there's only one group."""
        df = pd.DataFrame({'x': [1, 1, 1], 'value': [10, 20, 30]})
        result = get_grouped_df(df)
        expected = pd.DataFrame(
            {'value': [60]},  # Sum of values: 10 + 20 + 30 = 60
            index=pd.Index([1], name='x')
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_multiple_groups(self):
        """Test grouping with multiple distinct groups."""
        df = pd.DataFrame({'x': [1, 2, 1, 2], 'value': [10, 20, 30, 40]})
        result = get_grouped_df(df)
        expected = pd.DataFrame(
            {'value': [60, 60]},  # Sum of values for group 1: 10 + 30 = 40; for group 2: 20 + 40 = 60
            index=pd.Index([1, 2], name='x')
        )
        pd.testing.assert_frame_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
