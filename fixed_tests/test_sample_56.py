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
        Ensures that 'x' is float dtype so that the grouped result
        has a Float64Index (matching what the function returns).
        """
        # Create an empty DataFrame specifying float dtype for 'x'
        df = pd.DataFrame({
            'x': pd.Series(dtype=float),
            'value': pd.Series(dtype=float)
        })

        # Get the grouped DataFrame
        result = get_grouped_df(df)

        # Expect an empty DataFrame with the correct float64 index
        expected = pd.DataFrame(
            columns=['value'],
            index=pd.Index([], name='x', dtype='float64')
        )

        pd.testing.assert_frame_equal(result, expected)

    def test_single_group(self):
        """
        Test grouping when there's only one group.
        We expect the sum of values for that single group.
        """
        df = pd.DataFrame({'x': [1, 1, 1], 'value': [10, 20, 30]})
        result = get_grouped_df(df)

        # If the function sums the values, the result for x=1 should be 60
        expected = pd.DataFrame(
            {'value': [60]},
            index=pd.Index([1], name='x')
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_multiple_groups(self):
        """
        Test grouping with multiple distinct groups.
        We expect the sum of values in each group:
        - Group x=1: 10 + 30 = 40
        - Group x=2: 20 + 40 = 60
        """
        df = pd.DataFrame({'x': [1, 2, 1, 2], 'value': [10, 20, 30, 40]})
        result = get_grouped_df(df)

        expected = pd.DataFrame(
            {'value': [40, 60]},
            index=pd.Index([1, 2], name='x')
        )
        pd.testing.assert_frame_equal(result, expected)


if __name__ == "__main__":
    unittest.main()