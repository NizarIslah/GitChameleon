import unittest
import pandas as pd
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_56 import get_grouped_df


class TestGetGroupedDF(unittest.TestCase):
    def test_empty_dataframe(self):
        """Test grouping with an empty DataFrame."""
        df = pd.DataFrame({'x': [], 'value': []})
        result = get_grouped_df(df)

        # Expected result is an empty DataFrame with the correct structure
        expected = pd.DataFrame(
            columns=['value'],
            index=pd.Float64Index([], name='x')
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_single_group(self):
        """Test grouping when there's only one group."""
        df = pd.DataFrame({'x': [1, 1, 1], 'value': [10, 20, 30]})
        result = get_grouped_df(df)
        
        # Adjusting expectation to match the sum of [10, 20, 30] = 60
        expected = pd.DataFrame(
            {'value': [60]},
            index=pd.Index([1], name='x')
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_multiple_groups(self):
        """Test grouping with multiple distinct groups."""
        df = pd.DataFrame({'x': [1, 2, 1, 2], 'value': [10, 20, 30, 40]})
        result = get_grouped_df(df)
        
        # For x=1 => sum(10, 30) = 40; for x=2 => sum(20, 40) = 60
        expected = pd.DataFrame(
            {'value': [40, 60]},
            index=pd.Index([1, 2], name='x')
        )
        pd.testing.assert_frame_equal(result, expected)


if __name__ == "__main__":
    unittest.main()