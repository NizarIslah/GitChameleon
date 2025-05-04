import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_65 import combined


class TestCombined(unittest.TestCase):
    def setUp(self):
        # Create test data
        self.df1 = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})

        self.df2 = pd.DataFrame({"A": [4, 5, 6], "B": ["d", "e", "f"]})

        self.series1 = pd.Series([10, 20, 30])
        self.series2 = pd.Series([40, 50, 60])

    def test_combined_dataframes(self):
        """Test that DataFrames are correctly concatenated"""
        result_df, _ = combined(self.df1, self.df2, self.series1, self.series2)

        # Check the result is a DataFrame
        self.assertIsInstance(result_df, pd.DataFrame)

        # Check the shape of the concatenated DataFrame
        self.assertEqual(result_df.shape, (6, 2))

        # Check the values in the concatenated DataFrame
        expected_df = pd.DataFrame(
            {"A": [1, 2, 3, 4, 5, 6], "B": ["a", "b", "c", "d", "e", "f"]}
        )
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_combined_series(self):
        """Test that Series are correctly concatenated"""
        _, result_series = combined(self.df1, self.df2, self.series1, self.series2)

        # Check the result is a Series
        self.assertIsInstance(result_series, pd.Series)

        # Check the length of the concatenated Series
        self.assertEqual(len(result_series), 6)

        # Check the values in the concatenated Series
        expected_series = pd.Series([10, 20, 30, 40, 50, 60])
        pd.testing.assert_series_equal(result_series, expected_series)

    def test_with_empty_dataframes(self):
        """Test with empty DataFrames and Series"""
        empty_df1 = pd.DataFrame(columns=["A", "B"])
        empty_df2 = pd.DataFrame(columns=["A", "B"])
        empty_series1 = pd.Series([], dtype=float)
        empty_series2 = pd.Series([], dtype=float)

        result_df, result_series = combined(
            empty_df1, empty_df2, empty_series1, empty_series2
        )

        # Check results
        self.assertEqual(result_df.shape, (0, 2))
        self.assertEqual(len(result_series), 0)

    def test_with_different_column_names(self):
        """Test with DataFrames that have different column names"""
        df1 = pd.DataFrame({"A": [1, 2], "B": ["a", "b"]})

        df2 = pd.DataFrame({"C": [3, 4], "D": ["c", "d"]})

        result_df, _ = combined(df1, df2, self.series1, self.series2)

        # Check the shape of the result
        self.assertEqual(result_df.shape, (4, 4))

        # Check that all columns are present
        self.assertSetEqual(set(result_df.columns), {"A", "B", "C", "D"})


if __name__ == "__main__":
    unittest.main()
