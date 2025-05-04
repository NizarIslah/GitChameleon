import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_63 import combined


class TestCombined(unittest.TestCase):
    def test_empty_dataframes(self):
        # Test with empty DataFrames
        df1 = pd.DataFrame({"A": [], "B": []})
        df2 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        series1 = pd.Series([], name="C")
        series2 = pd.Series([5, 6], name="C")

        result_df, result_series = combined(df1, df2, series1, series2)

        # Check results: values should match, but dtypes may be float
        expected_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=False)

        expected_series = pd.Series([5, 6], name="C")
        pd.testing.assert_series_equal(
            result_series, expected_series, check_dtype=False
        )

    def test_non_empty_dataframes(self):
        # Test merging two non-empty DataFrames
        df1 = pd.DataFrame({"A": [10], "B": [20]})
        df2 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        series1 = pd.Series([7], name="C")
        series2 = pd.Series([5, 6], name="C")

        result_df, result_series = combined(df1, df2, series1, series2)

        expected_df = pd.DataFrame({"A": [10, 1, 2], "B": [20, 3, 4]})
        pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=False)

        expected_series = pd.Series([7, 5, 6], name="C")
        pd.testing.assert_series_equal(
            result_series, expected_series, check_dtype=False
        )


if __name__ == "__main__":
    unittest.main()
