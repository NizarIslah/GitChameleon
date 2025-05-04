import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_57 import get_grouped_df


class TestGetGroupedDF(unittest.TestCase):
    def test_basic_grouping(self):
        """Test basic grouping functionality with integer values."""
        # Create a test DataFrame
        df = pd.DataFrame({"x": ["A", "B", "A", "C", "B"], "value": [1, 2, 3, 4, 5]})

        # Get the grouped DataFrame
        result = get_grouped_df(df)

        # Expected result
        expected = pd.DataFrame(
            {"value": [4, 7, 4]}, index=pd.Index(["A", "B", "C"], name="x")
        )

        # Check if the result matches the expected output
        pd.testing.assert_frame_equal(result, expected)

    def test_with_nan_values(self):
        """Test grouping with NaN values in the grouping column."""
        # Create a test DataFrame with NaN values
        df = pd.DataFrame(
            {"x": ["A", "B", np.nan, "A", np.nan], "value": [1, 2, 3, 4, 5]}
        )

        # Get the grouped DataFrame
        result = get_grouped_df(df)

        # Expected result (dropna=False means NaN values are included)
        expected = pd.DataFrame(
            {"value": [5, 2, 8]}, index=pd.Index(["A", "B", np.nan], name="x")
        )

        # Check if the result matches the expected output
        pd.testing.assert_frame_equal(result, expected)

    def test_with_multiple_columns(self):
        """Test grouping with multiple data columns."""
        # Create a test DataFrame with multiple columns
        df = pd.DataFrame(
            {
                "x": ["A", "B", "A", "C", "B"],
                "value1": [1, 2, 3, 4, 5],
                "value2": [10, 20, 30, 40, 50],
            }
        )

        # Get the grouped DataFrame
        result = get_grouped_df(df)

        # Expected result
        expected = pd.DataFrame(
            {"value1": [4, 7, 4], "value2": [40, 70, 40]},
            index=pd.Index(["A", "B", "C"], name="x"),
        )

        # Check if the result matches the expected output
        pd.testing.assert_frame_equal(result, expected)

    def test_with_categorical_data(self):
        """Test grouping with categorical data."""
        # Create a test DataFrame with categorical data
        df = pd.DataFrame(
            {
                "x": pd.Categorical(
                    ["A", "B", "A", "D"], categories=["A", "B", "C", "D"]
                ),
                "value": [1, 2, 3, 4],
            }
        )

        # Get the grouped DataFrame
        result = get_grouped_df(df)

        # Expected result (observed=False means all categories are included)
        expected = pd.DataFrame(
            {"value": [4, 2, 0, 4]},
            index=pd.CategoricalIndex(["A", "B", "C", "D"], name="x"),
        )

        # Check if the result matches the expected output
        pd.testing.assert_frame_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
