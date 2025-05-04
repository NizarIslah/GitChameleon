import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_58 import get_expected_value


class TestSample58(unittest.TestCase):
    def test_get_expected_value_returns_correct_series(self):
        # Create a dummy DataFrame as input (the function doesn't use it)
        dummy_df = pd.DataFrame()

        # Call the function
        result = get_expected_value(dummy_df)

        # Check that the result is a pandas Series
        self.assertIsInstance(result, pd.Series)

        # Check that the Series has the expected values
        expected_values = [11.1, 12.2]
        pd.testing.assert_series_equal(
            result,
            pd.Series(expected_values, index=["book1", "book2"], dtype=np.float64),
        )

    def test_get_expected_value_has_correct_index(self):
        dummy_df = pd.DataFrame()
        result = get_expected_value(dummy_df)

        # Check that the Series has the expected index
        self.assertEqual(list(result.index), ["book1", "book2"])

    def test_get_expected_value_has_correct_dtype(self):
        dummy_df = pd.DataFrame()
        result = get_expected_value(dummy_df)

        # Check that the Series has the expected dtype
        self.assertEqual(result.dtype, np.float64)

    def test_get_expected_value_has_correct_values(self):
        dummy_df = pd.DataFrame()
        result = get_expected_value(dummy_df)

        # Check individual values
        self.assertEqual(result["book1"], 11.1)
        self.assertEqual(result["book2"], 12.2)


if __name__ == "__main__":
    unittest.main()
