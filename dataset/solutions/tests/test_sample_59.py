import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_59 import get_expected_value


class TestSample59(unittest.TestCase):
    def test_get_expected_value_returns_correct_series(self):
        # Create a sample DataFrame (input doesn't matter for this function)
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        # Call the function
        result = get_expected_value(df)

        # Check that the result is a pandas Series
        self.assertIsInstance(result, pd.Series)

        # Check that the Series has the expected values
        expected_values = [98.0, 99.0]
        self.assertTrue(all(result.values == expected_values))

        # Check that the Series has the expected index
        expected_index = ["book1", "book2"]
        self.assertTrue(all(result.index == expected_index))

        # Check that the Series has the expected dtype
        self.assertEqual(result.dtype, np.float64)

    def test_get_expected_value_exact_match(self):
        # Create a sample DataFrame (input doesn't matter for this function)
        df = pd.DataFrame()

        # Call the function
        result = get_expected_value(df)

        # Create the expected Series directly
        expected = pd.Series([98.0, 99.0], index=["book1", "book2"], dtype=np.float64)

        # Check that the result exactly matches the expected Series
        pd.testing.assert_series_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
