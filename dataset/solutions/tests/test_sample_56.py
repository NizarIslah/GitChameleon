import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_56 import get_grouped_df


class TestGetGroupedDF(unittest.TestCase):
    def test_multiple_groups(self):
        """Test grouping with multiple distinct groups."""
        df = pd.DataFrame({"x": [1, 2, 1, 2], "value": [10, 20, 30, 40]})
        result = get_grouped_df(df)
        expected = pd.DataFrame({"value": [40, 60]}, index=pd.Index([1, 2], name="x"))
        pd.testing.assert_frame_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
