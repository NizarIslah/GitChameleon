# Add the parent directory to import sys
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_209 import custom_pointplot


class TestCustomPointplot(unittest.TestCase):
    """Test cases for the custom_pointplot function in sample_209.py."""

    def setUp(self):
        """Set up test data."""
        # Create a sample DataFrame for testing
        self.test_data = pd.DataFrame(
            {
                "x": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
                "y": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            }
        )

        # Close any existing plots to avoid interference
        plt.close("all")

    def tearDown(self):
        """Clean up after tests."""
        plt.close("all")

    def test_return_type(self):
        """Test that custom_pointplot returns a matplotlib Axes object."""
        result = custom_pointplot(self.test_data)
        self.assertIsInstance(result, Axes)

    def test_with_different_column_types(self):
        """Test with different data types for x and y columns."""
        # Numeric x and y
        numeric_df = pd.DataFrame({"x": [1, 2, 3, 1, 2, 3], "y": [4, 5, 6, 7, 8, 9]})

        result = custom_pointplot(numeric_df)
        self.assertIsInstance(result, Axes)

        # String x and numeric y
        mixed_df = pd.DataFrame(
            {"x": ["A", "B", "C", "A", "B", "C"], "y": [4, 5, 6, 7, 8, 9]}
        )

        result = custom_pointplot(mixed_df)
        self.assertIsInstance(result, Axes)


if __name__ == "__main__":
    unittest.main()
