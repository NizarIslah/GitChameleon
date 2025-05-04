import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.axes import Axes

# Add the parent directory to sys.path to import the module to test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_214 import custom_set_axis_labels


class TestCustomSetAxisLabels(unittest.TestCase):
    def setUp(self):
        """Set up test data before each test method."""
        # Create a sample DataFrame with x and y columns
        self.test_data = pd.DataFrame(
            {"x": np.random.rand(10), "y": np.random.rand(10)}
        )

    def tearDown(self):
        """Clean up after each test method."""
        # Close any open matplotlib figures
        plt.close("all")

    def test_custom_set_axis_labels_returns_axes(self):
        """Test that the function returns a matplotlib Axes object."""
        result = custom_set_axis_labels(self.test_data)
        self.assertIsInstance(result, Axes)

    def test_custom_set_axis_labels_sets_correct_labels(self):
        """Test that the function sets the correct x and y axis labels."""
        ax = custom_set_axis_labels(self.test_data)

        # Check that the labels are set correctly
        self.assertEqual(ax.get_xlabel(), "My X Label")
        self.assertEqual(ax.get_ylabel(), "My Y Label")

    def test_custom_set_axis_labels_with_empty_dataframe(self):
        """Test the function with an empty DataFrame."""
        empty_df = pd.DataFrame({"x": [], "y": []})
        ax = custom_set_axis_labels(empty_df)

        # Even with empty data, the function should return an Axes object with correct labels
        self.assertIsInstance(ax, Axes)
        self.assertEqual(ax.get_xlabel(), "My X Label")
        self.assertEqual(ax.get_ylabel(), "My Y Label")

    def test_custom_set_axis_labels_plot_content(self):
        """Test that the function creates a scatter plot with the correct data."""
        ax = custom_set_axis_labels(self.test_data)

        # Check that there's at least one collection (scatter points)
        self.assertGreaterEqual(len(ax.collections), 1)

        # Verify the plot is a scatter plot (has PathCollection)
        self.assertTrue(
            any(
                isinstance(item, plt.matplotlib.collections.PathCollection)
                for item in ax.collections
            )
        )


if __name__ == "__main__":
    unittest.main()
