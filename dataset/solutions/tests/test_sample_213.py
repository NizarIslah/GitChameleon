import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import sys
import os
import seaborn as sns

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_213 import custom_boxenplot


class TestCustomBoxenplot(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        np.random.seed(42)  # For reproducibility
        self.data = pd.DataFrame(
            {
                "x": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
                "y": np.random.normal(0, 1, 9),  # Random values for y
            }
        )

    def test_return_type(self):
        # Test that the function returns a matplotlib Axes object
        plt.figure()
        ax = custom_boxenplot(self.data)
        self.assertIsInstance(ax, Axes)
        plt.close()

    def test_boxenplot_properties(self):
        # Test that the boxen plot has the expected properties
        plt.figure()
        ax = custom_boxenplot(self.data)

        # Check that there are boxen elements on the axes
        self.assertTrue(len(ax.collections) > 0, "No boxen elements found on the axes")

        # Check that the x-axis has the correct labels
        x_labels = [text.get_text() for text in ax.get_xticklabels()]
        self.assertEqual(sorted(x_labels), ["A", "B", "C"])

        plt.close()

    def test_with_empty_dataframe(self):
        # Test behavior with an empty DataFrame
        empty_df = pd.DataFrame({"x": [], "y": []})
        plt.figure()

        # This should not raise an error, but return an Axes object
        ax = custom_boxenplot(empty_df)
        self.assertIsInstance(ax, Axes)
        plt.close()

    def test_with_single_category(self):
        # Test with a DataFrame containing a single category
        single_cat_df = pd.DataFrame(
            {"x": ["A", "A", "A", "A", "A"], "y": np.random.normal(0, 1, 5)}
        )

        plt.figure()
        ax = custom_boxenplot(single_cat_df)
        self.assertIsInstance(ax, Axes)

        # Check that there is one boxen element
        self.assertTrue(len(ax.collections) > 0, "No boxen elements found on the axes")

        # Check that the x-axis has the correct label
        x_labels = [text.get_text() for text in ax.get_xticklabels()]
        self.assertEqual(x_labels, ["A"])

        plt.close()

    def test_with_numeric_x(self):
        # Test with numeric x values
        numeric_df = pd.DataFrame(
            {"x": [1, 1, 1, 2, 2, 2, 3, 3, 3], "y": np.random.normal(0, 1, 9)}
        )

        plt.figure()
        ax = custom_boxenplot(numeric_df)
        self.assertIsInstance(ax, Axes)

        # Check that there are boxen elements on the axes
        self.assertTrue(len(ax.collections) > 0, "No boxen elements found on the axes")

        plt.close()


if __name__ == "__main__":
    unittest.main()
