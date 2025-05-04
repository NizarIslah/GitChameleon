import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_212 import custom_barplot


class TestCustomBarplot(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        np.random.seed(42)  # For reproducibility
        self.data = pd.DataFrame(
            {"x": ["A", "B", "C", "D", "E"], "y": [25, 18, 30, 15, 22]}
        )

    def test_return_type(self):
        # Test that the function returns a matplotlib Axes object
        plt.figure()
        ax = custom_barplot(self.data)
        self.assertIsInstance(ax, Axes)
        plt.close()

    def test_barplot_properties(self):
        # Test that the bar plot has the expected properties
        plt.figure()
        ax = custom_barplot(self.data)

        # Check that there are bars on the axes
        self.assertTrue(len(ax.patches) > 0, "No bars found on the axes")

        # Check that the x-axis has the correct labels
        x_labels = [text.get_text() for text in ax.get_xticklabels()]
        self.assertEqual(sorted(x_labels), ["A", "B", "C", "D", "E"])

        # Check that the error bars have the specified properties
        # The error bars are stored in the errorbar container
        for container in ax.containers:
            if hasattr(container, "errorbar"):
                errorbar = container.errorbar
                if errorbar:
                    # Check the color and linewidth of error bars
                    self.assertEqual(errorbar.lines[0].get_color(), "red")
                    self.assertEqual(errorbar.lines[0].get_linewidth(), 2)

        plt.close()

    def test_with_empty_dataframe(self):
        # Test behavior with an empty DataFrame
        empty_df = pd.DataFrame({"x": [], "y": []})
        plt.figure()

        # This should not raise an error, but return an Axes object
        ax = custom_barplot(empty_df)
        self.assertIsInstance(ax, Axes)
        plt.close()

    def test_with_single_category(self):
        # Test with a DataFrame containing a single category
        single_cat_df = pd.DataFrame({"x": ["A"], "y": [10]})

        plt.figure()
        ax = custom_barplot(single_cat_df)
        self.assertIsInstance(ax, Axes)

        # Check that there is one bar
        self.assertEqual(len(ax.patches), 1, "Expected one bar in the plot")

        # Check that the x-axis has the correct label
        x_labels = [text.get_text() for text in ax.get_xticklabels()]
        self.assertEqual(x_labels, ["A"])

        plt.close()

    def test_with_numeric_x(self):
        # Test with numeric x values
        numeric_df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 15, 20, 25, 30]})

        plt.figure()
        ax = custom_barplot(numeric_df)
        self.assertIsInstance(ax, Axes)

        # Check that there are bars on the axes
        self.assertEqual(len(ax.patches), 5, "Expected five bars in the plot")

        plt.close()


if __name__ == "__main__":
    unittest.main()
