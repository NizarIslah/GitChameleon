import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_211 import custom_violinplot


class TestCustomViolinplot(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        np.random.seed(42)  # For reproducibility
        self.data = pd.DataFrame(
            {
                "x": ["A"] * 30 + ["B"] * 30 + ["C"] * 30,
                "y": np.concatenate(
                    [
                        np.random.normal(0, 1, 30),
                        np.random.normal(2, 1, 30),
                        np.random.normal(4, 1, 30),
                    ]
                ),
            }
        )

    def test_return_type(self):
        # Test that the function returns a matplotlib Axes object
        plt.figure()
        ax = custom_violinplot(self.data)
        self.assertIsInstance(ax, Axes)
        plt.close()

    def test_violinplot_properties(self):
        # Test that the violin plot has the expected properties
        plt.figure()
        ax = custom_violinplot(self.data)

        # Check that there are violin plots on the axes
        self.assertTrue(len(ax.collections) > 0, "No violin plots found on the axes")

        # Check that the x-axis has the correct labels
        x_labels = [text.get_text() for text in ax.get_xticklabels()]
        self.assertEqual(sorted(x_labels), ["A", "B", "C"])

        plt.close()

    def test_with_empty_dataframe(self):
        # Test behavior with an empty DataFrame
        empty_df = pd.DataFrame({"x": [], "y": []})
        plt.figure()

        # This should not raise an error, but return an Axes object
        ax = custom_violinplot(empty_df)
        self.assertIsInstance(ax, Axes)
        plt.close()

    def test_with_single_category(self):
        # Test with a DataFrame containing a single category
        single_cat_df = pd.DataFrame({"x": ["A"] * 10, "y": np.random.normal(0, 1, 10)})

        plt.figure()
        ax = custom_violinplot(single_cat_df)
        self.assertIsInstance(ax, Axes)

        # Check that there is one violin plot
        self.assertTrue(len(ax.collections) > 0, "No violin plots found on the axes")

        # Check that the x-axis has the correct label
        x_labels = [text.get_text() for text in ax.get_xticklabels()]
        self.assertEqual(x_labels, ["A"])

        plt.close()


if __name__ == "__main__":
    unittest.main()
