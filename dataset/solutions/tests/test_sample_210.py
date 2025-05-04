import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_210 import custom_violinplot


class TestCustomViolinplot(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        np.random.seed(42)  # For reproducibility
        self.data = pd.DataFrame(
            {
                "x": np.repeat(["A", "B", "C"], 30),
                "y": np.concatenate(
                    [
                        np.random.normal(0, 1, 30),
                        np.random.normal(2, 1, 30),
                        np.random.normal(4, 1, 30),
                    ]
                ),
            }
        )

    def test_custom_violinplot_returns_axes(self):
        # Test that the function returns a matplotlib Axes object
        plt.figure()  # Create a new figure
        ax = custom_violinplot(self.data)
        self.assertIsInstance(ax, Axes)
        plt.close()  # Close the figure to avoid memory leaks

    def test_custom_violinplot_parameters(self):
        # Test that the function correctly sets the parameters
        # We'll use a mock to verify the parameters
        import seaborn as sns

        original_violinplot = sns.violinplot

        try:
            # Create a mock function to replace sns.violinplot
            calls = []

            def mock_violinplot(*args, **kwargs):
                calls.append(kwargs)
                # Return a mock axes object
                fig, ax = plt.subplots()
                return ax

            # Replace the original function with our mock
            sns.violinplot = mock_violinplot

            # Call the function
            custom_violinplot(self.data)

            # Check that the function was called with the correct parameters
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["x"], "x")
            self.assertEqual(calls[0]["y"], "y")
            self.assertEqual(calls[0]["bw_adjust"], 1.5)
            self.assertIs(calls[0]["data"], self.data)

        finally:
            # Restore the original function
            sns.violinplot = original_violinplot
            plt.close("all")  # Close all figures


if __name__ == "__main__":
    unittest.main()
