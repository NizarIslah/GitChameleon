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


data = pd.DataFrame({"x": ["A", "B", "C"], "y": [5, 10, 15]})
import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    output = custom_violinplot(data)

    warning_messages = [str(warn.message).strip().lower() for warn in w]
    if any("bw" in msg and "deprecated" in msg for msg in warning_messages):
        raise AssertionError(
            "bw parameter should not be used. Use bw_method and bw_adjust instead."
        )

    for collection in output.collections:
        if hasattr(collection, "get_paths"):
            assert sns.violinplot.__defaults__[0] == 1.5
