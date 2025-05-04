import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend for testing
# Import the function to test
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_328 import modify


class TestSample328(unittest.TestCase):
    def setUp(self):
        """Set up a figure and axes for each test."""
        self.fig, self.ax = plt.subplots()
        # Add some default ticks to make sure they get cleared
        self.ax.set_xticks([0, 1, 2])
        self.ax.set_yticks([0, 1, 2])

    def tearDown(self):
        """Clean up after each test."""
        plt.close(self.fig)

    def test_modify_clears_ticks(self):
        """Test that the modify function clears both x and y ticks."""
        # Verify initial state has ticks
        self.assertEqual(len(self.ax.get_xticks()), 3)
        self.assertEqual(len(self.ax.get_yticks()), 3)

        # Call the function to test
        modify(self.fig, self.ax)

        # Verify that ticks are cleared
        self.assertEqual(len(self.ax.get_xticks()), 0)
        self.assertEqual(len(self.ax.get_yticks()), 0)

    def test_modify_sets_minor_false(self):
        """Test that the modify function sets minor parameter to False."""
        # Call the function to test
        modify(self.fig, self.ax)

        # Get the tick parameters
        x_params = self.ax.xaxis._minor_tick_kw
        y_params = self.ax.yaxis._minor_tick_kw

        # Check that minor is set to False
        self.assertFalse(self.ax.xaxis.get_minor_locator()())
        self.assertFalse(self.ax.yaxis.get_minor_locator()())


if __name__ == "__main__":
    unittest.main()
