import unittest

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Import the function to test

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_326 import modify


class TestSample326(unittest.TestCase):
    def setUp(self):
        """Set up a figure and axes for each test."""
        self.fig, self.ax = plt.subplots()
        # Add some initial ticks to verify they get removed
        self.ax.set_xticks([0, 1, 2])
        self.ax.set_yticks([0, 1, 2])

    def test_modify_removes_ticks(self):
        """Test that the modify function removes all ticks."""
        # Verify initial state has ticks
        self.assertEqual(len(self.ax.get_xticks()), 3)
        self.assertEqual(len(self.ax.get_yticks()), 3)

        # Call the function to test
        modify(self.fig, self.ax)

        # Verify ticks are removed
        self.assertEqual(len(self.ax.get_xticks()), 0)
        self.assertEqual(len(self.ax.get_yticks()), 0)

    def test_modify_preserves_minor_ticks_setting(self):
        """Test that the modify function sets minor=False for ticks."""
        # Add some minor ticks
        self.ax.set_xticks([0.5, 1.5], minor=True)
        self.ax.set_yticks([0.5, 1.5], minor=True)

        # Call the function to test
        modify(self.fig, self.ax)

        # Verify major ticks are removed but minor ticks remain untouched
        self.assertEqual(len(self.ax.get_xticks()), 0)
        self.assertEqual(len(self.ax.get_yticks()), 0)
        self.assertEqual(len(self.ax.get_xticks(minor=True)), 2)
        self.assertEqual(len(self.ax.get_yticks(minor=True)), 2)

    def tearDown(self):
        """Clean up after each test."""
        plt.close(self.fig)


if __name__ == "__main__":
    unittest.main()
