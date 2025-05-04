import os

# Import the function to test
import sys
import unittest

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "dataset", "solutions")
    )
)
from sample_327 import modify


class TestSample327(unittest.TestCase):
    """Test cases for the modify function in sample_327.py"""

    def setUp(self):
        """Set up a figure and axes for each test"""
        self.fig, self.ax = plt.subplots()
        # Add some initial ticks to verify they get cleared
        self.ax.set_xticks([0, 1, 2])
        self.ax.set_yticks([0, 1, 2])

    def test_modify_clears_ticks(self):
        """Test that modify function clears both x and y ticks"""
        # Verify initial state has ticks
        self.assertEqual(len(self.ax.get_xticks()), 3)
        self.assertEqual(len(self.ax.get_yticks()), 3)

        # Call the function to test
        modify(self.fig, self.ax)

        # Verify ticks are cleared
        self.assertEqual(len(self.ax.get_xticks()), 0)
        self.assertEqual(len(self.ax.get_yticks()), 0)

    def test_modify_preserves_figure(self):
        """Test that modify function doesn't alter the figure object"""
        # Store original figure properties
        original_figsize = self.fig.get_size_inches()

        # Call the function to test
        modify(self.fig, self.ax)

        # Verify figure properties remain unchanged
        self.assertTrue(np.array_equal(original_figsize, self.fig.get_size_inches()))

    def tearDown(self):
        """Clean up after each test"""
        plt.close(self.fig)


if __name__ == "__main__":
    unittest.main()
