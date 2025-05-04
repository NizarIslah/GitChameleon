# Add the parent directory to import sys
import os
import sys
import unittest
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_271 import custom_figure


class TestSample271(unittest.TestCase):
    """Test cases for the custom_figure function in sample_271.py"""

    def test_custom_figure_creation(self):
        """Test that the custom_figure function creates a Figure object"""
        # Test data
        x_data = [1, 2, 3, 4, 5]
        y_data = [10, 11, 12, 13, 14]

        # Create figure
        fig = custom_figure(x_data, y_data)

        # Check that the figure is created
        self.assertIsNotNone(fig)

        # Check that the figure has the correct type
        import plotly.graph_objects as go

        self.assertIsInstance(fig, go.Figure)

    def test_custom_figure_data(self):
        """Test that the custom_figure function adds the correct data to the figure"""
        # Test data
        x_data = [1, 2, 3, 4, 5]
        y_data = [10, 11, 12, 13, 14]

        # Create figure
        fig = custom_figure(x_data, y_data)

        # Check that the figure has one trace
        self.assertEqual(len(fig.data), 1)

        # Check that the trace is a Scatter object
        import plotly.graph_objects as go

        self.assertIsInstance(fig.data[0], go.Scatter)

        # Check that the trace has the correct x and y data
        self.assertEqual(list(fig.data[0].x), x_data)
        self.assertEqual(list(fig.data[0].y), y_data)

    def test_custom_figure_with_empty_data(self):
        """Test that the custom_figure function works with empty data"""
        # Test with empty data
        x_data = []
        y_data = []

        # Create figure
        fig = custom_figure(x_data, y_data)

        # Check that the figure is created
        self.assertIsNotNone(fig)

        # Check that the trace has empty x and y data
        self.assertEqual(list(fig.data[0].x), x_data)
        self.assertEqual(list(fig.data[0].y), y_data)


if __name__ == "__main__":
    unittest.main()
