# Add the parent directory to import sys
import os
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import plotly.graph_objects as go
from sample_268 import custom_fig


class TestSample268(unittest.TestCase):
    def test_custom_fig_returns_figure(self):
        """Test that custom_fig returns a plotly Figure object."""
        x_data = [1, 2, 3]
        y_data = [4, 5, 6]
        color_set = "red"

        fig = custom_fig(x_data, y_data, color_set)

        self.assertIsInstance(fig, go.Figure)

    def test_custom_fig_data_values(self):
        """Test that the Figure contains the correct x and y data."""
        x_data = [1, 2, 3]
        y_data = [4, 5, 6]
        color_set = "blue"

        fig = custom_fig(x_data, y_data, color_set)

        # Check that there's exactly one trace
        self.assertEqual(len(fig.data), 1)

        # Check that the trace is a Scatter object
        self.assertIsInstance(fig.data[0], go.Scatter)

        # Check x and y values
        self.assertEqual(list(fig.data[0].x), x_data)
        self.assertEqual(list(fig.data[0].y), y_data)

    def test_custom_fig_error_y_color(self):
        """Test that the error_y color is set correctly."""
        x_data = [1, 2, 3]
        y_data = [4, 5, 6]
        color_set = "green"

        fig = custom_fig(x_data, y_data, color_set)

        # Check that error_y is set and has the correct color
        self.assertIsNotNone(fig.data[0].error_y)
        self.assertEqual(fig.data[0].error_y.color, color_set)


if __name__ == "__main__":
    unittest.main()
