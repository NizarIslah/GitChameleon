# Add the parent directory to import sys
import os
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import plotly.graph_objects as go
from sample_266 import custom_fig


class TestSample266(unittest.TestCase):
    def test_custom_fig_returns_figure(self):
        """Test that custom_fig returns a plotly Figure object."""
        x_data = ["A", "B", "C"]
        y_data = [1, 2, 3]
        fig = custom_fig(x_data, y_data)
        self.assertIsInstance(fig, go.Figure)

    def test_custom_fig_has_bar_trace(self):
        """Test that the figure contains a Bar trace."""
        x_data = ["A", "B", "C"]
        y_data = [1, 2, 3]
        fig = custom_fig(x_data, y_data)

        # Check that there's at least one trace
        self.assertGreater(len(fig.data), 0)

        # Check that the first trace is a Bar
        self.assertIsInstance(fig.data[0], go.Bar)

    def test_custom_fig_data_values(self):
        """Test that the figure contains the correct data values."""
        x_data = ["A", "B", "C"]
        y_data = [1, 2, 3]
        fig = custom_fig(x_data, y_data)

        # Check that the x and y data match what was provided
        self.assertEqual(list(fig.data[0].x), x_data)
        self.assertEqual(list(fig.data[0].y), y_data)

    def test_custom_fig_orientation(self):
        """Test that the bar orientation is vertical."""
        x_data = ["A", "B", "C"]
        y_data = [1, 2, 3]
        fig = custom_fig(x_data, y_data)

        # Check that the orientation is vertical
        self.assertEqual(fig.data[0].orientation, "v")


if __name__ == "__main__":
    unittest.main()
