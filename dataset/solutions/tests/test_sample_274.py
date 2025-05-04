# Add the parent directory to import sys
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_274 import custom_scatter


class TestCustomScatter(unittest.TestCase):
    def test_custom_scatter_creates_figure(self):
        """Test that custom_scatter returns a plotly Figure object."""
        import plotly.graph_objs as go

        # Test with a specific color
        test_color = "red"
        fig = custom_scatter(test_color)

        # Check that the function returns a Figure object
        self.assertIsInstance(fig, go.Figure)

    def test_custom_scatter_sets_color(self):
        """Test that custom_scatter sets the marker color correctly."""
        import plotly.graph_objs as go

        # Test with a specific color
        test_color = "blue"
        fig = custom_scatter(test_color)

        # Check that the marker color is set correctly
        self.assertEqual(fig.data[0].marker.color, test_color)

    def test_custom_scatter_data_structure(self):
        """Test that custom_scatter creates the correct data structure."""
        import plotly.graph_objs as go

        test_color = "green"
        fig = custom_scatter(test_color)

        # Check that there's exactly one trace
        self.assertEqual(len(fig.data), 1)

        # Check that the trace is a Scatter object
        self.assertIsInstance(fig.data[0], go.Scatter)

        # Check that x and y coordinates are as expected
        self.assertEqual(list(fig.data[0].x), [0])
        self.assertEqual(list(fig.data[0].y), [0])


if __name__ == "__main__":
    unittest.main()
