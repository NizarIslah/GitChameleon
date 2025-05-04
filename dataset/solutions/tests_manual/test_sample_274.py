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


color = "rgb(255,45,15)"
import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    fig = custom_scatter(color)
    for warn in w:
        assert not issubclass(warn.category, DeprecationWarning), "Deprecated API used!"

scatter_trace = fig.data[0]
marker_color = scatter_trace.marker.color
expect = color
assert marker_color == expect
