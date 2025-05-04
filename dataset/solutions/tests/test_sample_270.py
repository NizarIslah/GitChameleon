import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import plotly.graph_objects as go
from sample_270 import custom_make_subplots


class TestSample270(unittest.TestCase):
    def test_custom_make_subplots(self):
        """Test that custom_make_subplots returns a Figure with correct rows and cols."""
        # Test with 2 rows and 3 columns
        fig = custom_make_subplots(rows=2, cols=3)

        # Check that the return value is a plotly Figure
        self.assertIsInstance(fig, go.Figure)

        # Check that the figure has the correct number of rows and columns
        # Count the number of xaxes and yaxes in the layout
        xaxes = [k for k in fig.layout if k.startswith("xaxis")]
        yaxes = [k for k in fig.layout if k.startswith("yaxis")]
        self.assertEqual(len(xaxes), 2 * 3)
        self.assertEqual(len(yaxes), 2 * 3)

    def test_custom_make_subplots_single(self):
        """Test custom_make_subplots with a single row and column."""
        fig = custom_make_subplots(rows=1, cols=1)

        # Check that the return value is a plotly Figure
        self.assertIsInstance(fig, go.Figure)

        # Check that the figure has the correct number of rows and columns
        xaxes = [k for k in fig.layout if k.startswith("xaxis")]
        yaxes = [k for k in fig.layout if k.startswith("yaxis")]
        self.assertEqual(len(xaxes), 1)
        self.assertEqual(len(yaxes), 1)


if __name__ == "__main__":
    unittest.main()
