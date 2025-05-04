# Add the parent directory to import sys
import os
import sys
import unittest

import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_267 import custom_fig


class TestSample267(unittest.TestCase):
    def test_custom_fig_adds_annotation(self):
        # Create a basic figure
        fig = go.Figure()

        # Apply the custom_fig function
        result = custom_fig(fig)

        # Verify that an annotation was added
        self.assertEqual(len(result.layout.annotations), 1)

        # Verify the annotation properties
        annotation = result.layout.annotations[0]
        self.assertEqual(annotation.x, 0.5)
        self.assertEqual(annotation.y, 0.5)
        self.assertEqual(annotation.text, "Example Annotation")
        self.assertEqual(annotation.xref, "paper")
        self.assertEqual(annotation.yref, "paper")
        self.assertEqual(annotation.showarrow, False)

    def test_custom_fig_returns_figure_object(self):
        # Create a basic figure
        fig = go.Figure()

        # Apply the custom_fig function
        result = custom_fig(fig)

        # Verify that the result is a Figure object
        self.assertIsInstance(result, go.Figure)

        # Verify that the function returns the same figure object (modified)
        self.assertIs(result, fig)


if __name__ == "__main__":
    unittest.main()
