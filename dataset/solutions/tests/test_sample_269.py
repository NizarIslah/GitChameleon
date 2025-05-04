import os
import sys
import unittest

import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_269 import custom_fig


class TestSample269(unittest.TestCase):
    def test_custom_fig(self):
        # Create a simple plotly figure
        fig = go.Figure()

        # Apply the custom_fig function
        result = custom_fig(fig)

        # Check that the figure was returned
        self.assertIsInstance(result, go.Figure)

        # Check that the scene camera settings were applied correctly
        self.assertTrue(hasattr(result.layout, "scene"))
        self.assertIsNotNone(result.layout.scene.camera)
        self.assertEqual(result.layout.scene.camera.eye.x, 1.25)
        self.assertEqual(result.layout.scene.camera.eye.y, 1.25)
        self.assertEqual(result.layout.scene.camera.eye.z, 1.25)

    def test_custom_fig_preserves_other_settings(self):
        # Create a figure with some existing settings
        fig = go.Figure()
        fig.update_layout(title="Test Figure", width=800, height=600)

        # Apply the custom_fig function
        result = custom_fig(fig)

        # Check that the original settings are preserved
        self.assertEqual(result.layout.title.text, "Test Figure")
        self.assertEqual(result.layout.width, 800)
        self.assertEqual(result.layout.height, 600)

        # And the new camera settings are applied
        self.assertTrue(hasattr(result.layout, "scene"))
        self.assertIsNotNone(result.layout.scene.camera)
        self.assertEqual(result.layout.scene.camera.eye.x, 1.25)
        self.assertEqual(result.layout.scene.camera.eye.y, 1.25)
        self.assertEqual(result.layout.scene.camera.eye.z, 1.25)


if __name__ == "__main__":
    unittest.main()
