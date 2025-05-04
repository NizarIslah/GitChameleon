import os
import sys
import unittest

import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_269 import custom_fig


fig = go.Figure(
    data=[go.Scatter3d(x=[1, 2, 3], y=[1, 2, 3], z=[1, 2, 3], mode="markers")]
)
expect = 1.25
output = custom_fig(fig)
assert output.layout.scene.camera.eye.x == expect
assert output.layout.scene.camera.eye.y == expect
assert output.layout.scene.camera.eye.z == expect
