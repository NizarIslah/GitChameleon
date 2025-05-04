# Add the parent directory to import sys
import os
import sys
import unittest

import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_267 import custom_fig

fig = go.Figure()
output = custom_fig(fig)
expect = "paper"

assert output.layout.annotations[0].xref == expect
assert output.layout.annotations[0].yref == expect
