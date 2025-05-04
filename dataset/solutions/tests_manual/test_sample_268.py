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


import plotly.graph_objects as go

x_data = [1, 2, 3]
y_data = [2, 3, 1]
color_set = "rgba(0, 0, 0, 0.5)"

output = custom_fig(x_data, y_data, color_set)

expect = "rgba("
assert output.data[0].error_y.color.startswith(expect)
