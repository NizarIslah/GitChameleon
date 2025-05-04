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


x_data = ["A", "B", "C"]
y_data = [10, 15, 7]
output = custom_fig(x_data, y_data)

expect = "v"

assert output.data[0].orientation == expect
