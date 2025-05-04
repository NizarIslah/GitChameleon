import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import plotly.graph_objects as go
from sample_270 import custom_make_subplots

import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    fig = custom_make_subplots(2, 2)
    for warn in w:
        assert not issubclass(warn.category, DeprecationWarning), "Deprecated API used!"

num_xaxes = sum(1 for key in fig.layout if key.startswith("xaxis"))
num_yaxes = sum(1 for key in fig.layout if key.startswith("yaxis"))
expect1 = 4
expect2 = 4
assert num_xaxes == expect1
assert num_yaxes == expect2
