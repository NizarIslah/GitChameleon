import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_211 import custom_violinplot

data = pd.DataFrame({"x": ["A", "B", "C"], "y": [5, 10, 15]})

import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    output = custom_violinplot(data)

    warning_messages = [str(warn.message).strip().lower() for warn in w]
    if any("bw" in msg and "deprecated" in msg for msg in warning_messages):
        raise AssertionError(
            "bw parameter should not be used. Use bw_method and bw_adjust instead."
        )

    collections = [
        c for c in output.collections if isinstance(c, plt.Line2D)
    ]  # Extract violin plot lines

    assert output is not None, "Violin plot output should not be None."
