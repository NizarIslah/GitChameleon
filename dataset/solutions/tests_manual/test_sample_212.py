import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_212 import custom_barplot

data = pd.DataFrame({"x": ["A", "B", "C"], "y": [5, 10, 15]})
import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    ax = custom_barplot(data)

    warning_messages = [str(warn.message).strip().lower() for warn in w]
    if any("errcolor" in msg or "errwidth" in msg for msg in warning_messages):
        raise AssertionError(
            "errcolor and errwidth should not be used. Use err_kws instead."
        )

    for line in ax.lines:
        if line.get_linewidth() == 2 and line.get_color() == "red":
            break
    else:
        raise AssertionError("Error bars are not set with err_kws correctly.")
