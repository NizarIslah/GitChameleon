import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import sys
import os
import seaborn as sns

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_213 import custom_boxenplot


import warnings

data = pd.DataFrame({"x": ["A", "B", "C"], "y": [5, 10, 15]})

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    output = custom_boxenplot(data)

    warning_messages = [str(warn.message).strip().lower() for warn in w]
    if any("scale" in msg and "deprecated" in msg for msg in warning_messages):
        raise AssertionError(
            "scale should not be used in boxenplot. Use width_method instead."
        )

    for artist in output.get_children():
        if hasattr(artist, "get_linestyle") and artist.get_linestyle() in ["-", "--"]:
            break
    else:
        raise AssertionError(
            "Boxen elements are missing, width_method might not be applied."
        )
