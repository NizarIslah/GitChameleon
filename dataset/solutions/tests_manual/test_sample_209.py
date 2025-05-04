# Add the parent directory to import sys
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_209 import custom_pointplot

data = pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 15, 13, 17]})

import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    output = custom_pointplot(data)

    warning_messages = [
        word for warn in w for word in str(warn.message).strip().lower().split()
    ]

    if any("dataframegroupby.apply" in msg for msg in warning_messages):
        pass
    elif any("deprecated" in msg and "removed" in msg for msg in warning_messages):
        raise AssertionError("Expected deprecation warning was not raised.")

    found_correct_linewidth = False
    for line in output.lines:
        linewidth = line.get_linewidth()
        if linewidth == 2:
            found_correct_linewidth = True
            break

    if not found_correct_linewidth:
        raise AssertionError("Error bar linewidth is not set to 2 as expected.")
