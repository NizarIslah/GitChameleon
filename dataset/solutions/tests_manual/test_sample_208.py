import unittest
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# Add the parent directory to the path so we can import the sample_208 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_208 import custom_pointplot

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

    for line in output.lines:
        if line.get_linestyle() != "None":
            raise AssertionError("Linestyle is not set to 'none' as expected.")
        break
