import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.axes import Axes

# Add the parent directory to sys.path to import the module to test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_214 import custom_set_axis_labels

data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

ax = custom_set_axis_labels(data)
x_expect = "My X Label"
y_expect = "My Y Label"
assert (
    ax.get_xlabel() == x_expect and ax.get_ylabel() == y_expect
), "Axis labels not set correctly using ax.set()."
