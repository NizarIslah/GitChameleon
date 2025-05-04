import os

# Import the function to test
import sys
import unittest

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "dataset", "solutions")
    )
)
from sample_327 import modify

import numpy as np

fig, ax = plt.subplots()
modify(fig, ax)

assert np.array_equal(ax.get_xticks(), np.array([]))
assert (ax.get_xticks() == np.array([])).all()

assert np.array_equal(ax.get_xticklabels(), np.array([]))
assert (ax.get_xticklabels() == np.array([])).all()
