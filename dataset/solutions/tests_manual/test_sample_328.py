import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend for testing
# Import the function to test
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_328 import modify

import numpy as np

fig, ax = plt.subplots()
modify(fig, ax)

assert np.array_equal(ax.get_xticks(), np.array([]))
assert (ax.get_xticks() == np.array([])).all()

assert np.array_equal(ax.get_xticklabels(), np.array([]))
assert (ax.get_xticklabels() == np.array([])).all()
