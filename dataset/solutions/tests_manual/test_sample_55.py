import unittest
import sys
import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Add the parent directory to sys.path to import the module to test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sample_55


expected_cmap_reversed = {
    "blue": [(-1.0, 1, 2), (0.0, 2, 2)],
    "red": [(0.0, 0, 0), (1.0, 0, 0)],
    "green": [(0.0, 0, 0), (1.0, 0, 0)],
}

reversed_cmap_dict = cmap_reversed._segmentdata

assert reversed_cmap_dict == expected_cmap_reversed
