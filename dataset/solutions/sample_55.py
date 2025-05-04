# library: matplotlib
# version: 3.4.0
# extra_dependencies: []
from matplotlib.colors import *
import numpy as np

cmap = {
    "blue": [[1, 2, 2], [2, 2, 1]],
    "red": [[0, 0, 0], [1, 0, 0]],
    "green": [[0, 0, 0], [1, 0, 0]],
}

cmap_reversed = LinearSegmentedColormap("custom_cmap", cmap).reversed()
