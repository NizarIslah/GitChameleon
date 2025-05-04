# library: scipy
# version: 1.11.2
# extra_dependencies: []
import scipy.signal.windows as windows
import numpy as np


def compute_lanczos_window(window_size: int) -> np.ndarray:
    return windows.lanczos(window_size)
