# library: scipy
# version: 1.9.2
# extra_dependencies: []
import scipy.signal.windows as windows
import numpy as np


def compute_lanczos_window(window_size: int) -> np.ndarray:
    window = 2 * np.arange(window_size) / (window_size - 1) - 1
    window = np.sinc(window)
    window = window / np.max(window)
    return window
