# library: seaborn
# version: 0.12.0
# extra_dependencies: ['scipy==1.8.0']
import numpy as np


def custom_iqr(data: np.ndarray) -> float:
    from scipy.stats import iqr

    return iqr(data)
