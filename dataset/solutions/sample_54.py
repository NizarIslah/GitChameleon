# library: scikit-learn
# version: 1.2
# extra_dependencies: ['numpy==1.23.5']
from sklearn.metrics.pairwise import manhattan_distances
import numpy as np


def get_pairwise_dist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return manhattan_distances(X, Y)
