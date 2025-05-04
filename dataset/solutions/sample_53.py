# library: scikit-learn
# version: 1.1
# extra_dependencies: ['numpy==1.23.5']
from sklearn.metrics.pairwise import manhattan_distances
import numpy as np


def get_pairwise_dist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    distances = manhattan_distances(X, Y, sum_over_features=False)
    return np.sum(distances, axis=1)
