# library: librosa
# version: 0.7.0
# extra_dependencies: ['pip==24.0', 'numba==0.46', 'llvmlite==0.30', 'joblib==0.12', 'numpy==1.16.0', 'audioread==2.1.5', 'scipy==1.1.0', 'resampy==0.2.2']
import numpy as np
import librosa
from scipy.spatial.distance import cdist


def compute_dtw(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    dist_matrix = cdist(X.T, Y.T, metric="euclidean")
    return librosa.sequence.dtw(C=dist_matrix, metric="invalid")[0]
