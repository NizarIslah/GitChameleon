# library: scikit-learn
# version: 1.3
# extra_dependencies: ['numpy==1.23.5']
from sklearn.cross_decomposition import CCA
import numpy as np


def get_coef_shape(cca_model: CCA, X: np.ndarray, Y: np.ndarray) -> tuple:
    cca_model.fit(X, Y)
    return cca_model.coef_.shape
