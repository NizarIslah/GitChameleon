# library: scikit-learn
# version: 1.1
# extra_dependencies: ['numpy==1.23.5']
from sklearn.impute import SimpleImputer
import numpy as np


def get_imputer(data: np.ndarray) -> SimpleImputer:
    return SimpleImputer()
