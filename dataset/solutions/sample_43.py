# library: scikit-learn
# version: 1.1
# extra_dependencies: ['numpy==1.23.5']
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np


def get_n_features(clf: GradientBoostingClassifier) -> int:
    n_features_used = clf.n_features_in_
    return n_features_used
