# library: scikit-learn
# version: 1.3
# extra_dependencies: ['numpy==1.23.5']
from sklearn import metrics


def get_scorer_names() -> list:
    return metrics.get_scorer_names()
