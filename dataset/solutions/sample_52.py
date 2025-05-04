# library: scikit-learn
# version: 1.2
# extra_dependencies: ['numpy==1.23.5']
from sklearn import metrics


def get_scorer_names() -> list:
    return list(metrics.SCORERS.keys())
