# library: scikit-learn
# version: 1.1
# extra_dependencies: ['numpy==1.23.5']
from sklearn.ensemble import GradientBoostingClassifier


# Initialize the classifier
def init_clf() -> GradientBoostingClassifier:
    classifier = GradientBoostingClassifier(criterion="squared_error")
    return classifier
