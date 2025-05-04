# Add the parent directory to import sys
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import sample_43
from sklearn.ensemble import GradientBoostingClassifier

X = np.random.rand(100, 20)  # 100 samples, 20 features
y = np.random.randint(0, 2, 100)
clf = GradientBoostingClassifier()
clf.fit(X, y)
expected_n_features = 20
assert get_n_features(clf) == expected_n_features
