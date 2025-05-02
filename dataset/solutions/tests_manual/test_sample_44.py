# Add the parent directory to import sys
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sample_44 import init_clf
from sklearn.ensemble import GradientBoostingClassifier

expected_clf=GradientBoostingClassifier
assert isinstance(init_clf(), expected_clf)