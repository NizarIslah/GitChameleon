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

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestGetNFeatures(unittest.TestCase):
    """Test cases for the get_n_features function in sample_43.py."""

    def test_get_n_features_returns_correct_value(self):
        """Test that get_n_features returns the correct number of features."""
        # Create a classifier and fit it with some data
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([0, 1, 0])
        clf = GradientBoostingClassifier()
        clf.fit(X, y)

        # The number of features should be 3
        result = sample_43.get_n_features(clf)
        self.assertEqual(result, 3)

        # Check that the result is an integer
        self.assertIsInstance(result, int)

    def test_get_n_features_with_different_dimensions(self):
        """Test get_n_features with different feature dimensions."""
        # Create a classifier with more features
        X = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
        y = np.array([0, 1, 0])
        clf = GradientBoostingClassifier()
        clf.fit(X, y)

        # The number of features should be 5
        result = sample_43.get_n_features(clf)
        self.assertEqual(result, 5)

    def test_get_n_features_with_unfitted_classifier(self):
        """Test that get_n_features raises an error with an unfitted classifier."""
        # Create a classifier without fitting it
        clf = GradientBoostingClassifier()

        # Attempting to get n_features should raise an AttributeError
        with self.assertRaises(AttributeError):
            sample_43.get_n_features(clf)

    def test_get_n_features_matches_classifier_attribute(self):
        """Test that get_n_features returns the same value as the classifier's n_features_in_ attribute."""
        # Create and fit a classifier
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        clf = GradientBoostingClassifier()
        clf.fit(X, y)

        # The function should return the same value as the classifier's attribute
        self.assertEqual(sample_43.get_n_features(clf), clf.n_features_in_)


if __name__ == "__main__":
    unittest.main()
