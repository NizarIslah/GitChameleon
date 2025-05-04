# Add the parent directory to import sys
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sample_44
from sklearn.ensemble import GradientBoostingClassifier

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestInitClf(unittest.TestCase):
    """Test cases for the init_clf function in sample_44.py."""

    def test_init_clf_returns_gradient_boosting_classifier(self):
        """Test that init_clf returns a GradientBoostingClassifier instance."""
        classifier = sample_44.init_clf()
        self.assertIsInstance(classifier, GradientBoostingClassifier)

    def test_init_clf_uses_squared_error_criterion(self):
        """Test that init_clf sets the criterion parameter to 'squared_error'."""
        classifier = sample_44.init_clf()
        self.assertEqual(classifier.criterion, "squared_error")

    def test_init_clf_default_parameters(self):
        """Test that init_clf uses default parameters except for criterion."""
        classifier = sample_44.init_clf()
        default_clf = GradientBoostingClassifier()

        # Check that all parameters except criterion match the defaults
        self.assertEqual(classifier.n_estimators, default_clf.n_estimators)
        self.assertEqual(classifier.learning_rate, default_clf.learning_rate)
        self.assertEqual(classifier.max_depth, default_clf.max_depth)
        self.assertEqual(classifier.min_samples_split, default_clf.min_samples_split)
        self.assertEqual(classifier.min_samples_leaf, default_clf.min_samples_leaf)
        self.assertEqual(classifier.subsample, default_clf.subsample)
        self.assertEqual(classifier.max_features, default_clf.max_features)

    def test_init_clf_returns_new_instance(self):
        """Test that init_clf returns a new instance each time it's called."""
        classifier1 = sample_44.init_clf()
        classifier2 = sample_44.init_clf()

        # Check that they are different instances
        self.assertIsNot(classifier1, classifier2)


if __name__ == "__main__":
    unittest.main()
