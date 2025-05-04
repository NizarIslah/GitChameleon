# Add the parent directory to import sys
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import sample_45
from sklearn.cross_decomposition import CCA

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestGetCoefShape(unittest.TestCase):
    def test_get_coef_shape_correct_dimensions(self):
        """Test that get_coef_shape returns the correct dimensions."""
        # Create sample data
        X = np.random.rand(100, 4)
        Y = np.random.rand(100, 3)
        cca = CCA(n_components=2)

        # Get the shape of the coefficients
        shape = sample_45.get_coef_shape(cca, X, Y)

        # The function returns (n_features_X, n_features_Y)
        self.assertEqual(shape, (4, 3))

    def test_get_coef_shape_with_different_dimensions(self):
        """Test get_coef_shape with different input dimensions."""
        # Create sample data with different dimensions
        X = np.random.rand(100, 6)
        Y = np.random.rand(100, 5)
        cca = CCA(n_components=3)

        # Get the shape of the coefficients
        shape = sample_45.get_coef_shape(cca, X, Y)

        # The function returns (n_features_X, n_features_Y)
        self.assertEqual(shape, (6, 5))

    def test_get_coef_shape_with_different_n_components(self):
        """Test that get_coef_shape works with different n_components."""
        # Create sample data
        X = np.random.rand(100, 4)
        Y = np.random.rand(100, 3)

        # Test with different n_components
        for n_components in [1, 2, 3]:
            cca = CCA(n_components=n_components)
            shape = sample_45.get_coef_shape(cca, X, Y)

            # The function returns (n_features_X, n_features_Y) regardless of n_components
            self.assertEqual(shape, (4, 3))


if __name__ == "__main__":
    unittest.main()
