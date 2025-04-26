import unittest
import sys
import os
import warnings
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.cross_decomposition import CCA
import numpy as np
import sample_45

# Filter deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


class TestGetCoefShape(unittest.TestCase):
    """Test cases for the get_coef_shape function in sample_45.py."""

    def test_get_coef_shape_returns_tuple(self):
        """Test that get_coef_shape returns a tuple."""
        # Create sample data
        X = np.random.rand(100, 4)
        Y = np.random.rand(100, 3)
        cca = CCA(n_components=2)
        
        # Get the shape of the coefficients
        shape = sample_45.get_coef_shape(cca, X, Y)
        
        # Check that the result is a tuple
        self.assertIsInstance(shape, tuple)
    
    def test_get_coef_shape_correct_dimensions(self):
        """Test that get_coef_shape returns the correct dimensions."""
        # Create sample data
        X = np.random.rand(100, 4)
        Y = np.random.rand(100, 3)
        cca = CCA(n_components=2)
        
        # Get the shape of the coefficients
        shape = sample_45.get_coef_shape(cca, X, Y)
        
        # The shape should be (n_features_Y, n_features_X)
        self.assertEqual(shape, (3, 4))
    
    def test_get_coef_shape_with_different_dimensions(self):
        """Test get_coef_shape with different input dimensions."""
        # Create sample data with different dimensions
        X = np.random.rand(100, 6)
        Y = np.random.rand(100, 5)
        cca = CCA(n_components=3)
        
        # Get the shape of the coefficients
        shape = sample_45.get_coef_shape(cca, X, Y)
        
        # The shape should be (n_features_Y, n_features_X)
        self.assertEqual(shape, (5, 6))
    
    def test_get_coef_shape_with_different_n_components(self):
        """Test that get_coef_shape works with different n_components."""
        # Create sample data
        X = np.random.rand(100, 4)
        Y = np.random.rand(100, 3)
        
        # Test with different n_components
        for n_components in [1, 2, 3]:
            cca = CCA(n_components=n_components)
            shape = sample_45.get_coef_shape(cca, X, Y)
            
            # The shape should be (n_features_Y, n_features_X) regardless of n_components
            self.assertEqual(shape, (3, 4))
    
    def test_get_coef_shape_calls_fit(self):
        """Test that get_coef_shape calls the fit method of the CCA model."""
        # Create sample data
        X = np.random.rand(100, 4)
        Y = np.random.rand(100, 3)
        
        # Create a mock CCA model
        mock_cca = MagicMock(spec=CCA)
        mock_cca.fit.return_value = mock_cca
        mock_cca.coef_ = np.zeros((3, 4))
        
        # Call the function with the mock
        sample_45.get_coef_shape(mock_cca, X, Y)
        
        # Verify that fit was called with the correct arguments
        mock_cca.fit.assert_called_once_with(X, Y)


if __name__ == '__main__':
    unittest.main()