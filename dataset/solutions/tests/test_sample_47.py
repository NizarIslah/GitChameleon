# Add the parent directory to import sys
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import sample_47
from sklearn.datasets import make_sparse_coded_signal

# Filter deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


class TestGetSignal(unittest.TestCase):
    """Test cases for the get_signal function in sample_47.py."""

    def test_returns_iterable_result(self):
        """Test that get_signal returns an iterable result."""
        # Call the function with sample parameters
        result = sample_47.get_signal(n_samples=5, n_features=10, n_components=8, n_nonzero_coefs=3)
        
        # Convert to tuple if it's not already (handles both tuple and map objects)
        result_tuple = tuple(result)
        
        # Check that we can unpack the result into three components
        data, dictionary, code = result_tuple
        self.assertEqual(len(result_tuple), 3)
    
    def test_returns_correct_output_shapes(self):
        """Test that get_signal returns matrices with the correct shapes."""
        # Define test parameters
        n_samples = 5
        n_features = 10
        n_components = 8
        n_nonzero_coefs = 3
        
        # Call the function and convert to tuple
        result_tuple = tuple(sample_47.get_signal(
            n_samples=n_samples, 
            n_features=n_features, 
            n_components=n_components, 
            n_nonzero_coefs=n_nonzero_coefs
        ))
        
        # Unpack the tuple
        data, dictionary, code = result_tuple
        
        # Check the shapes of the returned matrices
        self.assertEqual(data.shape, (n_samples, n_features))
        self.assertEqual(dictionary.shape, (n_components, n_features))
        self.assertEqual(code.shape, (n_samples, n_components))
    
    def test_works_with_different_parameters(self):
        """Test that get_signal works with different parameter values."""
        # Test cases with different parameter combinations
        test_cases = [
            {"n_samples": 10, "n_features": 15, "n_components": 12, "n_nonzero_coefs": 5},
            {"n_samples": 3, "n_features": 8, "n_components": 6, "n_nonzero_coefs": 2},
            {"n_samples": 20, "n_features": 30, "n_components": 25, "n_nonzero_coefs": 10}
        ]
        
        for params in test_cases:
            # Call the function with the current parameters and convert to tuple
            result_tuple = tuple(sample_47.get_signal(**params))
            
            # Unpack the tuple
            data, dictionary, code = result_tuple
            
            # Check the shapes of the returned matrices
            self.assertEqual(data.shape, (params["n_samples"], params["n_features"]))
            self.assertEqual(dictionary.shape, (params["n_components"], params["n_features"]))
            self.assertEqual(code.shape, (params["n_samples"], params["n_components"]))
    
    def test_handles_minimum_valid_values(self):
        """Test that get_signal works with minimum valid parameter values."""
        # Call the function with minimum valid values
        # n_nonzero_coefs must be less than n_components
        result = sample_47.get_signal(n_samples=1, n_features=1, n_components=2, n_nonzero_coefs=1)
        
        # Convert to tuple if it's not already
        result_tuple = tuple(result)
        
        # Check that the function returns a valid result with 3 components
        self.assertEqual(len(result_tuple), 3)
    
    @patch('sample_47.make_sparse_coded_signal')
    def test_passes_parameters_correctly(self, mock_make_sparse_coded_signal):
        """Test that get_signal passes parameters correctly to make_sparse_coded_signal."""
        # Set up the mock to return a valid result
        mock_result = (
            np.zeros((5, 10)),  # data
            np.zeros((8, 10)),  # dictionary
            np.zeros((5, 8))    # code
        )
        mock_make_sparse_coded_signal.return_value = mock_result
        
        # Call the function with test parameters
        n_samples = 5
        n_features = 10
        n_components = 8
        n_nonzero_coefs = 3
        
        sample_47.get_signal(
            n_samples=n_samples,
            n_features=n_features,
            n_components=n_components,
            n_nonzero_coefs=n_nonzero_coefs
        )
        
        # Verify that make_sparse_coded_signal was called with the correct parameters
        mock_make_sparse_coded_signal.assert_called_once_with(
            n_samples=n_samples,
            n_features=n_features,
            n_components=n_components,
            n_nonzero_coefs=n_nonzero_coefs
        )


if __name__ == '__main__':
    unittest.main()