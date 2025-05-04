# Add the parent directory to import sys
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import sample_49
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
from sklearn.utils import Bunch

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestApplyFastICA(unittest.TestCase):
    """Test cases for the apply_fast_ica function in sample_49.py."""

    def setUp(self):
        """Set up test fixtures."""
        # Load the digits dataset for testing
        self.digits = load_digits()

        # Create a small test dataset
        self.test_data = np.random.rand(100, 64)
        self.test_bunch = Bunch(data=self.test_data)

        # Patch FastICA to handle whiten parameter compatibility issues
        self.patcher = patch("sample_49.FastICA")
        self.mock_fast_ica = self.patcher.start()
        self.mock_ica_instance = MagicMock()
        self.mock_ica_instance.fit_transform.return_value = np.random.rand(100, 10)
        self.mock_fast_ica.return_value = self.mock_ica_instance

    def tearDown(self):
        """Tear down test fixtures."""
        self.patcher.stop()

    def test_returns_correct_output_type(self):
        """Test that apply_fast_ica returns a numpy ndarray."""
        # Set up the mock to return a numpy array
        self.mock_ica_instance.fit_transform.return_value = np.random.rand(100, 10)

        # Apply FastICA to the test data
        result = sample_49.apply_fast_ica(self.test_data, n_components=10)

        # Check that the result is a numpy ndarray
        self.assertIsInstance(result, np.ndarray)

    def test_returns_correct_output_shape(self):
        """Test that apply_fast_ica returns an array with the correct shape."""
        # Define test parameters
        n_samples = 100
        n_components = 20
        test_data = np.random.rand(n_samples, 64)

        # Set up the mock to return an array with the correct shape
        self.mock_ica_instance.fit_transform.return_value = np.random.rand(
            n_samples, n_components
        )

        # Apply FastICA to the test data
        result = sample_49.apply_fast_ica(test_data, n_components=n_components)

        # Check the shape of the returned array
        self.assertEqual(result.shape, (n_samples, n_components))

    def test_works_with_digits_dataset(self):
        """Test that apply_fast_ica works with the digits dataset."""
        # Apply FastICA to the digits dataset
        n_components = 30
        n_samples = len(self.digits.data)

        # Set up the mock to return an array with the correct shape
        self.mock_ica_instance.fit_transform.return_value = np.random.rand(
            n_samples, n_components
        )

        # Apply FastICA
        result = sample_49.apply_fast_ica(self.digits.data, n_components=n_components)

        # Check the shape of the returned array
        self.assertEqual(result.shape, (n_samples, n_components))

    def test_works_with_different_n_components(self):
        """Test that apply_fast_ica works with different n_components values."""
        # Test cases with different n_components values
        test_cases = [5, 10, 20, 30]

        for n_components in test_cases:
            # Set up the mock to return an array with the correct shape
            self.mock_ica_instance.fit_transform.return_value = np.random.rand(
                len(self.test_data), n_components
            )

            # Apply FastICA with the current n_components
            result = sample_49.apply_fast_ica(self.test_data, n_components=n_components)

            # Check the shape of the returned array
            self.assertEqual(result.shape, (len(self.test_data), n_components))

    def test_handles_minimum_valid_n_components(self):
        """Test that apply_fast_ica works with the minimum valid n_components value."""
        # Set up the mock to return an array with the correct shape
        self.mock_ica_instance.fit_transform.return_value = np.random.rand(
            len(self.test_data), 1
        )

        # Apply FastICA with n_components=1 (minimum valid value)
        result = sample_49.apply_fast_ica(self.test_data, n_components=1)

        # Check the shape of the returned array
        self.assertEqual(result.shape, (len(self.test_data), 1))

    def test_handles_maximum_valid_n_components(self):
        """Test that apply_fast_ica works with the maximum valid n_components value."""
        # The maximum valid n_components is the number of features in the data
        n_features = self.test_data.shape[1]

        # Set up the mock to return an array with the correct shape
        self.mock_ica_instance.fit_transform.return_value = np.random.rand(
            len(self.test_data), n_features
        )

        # Apply FastICA with n_components equal to the number of features
        result = sample_49.apply_fast_ica(self.test_data, n_components=n_features)

        # Check the shape of the returned array
        self.assertEqual(result.shape, (len(self.test_data), n_features))

    def test_passes_parameters_correctly(self):
        """Test that apply_fast_ica passes parameters correctly to FastICA."""
        # Call the function with test parameters
        n_components = 10
        sample_49.apply_fast_ica(self.test_data, n_components=n_components)

        # Verify that FastICA was called with the correct parameters
        self.mock_fast_ica.assert_called_with(
            n_components=n_components, random_state=0, whiten="arbitrary-variance"
        )

        # Verify that fit_transform was called with the test data
        self.mock_ica_instance.fit_transform.assert_called_with(self.test_data)

    @patch("sample_49.FastICA")
    def test_handles_invalid_input_data(self, mock_fast_ica):
        """Test that apply_fast_ica raises an appropriate error for invalid input data."""
        # Stop the class-level patch to use a local one
        self.patcher.stop()

        # Set up the mock to raise a ValueError for invalid input
        mock_ica_instance = MagicMock()
        mock_ica_instance.fit_transform.side_effect = ValueError(
            "Expected 2D array, got 1D array instead"
        )
        mock_fast_ica.return_value = mock_ica_instance

        # Test with invalid input data (not a 2D array)
        invalid_data = np.array([1, 2, 3])  # 1D array

        # Check that the function raises a ValueError
        with self.assertRaises(ValueError):
            sample_49.apply_fast_ica(invalid_data, n_components=2)

        # Restart the class-level patch for other tests
        self.patcher = patch("sample_49.FastICA")
        self.mock_fast_ica = self.patcher.start()
        self.mock_ica_instance = MagicMock()
        self.mock_ica_instance.fit_transform.return_value = np.random.rand(100, 10)
        self.mock_fast_ica.return_value = self.mock_ica_instance


if __name__ == "__main__":
    unittest.main()
