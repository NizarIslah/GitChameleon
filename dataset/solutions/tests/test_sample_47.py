# Add the parent directory to import sys
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import sample_47
from sklearn.datasets import make_sparse_coded_signal

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestGetSignal(unittest.TestCase):
    def test_returns_correct_output_shapes(self):
        """Test that get_signal returns matrices with the correct shapes."""
        # Define test parameters
        n_samples = 5
        n_features = 10
        n_components = 8
        n_nonzero_coefs = 3

        # Call the function
        data, dictionary, code = sample_47.get_signal(
            n_samples=n_samples,
            n_features=n_features,
            n_components=n_components,
            n_nonzero_coefs=n_nonzero_coefs,
        )

        # The data matrix comes back as (n_features, n_samples)
        self.assertEqual(data.shape, (n_features, n_samples))
        # The dictionary should be (n_features, n_components)
        self.assertEqual(dictionary.shape, (n_features, n_components))
        # The code matrix should be (n_components, n_samples)
        self.assertEqual(code.shape, (n_components, n_samples))

    def test_works_with_different_parameters(self):
        """Test that get_signal works with different parameter values."""
        test_cases = [
            {
                "n_samples": 10,
                "n_features": 15,
                "n_components": 12,
                "n_nonzero_coefs": 5,
            },
            {"n_samples": 3, "n_features": 8, "n_components": 6, "n_nonzero_coefs": 2},
            {
                "n_samples": 20,
                "n_features": 30,
                "n_components": 25,
                "n_nonzero_coefs": 10,
            },
        ]

        for params in test_cases:
            data, dictionary, code = sample_47.get_signal(**params)

            self.assertEqual(data.shape, (params["n_features"], params["n_samples"]))
            self.assertEqual(
                dictionary.shape, (params["n_features"], params["n_components"])
            )
            self.assertEqual(code.shape, (params["n_components"], params["n_samples"]))


if __name__ == "__main__":
    unittest.main()
