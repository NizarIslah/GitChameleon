import unittest
import sys
import os
import numpy as np

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sample_84


class TestSample84(unittest.TestCase):
    """Test cases for the LightGBM cross-validation in sample_84.py."""

    def test_cv_results_exists_and_type_from_module(self):
        """
        Tests that cv_results exists in the imported module and is a dictionary.
        """
        # Check if the cv_results variable is present in the imported module
        self.assertTrue(
            hasattr(sample_84, "cv_results"),
            "The variable 'cv_results' was not found in the imported sample_84.",
        )

        # If it exists, assign it to a local variable for easier access
        if hasattr(sample_84, "cv_results"):
            module_cv_results = sample_84.cv_results

            self.assertIsNotNone(
                module_cv_results, "cv_results from module should not be None"
            )
            self.assertIsInstance(
                module_cv_results, dict, "cv_results from module should be a dictionary"
            )

            # Optionally, you can add more specific checks about the content if needed
            self.assertIn(
                "valid binary_logloss-mean",
                module_cv_results,
                "Key 'valid binary_logloss-mean' not found in cv_results from module",
            )
            self.assertIn(
                "train binary_logloss-mean",
                module_cv_results,
                "Key 'train binary_logloss-mean' not found in cv_results from module",
            )


if __name__ == "__main__":
    unittest.main()
