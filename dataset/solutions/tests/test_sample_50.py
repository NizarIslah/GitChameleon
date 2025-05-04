import unittest
import numpy as np
from sklearn.impute import SimpleImputer
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_50 import get_imputer


class TestSample50(unittest.TestCase):
    def test_get_imputer_returns_simple_imputer(self):
        """Test that get_imputer returns a SimpleImputer instance."""
        data = np.array([[1, 2], [np.nan, 3], [7, 6]])
        imputer = get_imputer(data)
        self.assertIsInstance(imputer, SimpleImputer)

    def test_get_imputer_default_strategy(self):
        """Test that the returned imputer has the default strategy (mean)."""
        data = np.array([[1, 2], [np.nan, 3], [7, 6]])
        imputer = get_imputer(data)
        self.assertEqual(imputer.strategy, "mean")

    def test_imputer_functionality(self):
        """Test that the imputer can actually impute missing values."""
        data = np.array([[1, 2], [np.nan, 3], [7, 6]])
        imputer = get_imputer(data)
        imputer.fit(data)
        transformed_data = imputer.transform(data)
        # Check that NaN values are replaced (with mean of column)
        self.assertFalse(np.isnan(transformed_data).any())
        # The mean of the first column is (1+7)/2 = 4
        self.assertEqual(transformed_data[1, 0], 4)


if __name__ == "__main__":
    unittest.main()
