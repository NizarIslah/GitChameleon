import unittest
import numpy as np
from scipy.stats import hmean
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_160 import count_unique_hmean

class TestCountUniqueHmean(unittest.TestCase):
    
    def test_basic_functionality(self):
        # Test with simple arrays having different harmonic means
        data = np.array([
            [1, 2, 3],
            [2, 4, 6],
            [1, 2, 3]  # Same as first array, should not count twice
        ])
        result = count_unique_hmean(data)
        self.assertEqual(result, 2)  # Should have 2 unique harmonic means
    
    def test_with_nan_values(self):
        # Test with arrays containing NaN values
        data = np.array([
            [1, 2, 3],
            [2, np.nan, 6],  # Contains NaN, should result in NaN
            [4, 5, 6],
            [7, np.nan, 9]   # Another NaN, but counts as one unique NaN
        ])
        result = count_unique_hmean(data)
        self.assertEqual(result, 3)  # 2 unique non-NaN values + 1 for NaN
    
    def test_all_nan_arrays(self):
        # Test with arrays that all result in NaN
        data = np.array([
            [1, np.nan, 3],
            [2, np.nan, 6],
            [4, np.nan, 6]
        ])
        result = count_unique_hmean(data)
        self.assertEqual(result, 1)  # All are NaN, so count is 1
    
    def test_negative_values(self):
        # Harmonic mean requires positive values, negative values result in NaN
        data = np.array([
            [1, 2, 3],
            [-1, 2, 3],  # Contains negative, should result in NaN
            [4, 5, 6]
        ])
        result = count_unique_hmean(data)
        self.assertEqual(result, 3)  # 2 unique non-NaN values + 1 for NaN
    
    def test_zero_values(self):
        # Harmonic mean with zero values results in NaN
        data = np.array([
            [1, 2, 3],
            [0, 2, 3],  # Contains zero, should result in NaN
            [4, 5, 6]
        ])
        result = count_unique_hmean(data)
        self.assertEqual(result, 3)  # 2 unique non-NaN values + 1 for NaN
    
    def test_empty_array(self):
        # Test with an empty array
        data = np.array([])
        data = data.reshape(0, 1)  # Reshape to maintain 2D structure
        result = count_unique_hmean(data)
        self.assertEqual(result, 0)  # No values, so count is 0
    
    def test_same_hmean_different_arrays(self):
        # Test with different arrays that have the same harmonic mean
        data = np.array([
            [2, 2, 2],  # hmean = 2
            [1, 4, 4],  # different values but same hmean ≈ 2
            [10, 10, 10]  # hmean = 10
        ])
        result = count_unique_hmean(data)
        self.assertEqual(result, 2)  # Should have 2 unique harmonic means

if __name__ == '__main__':
    unittest.main()