import os
# Add the parent directory to the path so we can import the sample
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_301 import compute_shear


class TestComputeShear(unittest.TestCase):
    
    def test_compute_shear_basic(self):
        """Test basic functionality of compute_shear with a simple array."""
        # Create a simple 3x3 array
        E = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        
        # Apply shear with factor=1
        result = compute_shear(E, factor=1, axis=0)
        
        # Expected result: each column is rolled by factor * column_index
        # Column 0: roll by 0
        # Column 1: roll by 1
        # Column 2: roll by 2
        expected = np.array([
            [1, 5, 9],
            [4, 8, 3],
            [7, 2, 6]
        ])
        
        np.testing.assert_array_equal(result, expected)
    
    def test_compute_shear_zero_factor(self):
        """Test compute_shear with factor=0 (should return the original array)."""
        E = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        
        result = compute_shear(E, factor=0, axis=0)
        
        # With factor=0, no rolling should occur
        np.testing.assert_array_equal(result, E)
    
    def test_compute_shear_negative_factor(self):
        """Test compute_shear with a negative factor."""
        E = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        
        result = compute_shear(E, factor=-1, axis=0)
        
        # Expected result: each column is rolled by -1 * column_index
        # Column 0: roll by 0
        # Column 1: roll by -1
        # Column 2: roll by -2
        expected = np.array([
            [1, 8, 6],
            [4, 2, 9],
            [7, 5, 3]
        ])
        
        np.testing.assert_array_equal(result, expected)
    
    def test_compute_shear_rectangular_array(self):
        """Test compute_shear with a rectangular array."""
        E = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ])
        
        result = compute_shear(E, factor=1, axis=0)
        
        # Expected result:
        # Column 0: roll by 0
        # Column 1: roll by 1
        # Column 2: roll by 2
        # Column 3: roll by 3
        expected = np.array([
            [1, 6, 3, 8],
            [5, 2, 7, 4]
        ])
        
        np.testing.assert_array_equal(result, expected)
    
    def test_compute_shear_large_factor(self):
        """Test compute_shear with a factor larger than array dimensions."""
        E = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        
        # Factor = 4 (larger than array height)
        result = compute_shear(E, factor=4, axis=0)
        
        # Expected result:
        # Column 0: roll by 0
        # Column 1: roll by 4 (equivalent to roll by 1 in a 3x3 array)
        # Column 2: roll by 8 (equivalent to roll by 2 in a 3x3 array)
        expected = np.array([
            [1, 5, 9],
            [4, 8, 3],
            [7, 2, 6]
        ])
        
        np.testing.assert_array_equal(result, expected)


if __name__ == '__main__':
    unittest.main()