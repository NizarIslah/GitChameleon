import unittest
import sys
import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Add the parent directory to sys.path to import the module to test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sample_55

class TestSample55(unittest.TestCase):
    
    def test_cmap_creation(self):
        """Test that the colormap is created correctly."""
        # Check that cmap is a dictionary with the expected keys
        self.assertIsInstance(sample_55.cmap, dict)
        self.assertIn("blue", sample_55.cmap)
        self.assertIn("red", sample_55.cmap)
        self.assertIn("green", sample_55.cmap)
        
        # Check the values in the cmap dictionary
        self.assertEqual(sample_55.cmap["blue"], [[1, 2, 2], [2, 2, 1]])
        self.assertEqual(sample_55.cmap["red"], [[0, 0, 0], [1, 0, 0]])
        self.assertEqual(sample_55.cmap["green"], [[0, 0, 0], [1, 0, 0]])
    
    def test_cmap_reversed(self):
        """Test that the reversed colormap is created correctly."""
        # Check that cmap_reversed is a LinearSegmentedColormap
        self.assertIsInstance(sample_55.cmap_reversed, LinearSegmentedColormap)
        
        # Check that the name is correct (with _r suffix for reversed)
        self.assertEqual(sample_55.cmap_reversed.name, "custom_cmap_r")
        
        # Test that the colormap is actually reversed
        original_cmap = LinearSegmentedColormap("custom_cmap", sample_55.cmap)
        
        # Get colors at specific points and verify they're reversed
        test_points = [0.0, 0.5, 1.0]
        for point in test_points:
            # The color at point x in the original should match color at 1-x in the reversed
            np.testing.assert_array_almost_equal(
                original_cmap(point),
                sample_55.cmap_reversed(1.0 - point)
            )

if __name__ == '__main__':
    unittest.main()