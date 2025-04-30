import unittest
import sys
import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Add the parent directory to sys.path so we can import the module under test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sample_55

class TestSample55(unittest.TestCase):
    def test_cmap_reversed(self):
        """Test that the reversed colormap is created correctly."""
        # The reversed colormap should be a LinearSegmentedColormap
        self.assertIsInstance(sample_55.cmap_reversed, LinearSegmentedColormap)

        # Its name should be the original name plus "_r"
        orig_name = sample_55.cmap.name
        self.assertEqual(sample_55.cmap_reversed.name, orig_name + "_r")

        # Test that the reversed colormap maps points correctly:
        # cmap_reversed(x) == cmap(1 - x)
        test_points = [0.0, 0.5, 1.0]
        for pt in test_points:
            expected = sample_55.cmap(1.0 - pt)
            actual = sample_55.cmap_reversed(pt)
            np.testing.assert_array_almost_equal(actual, expected)

    def test_cmap_not_mutated(self):
        """Ensure that creating the reversed cmap did not alter the original."""
        # Sample a few points and verify the original cmap still returns the same values
        pts = [0.0, 0.25, 0.75, 1.0]
        before = [sample_55.cmap(p) for p in pts]
        # Access reversed (which should not mutate the original)
        _ = sample_55.cmap_reversed
        after = [sample_55.cmap(p) for p in pts]
        # Use assert_array_almost_equal to avoid floating comparison warnings
        for b, a in zip(before, after):
            np.testing.assert_array_almost_equal(a, b)

if __name__ == "__main__":
    unittest.main()