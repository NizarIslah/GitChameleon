import unittest
import sys
import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Add the parent directory to sys.path to import the module to test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sample_55


class TestSample55(unittest.TestCase):
    def test_cmap_reversed(self):
        """Test that the reversed colormap is created correctly."""
        # The reversed colormap should be a LinearSegmentedColormap
        self.assertIsInstance(sample_55.cmap_reversed, LinearSegmentedColormap)

        # Its name should be the original name plus "_r"
        reverse_reverse_cmap_dict = sample_55.cmap_reversed.reversed()._segmentdata

        for key in reverse_reverse_cmap_dict.keys():
            colors = reverse_reverse_cmap_dict[key]
            ref_colors = sample_55.cmap[key]
            for i in range(len(colors)):
                color = colors[i]
                ref_color = ref_colors[i]
                for j in range(len(color)):
                    self.assertAlmostEqual((color[j]), (ref_color[j]))


if __name__ == "__main__":
    unittest.main()
