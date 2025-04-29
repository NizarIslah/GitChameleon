import unittest
import sys
import os
import numpy as np

# Add the parent directory to sys.path to import the module to test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sample_55

class TestSample55(unittest.TestCase):
    def test_cmap_reversed(self):
        """Test that the reversed colormap dict is created correctly."""
        cmap = sample_55.cmap
        cmap_rev = sample_55.cmap_reversed

        # Both should be dicts mapping numeric keys to colors
        self.assertIsInstance(cmap, dict)
        self.assertIsInstance(cmap_rev, dict)

        orig_keys = set(cmap.keys())
        rev_keys = set(cmap_rev.keys())

        # Expect reversed keys = {1 - k for k in orig_keys}
        expected_rev_keys = {1 - k for k in orig_keys}
        self.assertEqual(rev_keys, expected_rev_keys)

        # Values at reversed keys should match original values
        for k, v in cmap.items():
            rv = cmap_rev[1 - k]
            np.testing.assert_array_almost_equal(np.array(rv), np.array(v))

    def test_cmap_not_mutated(self):
        """Ensure that creating the reversed cmap did not alter the original."""
        cmap = sample_55.cmap
        # Make a deep copy of the original values
        before = {k: np.array(v) for k, v in cmap.items()}

        # Access the reversed cmap (should not mutate the original)
        _ = sample_55.cmap_reversed

        after = sample_55.cmap
        self.assertEqual(set(before.keys()), set(after.keys()))

        for k in before:
            np.testing.assert_array_equal(before[k], np.array(after[k]))


if __name__ == "__main__":
    unittest.main()
