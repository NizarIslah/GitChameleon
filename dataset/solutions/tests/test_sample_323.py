#!/usr/bin/env python3
# Test file for sample_323.py

import io
import sys
import unittest
from unittest.mock import patch

from tqdm import tqdm

sys.path.append("/repo/dataset/solutions")
from sample_323 import infinite, sol_dict


class TestSample323(unittest.TestCase):
    def test_infinite_generator(self):
        """Test that the infinite generator yields expected values and stops at 1000."""
        # Create a generator from the infinite function
        gen = infinite()

        # Check the first few values
        self.assertEqual(next(gen), 0)
        self.assertEqual(next(gen), 1)
        self.assertEqual(next(gen), 2)

        # Check that it stops at 1000
        # Consume all values
        values = list(gen)
        self.assertEqual(
            len(values), 997
        )  # 997 more values after the 3 we already consumed
        self.assertEqual(values[-1], 999)  # Last value should be 999

    def test_sol_dict_initialization(self):
        """Test that sol_dict is initialized with total set to None."""
        self.assertIsNone(sol_dict["total"])

    @patch("sys.stderr", new_callable=io.StringIO)
    def test_progress_bar_creation(self, mock_stderr):
        """Test that a progress bar can be created with the infinite generator."""
        # Create a small test with a defined total
        test_total = 10
        test_dict = {"total": test_total}

        # Create a progress bar with a small number of iterations
        with tqdm(infinite(), total=test_dict["total"]) as pbar:
            # Just iterate a few times
            for i in range(5):
                pbar.update(1)
                pbar.set_description(f"Testing {i}")

        # Check that the progress bar output contains expected text
        output = mock_stderr.getvalue()
        self.assertIn("Testing", output)

    def test_progress_bar_with_none_total(self):
        """Test that a progress bar works even with None as total."""
        # This is similar to the actual code in sample_323
        test_dict = {"total": None}

        # Create a generator that will only yield a few values for testing
        def test_gen():
            for i in range(5):
                yield i

        # This should not raise an exception even though total is None
        with tqdm(test_gen(), total=test_dict["total"]) as pbar:
            for _ in pbar:
                pbar.set_description("Processing")

        # If we got here without exceptions, the test passes
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
