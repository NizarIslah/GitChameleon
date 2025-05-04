import os
import importlib.util
import io
import sys
import unittest
from unittest.mock import patch

import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sample_324


class TestSample324(unittest.TestCase):
    def setUp(self):
        self.sample_324 = sample_324

    def test_infinite_generator(self):
        """Test that the infinite generator yields values from 0 to 999 and then stops."""
        generator = self.sample_324.infinite()
        values = list(generator)

        # Check that the generator yields exactly 1000 values
        self.assertEqual(len(values), 1000)

        # Check that the values start at 0 and end at 999
        self.assertEqual(values[0], 0)
        self.assertEqual(values[-1], 999)

        # Check that the values are sequential
        for i in range(1000):
            self.assertEqual(values[i], i)

    def test_sol_dict_total(self):
        """Test that sol_dict['total'] is set to infinity."""
        self.assertEqual(self.sample_324.sol_dict["total"], float("inf"))


if __name__ == "__main__":
    unittest.main()
