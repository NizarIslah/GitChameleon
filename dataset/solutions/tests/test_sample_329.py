import os
import sys
import unittest

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_329 import use_seaborn


class TestSample329(unittest.TestCase):
    def setUp(self):
        # Reset to default style before each test
        plt.style.use("default")

    def test_use_seaborn(self):
        # Call the function
        use_seaborn()

        # Check if 'seaborn' is in the available styles
        self.assertIn("seaborn", plt.style.available)

        # Get the current style
        current_style = plt.rcParams["_internal.classic_mode"]

        # In matplotlib, when a style is applied, the classic_mode is False
        self.assertFalse(current_style)

        # Additional check: verify some characteristic seaborn style parameters
        # This is a more robust way to check if seaborn style is applied
        self.assertEqual(plt.rcParams["axes.axisbelow"], True)

    def tearDown(self):
        # Reset to default style after each test
        plt.style.use("default")


if __name__ == "__main__":
    unittest.main()
