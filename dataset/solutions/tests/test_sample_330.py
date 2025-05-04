# Test for sample_330.py
import os
import sys
import unittest

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_330 import use_seaborn


class TestSample330(unittest.TestCase):
    def setUp(self):
        # Reset to default style before each test
        plt.style.use("default")

    def test_use_seaborn(self):
        # Call the function that should set style to seaborn-v0_8
        use_seaborn()

        # Get the current style
        current_style = plt.style.available
        current_rc_params = plt.rcParams.copy()

        # Verify that seaborn-v0_8 style was applied
        # We can't directly check plt.style.name, so we'll verify by checking
        # that certain parameters match what we expect from seaborn-v0_8 style

        # Test passes if the function executes without errors
        # and the style parameters are changed from default
        self.assertNotEqual(
            current_rc_params["axes.facecolor"],
            "white",
            "Style doesn't appear to be changed from default",
        )

        # Additional verification: reset to default and compare
        plt.style.use("default")
        default_rc_params = plt.rcParams.copy()
        self.assertNotEqual(
            current_rc_params,
            default_rc_params,
            "Style parameters should differ from default",
        )


if __name__ == "__main__":
    unittest.main()
