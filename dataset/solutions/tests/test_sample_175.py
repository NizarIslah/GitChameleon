import unittest
import sys
import os
from typing import List
import numpy as np

# Add the parent directory to sys.path to import the module to test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_175 import custom_generateRandomSampleDice

# Import required libraries from sample_175.py
from sympy.stats import Die
import sympy.stats.rv


class TestCustomGenerateRandomSampleDice(unittest.TestCase):
    def test_sample_generation(self):
        """Test that the function generates the correct number of samples."""
        # Create a fair die with 6 sides
        fair_die = Die("D", 6)

        # Test with different sample sizes
        sample_sizes = [1, 5, 10]

        for size in sample_sizes:
            samples = custom_generateRandomSampleDice(fair_die, size)

            # Check that the correct number of samples was generated
            self.assertEqual(len(samples), size)

            # Check that all samples are within the expected range (1-6 for a standard die)
            for sample_value in samples:
                self.assertGreaterEqual(sample_value, 1)
                self.assertLessEqual(sample_value, 6)

    def test_different_dice(self):
        """Test that the function works with different types of dice."""
        # Test with different dice (4-sided, 8-sided, 20-sided)
        dice_sides = [4, 8, 20]
        sample_size = 5

        for sides in dice_sides:
            die = Die("D", sides)
            samples = custom_generateRandomSampleDice(die, sample_size)

            # Check that all samples are within the expected range
            for sample_value in samples:
                self.assertGreaterEqual(sample_value, 1)
                self.assertLessEqual(sample_value, sides)

    def test_return_type(self):
        """Test that the function returns a list of integers."""
        die = Die("D", 6)
        samples = custom_generateRandomSampleDice(die, 3)

        # Check that the return value is a list
        self.assertIsInstance(samples, list)

        # Check that all elements in the list are integers
        for sample_value in samples:
            self.assertIsInstance(sample_value, np.int64)


if __name__ == "__main__":
    unittest.main()
