import os

# Add the parent directory to the path so we can import the solution module
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_120 import compute_moment
from scipy.stats import expon, norm, uniform


class TestComputeMoment(unittest.TestCase):
    def test_normal_distribution_moments(self):
        """Test moments of a standard normal distribution."""
        # Standard normal distribution
        normal_dist = norm(loc=0, scale=1)

        # First moment (mean) of standard normal is 0
        self.assertAlmostEqual(compute_moment(normal_dist, 1), 0, places=10)

        # Second moment (variance + mean^2) of standard normal is 1
        self.assertAlmostEqual(compute_moment(normal_dist, 2), 1, places=10)

        # Third moment of standard normal is 0
        self.assertAlmostEqual(compute_moment(normal_dist, 3), 0, places=10)

        # Fourth moment of standard normal is 3
        self.assertAlmostEqual(compute_moment(normal_dist, 4), 3, places=10)

    def test_uniform_distribution_moments(self):
        """Test moments of a uniform distribution."""
        # Uniform distribution on [0, 1]
        uniform_dist = uniform(loc=0, scale=1)

        # First moment (mean) of uniform(0,1) is 0.5
        self.assertAlmostEqual(compute_moment(uniform_dist, 1), 0.5, places=10)

        # Second moment of uniform(0,1) is 1/3
        self.assertAlmostEqual(compute_moment(uniform_dist, 2), 1 / 3, places=10)

        # Third moment of uniform(0,1) is 1/4
        self.assertAlmostEqual(compute_moment(uniform_dist, 3), 0.25, places=10)

    def test_exponential_distribution_moments(self):
        """Test moments of an exponential distribution."""
        # Exponential distribution with rate parameter 1
        exp_dist = expon(scale=1)

        # First moment (mean) of exponential(1) is 1
        self.assertAlmostEqual(compute_moment(exp_dist, 1), 1, places=10)

        # Second moment of exponential(1) is 2
        self.assertAlmostEqual(compute_moment(exp_dist, 2), 2, places=10)

        # Third moment of exponential(1) is 6
        self.assertAlmostEqual(compute_moment(exp_dist, 3), 6, places=10)

    def test_non_integer_moment_error(self):
        """Test that non-integer moments raise appropriate errors."""
        normal_dist = norm()

        # The function should pass the n parameter directly to dist.moment()
        # which should raise a ValueError for non-integer n
        with self.assertRaises(ValueError):
            compute_moment(normal_dist, 1.5)

    def test_negative_moment_error(self):
        """Test that negative moments raise appropriate errors."""
        normal_dist = norm()

        # The function should pass the n parameter directly to dist.moment()
        # which should raise a ValueError for negative n
        with self.assertRaises(ValueError):
            compute_moment(normal_dist, -1)


if __name__ == "__main__":
    unittest.main()
