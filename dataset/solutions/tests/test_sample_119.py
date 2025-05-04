import os
import sys
import unittest

import numpy as np
from scipy.stats import expon, norm, uniform

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_119 import compute_moment


class TestComputeMoment(unittest.TestCase):
    def test_normal_distribution_moments(self):
        """Test moments of normal distribution."""
        # Standard normal distribution
        normal_dist = norm(loc=0, scale=1)

        # First moment (mean) of standard normal is 0
        self.assertAlmostEqual(compute_moment(normal_dist, 1), 0.0)

        # Second moment (variance + mean^2) of standard normal is 1
        self.assertAlmostEqual(compute_moment(normal_dist, 2), 1.0)

        # Third moment of standard normal is 0
        self.assertAlmostEqual(compute_moment(normal_dist, 3), 0.0)

        # Fourth moment of standard normal is 3
        self.assertAlmostEqual(compute_moment(normal_dist, 4), 3.0)

        # Test with non-standard normal
        shifted_normal = norm(loc=5, scale=2)
        self.assertAlmostEqual(compute_moment(shifted_normal, 1), 5.0)  # Mean is 5
        self.assertAlmostEqual(
            compute_moment(shifted_normal, 2), 29.0
        )  # 5^2 + 2^2 = 29

    def test_uniform_distribution_moments(self):
        """Test moments of uniform distribution."""
        # Uniform distribution on [0, 1]
        uniform_dist = uniform(loc=0, scale=1)

        # First moment (mean) of uniform(0,1) is 0.5
        self.assertAlmostEqual(compute_moment(uniform_dist, 1), 0.5)

        # Second moment of uniform(0,1) is 1/3
        self.assertAlmostEqual(compute_moment(uniform_dist, 2), 1 / 3)

        # Third moment of uniform(0,1) is 1/4
        self.assertAlmostEqual(compute_moment(uniform_dist, 3), 1 / 4)

    def test_exponential_distribution_moments(self):
        """Test moments of exponential distribution."""
        # Exponential distribution with scale=1 (rate=1)
        exp_dist = expon(scale=1)

        # First moment (mean) of exponential(1) is 1
        self.assertAlmostEqual(compute_moment(exp_dist, 1), 1.0)

        # Second moment of exponential(1) is 2
        self.assertAlmostEqual(compute_moment(exp_dist, 2), 2.0)

        # Third moment of exponential(1) is 6
        self.assertAlmostEqual(compute_moment(exp_dist, 3), 6.0)

    def test_invalid_moment_order(self):
        """Test that negative moment orders raise an error."""
        normal_dist = norm()
        with self.assertRaises(ValueError):
            compute_moment(normal_dist, -1)


if __name__ == "__main__":
    unittest.main()
