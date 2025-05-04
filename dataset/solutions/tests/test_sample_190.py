import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_190 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_190 import custom_body
from sympy.physics.mechanics import RigidBody, Particle


class TestCustomBody(unittest.TestCase):
    def test_return_types_are_correct(self):
        """Test that custom_body returns the correct types."""
        rigid_body, particle = custom_body("test_body", "test_particle")
        self.assertIsInstance(rigid_body, RigidBody)
        self.assertIsInstance(particle, Particle)
        self.assertIsInstance(custom_body("body", "particle"), tuple)
        self.assertEqual(len(custom_body("body", "particle")), 2)

    def test_rigid_body_has_correct_name(self):
        """Test that the RigidBody has the correct name."""
        rigid_body, _ = custom_body("test_body", "test_particle")
        self.assertEqual(rigid_body.name, "test_body")

        # Test with a different name
        rigid_body, _ = custom_body("another_body", "test_particle")
        self.assertEqual(rigid_body.name, "another_body")

    def test_particle_has_correct_name(self):
        """Test that the Particle has the correct name."""
        _, particle = custom_body("test_body", "test_particle")
        self.assertEqual(particle.name, "test_particle")

        # Test with a different name
        _, particle = custom_body("test_body", "another_particle")
        self.assertEqual(particle.name, "another_particle")

    def test_empty_string_inputs(self):
        """Test that empty strings are handled correctly."""
        rigid_body, particle = custom_body("", "")
        self.assertEqual(rigid_body.name, "")
        self.assertEqual(particle.name, "")

    def test_special_characters_in_names(self):
        """Test that special characters in names are handled correctly."""
        special_body_name = "body!@#$%^&*()"
        special_particle_name = "particle!@#$%^&*()"

        rigid_body, particle = custom_body(special_body_name, special_particle_name)

        self.assertEqual(rigid_body.name, special_body_name)
        self.assertEqual(particle.name, special_particle_name)

    def test_numeric_string_inputs(self):
        """Test that numeric strings are handled correctly."""
        numeric_body_name = "123"
        numeric_particle_name = "456"

        rigid_body, particle = custom_body(numeric_body_name, numeric_particle_name)

        self.assertEqual(rigid_body.name, numeric_body_name)
        self.assertEqual(particle.name, numeric_particle_name)


if __name__ == "__main__":
    unittest.main()
