import os
import sys
import unittest

import jinja2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_154 import setup_environment, solution


class TestSample154(unittest.TestCase):
    def test_setup_environment(self):
        """Test that setup_environment correctly adds a filter to the environment"""

        def dummy_filter(value):
            return f"filtered_{value}"

        env = setup_environment("test_filter", dummy_filter)

        # Check that the environment is created correctly
        self.assertIsInstance(env, jinja2.Environment)
        # Check that our filter was added
        self.assertIn("test_filter", env.filters)
        self.assertEqual(env.filters["test_filter"], dummy_filter)

        # Test the filter functionality
        template = env.from_string("{{ 'hello'|test_filter }}")
        self.assertEqual(template.render(), "filtered_hello")

    def test_greet_function(self):
        """Test the greet function returned by solution()"""
        greet_filter = solution()

        # Create a mock context
        env = jinja2.Environment()
        env.filters["greet"] = greet_filter  # Register the filter
        template = env.from_string("{{ name|greet }}")

        # Test with default prefix
        rendered = template.render(name="World")
        self.assertEqual(rendered, "Hello, World!")

        # Test with custom prefix
        rendered = template.render(name="World", prefix="Hi")
        self.assertEqual(rendered, "Hi, World!")

    def test_greet_integration(self):
        """Test the greet filter integrated into a Jinja2 environment"""
        env = setup_environment("greet", solution())

        # Test with default prefix
        template = env.from_string("{{ 'World'|greet }}")
        self.assertEqual(template.render(), "Hello, World!")

        # Test with custom prefix
        template = env.from_string("{{ 'World'|greet }}")
        self.assertEqual(template.render(prefix="Hola"), "Hola, World!")

        # Test with multiple uses in the same template
        template = env.from_string("{{ 'World'|greet }} {{ 'Universe'|greet }}")
        self.assertEqual(
            template.render(prefix="Greetings"),
            "Greetings, World! Greetings, Universe!",
        )


if __name__ == "__main__":
    unittest.main()
