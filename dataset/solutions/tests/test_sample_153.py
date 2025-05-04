import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_153 import setup_environment, solution


class TestSample153(unittest.TestCase):
    def test_setup_environment(self):
        # Get the greet filter function
        greet_filter = solution()

        # Setup environment with the filter
        env = setup_environment("greet", greet_filter)

        # Check if filter was registered correctly
        self.assertIn("greet", env.filters)
        self.assertEqual(env.filters["greet"], greet_filter)

        # Test the filter in a template
        template = env.from_string("{{ 'World' | greet }}")
        result = template.render(prefix="Welcome")
        self.assertEqual(result, "Welcome, World!")

        # Test with different prefix
        result = template.render(prefix="Hola")
        self.assertEqual(result, "Hola, World!")

        # Test without prefix (should use default 'Hello')
        result = template.render()
        self.assertEqual(result, "Hello, World!")


if __name__ == "__main__":
    unittest.main()
