# Add the parent directory to import sys
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_273 import custom_api_usage


class TestSample273(unittest.TestCase):
    def test_custom_api_usage(self):
        """Test that custom_api_usage returns the correct module name."""
        # The function should return the name of the chart_studio.api module
        result = custom_api_usage()
        self.assertEqual(result, "chart_studio.api")

    def test_dependencies_available(self):
        """Test that the required dependencies are available."""
        # Test that plotly is available
        import plotly

        self.assertIsNotNone(plotly)

        # Test that chart_studio is available
        import chart_studio

        self.assertIsNotNone(chart_studio)

        # Test that chart_studio.api is available
        import chart_studio.api

        self.assertIsNotNone(chart_studio.api)


if __name__ == "__main__":
    unittest.main()
