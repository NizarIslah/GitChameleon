import unittest
from typing import Dict, Any
import falcon.testing

# Import the function to test
from dataset.solutions.sample_246 import custom_environ


class TestCustomEnviron(unittest.TestCase):
    def test_custom_environ_returns_dict(self):
        """Test that custom_environ returns a dictionary."""
        result = custom_environ("1.1")
        self.assertIsInstance(result, dict)

    def test_custom_environ_sets_http_version(self):
        """Test that custom_environ sets the HTTP version correctly."""
        # Test with HTTP 1.0
        result_1_0 = custom_environ("1.0")
        self.assertEqual(result_1_0.get("SERVER_PROTOCOL"), "HTTP/1.0")

        # Test with HTTP 1.1
        result_1_1 = custom_environ("1.1")
        self.assertEqual(result_1_1.get("SERVER_PROTOCOL"), "HTTP/1.1")

        # Test with HTTP 2.0
        result_2_0 = custom_environ("2.0")
        self.assertEqual(result_2_0.get("SERVER_PROTOCOL"), "HTTP/2.0")

    def test_custom_environ_matches_falcon_testing(self):
        """Test that custom_environ produces the same result as falcon.testing.create_environ."""
        http_version = "1.1"
        result = custom_environ(http_version)
        expected = falcon.testing.create_environ(http_version=http_version)
        
        # Check that all keys in expected are in result with the same values
        for key, value in expected.items():
            self.assertEqual(result.get(key), value)
        
        # Check that all keys in result are in expected
        for key in result:
            self.assertIn(key, expected)


if __name__ == "__main__":
    unittest.main()