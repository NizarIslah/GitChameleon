import unittest
from unittest.mock import MagicMock
from falcon import Request
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_253 import custom_get_param


class TestSample253(unittest.TestCase):
    def test_custom_get_param(self):
        # Create a mock Request object
        mock_request = MagicMock(spec=Request)

        # Set up the mock to return a specific value when get_param_as_json is called with "foo"
        expected_result = {"key": "value"}
        mock_request.get_param_as_json.return_value = expected_result

        # Call the function with our mock
        result = custom_get_param(mock_request)

        # Assert that the result is what we expect
        self.assertEqual(result, expected_result)

        # Verify that get_param_as_json was called with the correct parameter
        mock_request.get_param_as_json.assert_called_once_with("foo")


if __name__ == "__main__":
    unittest.main()
