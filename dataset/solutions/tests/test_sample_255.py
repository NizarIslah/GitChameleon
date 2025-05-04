# Import the function to test
import os
import sys
import unittest
from unittest.mock import MagicMock

from falcon import Request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_255 import custom_get_dpr


class TestSample255(unittest.TestCase):
    def setUp(self):
        # Create a mock Request object for testing
        self.mock_request = MagicMock(spec=Request)

    def test_custom_get_dpr_valid_values(self):
        # Test with valid values within range
        for value in range(4):  # 0, 1, 2, 3
            # Configure the mock to return the specified value
            self.mock_request.get_param_as_int.return_value = value

            # Call the function with our mock
            result = custom_get_dpr(self.mock_request)

            # Assert the function returns the expected value
            self.assertEqual(result, value)

            # Verify the mock was called with correct parameters
            self.mock_request.get_param_as_int.assert_called_with(
                "dpr", min_value=0, max_value=3
            )

    def test_custom_get_dpr_default_behavior(self):
        # Test that the function correctly passes through the result from get_param_as_int
        self.mock_request.get_param_as_int.return_value = 2
        result = custom_get_dpr(self.mock_request)
        self.assertEqual(result, 2)

    def test_custom_get_dpr_parameter_constraints(self):
        # Test that the function passes the correct constraints to get_param_as_int
        custom_get_dpr(self.mock_request)
        self.mock_request.get_param_as_int.assert_called_once_with(
            "dpr", min_value=0, max_value=3
        )


if __name__ == "__main__":
    unittest.main()
