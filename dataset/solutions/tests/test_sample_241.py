import os
import sys
import unittest
from unittest.mock import MagicMock

import falcon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_241 import custom_data


class TestSample241(unittest.TestCase):
    def test_custom_data(self):
        # Create a mock Response object
        mock_resp = MagicMock(spec=falcon.Response)
        mock_resp.render_body.return_value = b"test_info"

        # Test data
        test_info = "test_info"

        # Call the function
        result = custom_data(mock_resp, test_info)

        # Verify the response data was set correctly
        mock_resp.data = test_info

        # Verify render_body was called
        mock_resp.render_body.assert_called_once()

        # Verify the function returns the rendered body
        self.assertEqual(result, b"test_info")

    def test_custom_data_with_real_response(self):
        # Create a real Response object
        resp = falcon.Response()

        # Test data
        test_info = "test_info"

        # Call the function
        result = custom_data(resp, test_info)

        # Verify the response data was set correctly
        self.assertEqual(resp.data, test_info)

        # Verify the function returns the rendered body (should be a string)
        self.assertEqual(result, test_info)


if __name__ == "__main__":
    unittest.main()
