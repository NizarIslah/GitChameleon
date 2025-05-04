import os
import sys
import unittest

import falcon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_239 import custom_body


class TestCustomBody(unittest.TestCase):
    def test_custom_body_sets_text(self):
        # Create a falcon HTTPStatus object
        status = falcon.HTTPStatus(falcon.HTTP_200)

        # Test message
        test_message = "Test message"

        # Call the custom_body function
        result = custom_body(status, test_message)

        # Verify the text was set correctly
        self.assertEqual(result.text, test_message)

        # Verify the result is the same object as the input
        self.assertIs(result, status)

    def test_custom_body_with_different_status_codes(self):
        # Test with different HTTP status codes
        status_codes = [
            falcon.HTTP_200,  # OK
            falcon.HTTP_201,  # Created
            falcon.HTTP_400,  # Bad Request
            falcon.HTTP_404,  # Not Found
            falcon.HTTP_500,  # Internal Server Error
        ]

        for code in status_codes:
            status = falcon.HTTPStatus(code)
            message = f"Status message for {code}"

            result = custom_body(status, message)

            self.assertEqual(result.text, message)
            self.assertEqual(result.status, code)

    def test_custom_body_with_empty_message(self):
        # Test with an empty message
        status = falcon.HTTPStatus(falcon.HTTP_200)
        empty_message = ""

        result = custom_body(status, empty_message)

        self.assertEqual(result.text, empty_message)


if __name__ == "__main__":
    unittest.main()
