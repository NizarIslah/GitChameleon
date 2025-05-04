import os
import sys
import unittest

from falcon import Response

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_240 import custom_body_length


class TestCustomBodyLength(unittest.TestCase):
    def test_custom_body_length_sets_content_length(self):
        # Arrange
        mock_response = Response()
        test_info = "test information"
        expected_length = len(test_info)

        # Act
        result = custom_body_length(mock_response, test_info)

        # Assert
        self.assertEqual(int(result.content_length), expected_length)
        self.assertEqual(
            result, mock_response, "Function should return the same response object"
        )

    def test_custom_body_length_with_empty_info(self):
        # Arrange
        mock_response = Response()
        test_info = ""
        expected_length = 0

        # Act
        result = custom_body_length(mock_response, test_info)

        # Assert
        self.assertEqual(int(result.content_length), expected_length)
        self.assertEqual(
            result, mock_response, "Function should return the same response object"
        )

    def test_custom_body_length_with_non_string_info(self):
        # Arrange
        mock_response = Response()
        test_info = [1, 2, 3]  # List with length 3
        expected_length = len(test_info)

        # Act
        result = custom_body_length(mock_response, test_info)

        # Assert
        self.assertEqual(int(result.content_length), expected_length)
        self.assertEqual(
            result, mock_response, "Function should return the same response object"
        )


if __name__ == "__main__":
    unittest.main()
