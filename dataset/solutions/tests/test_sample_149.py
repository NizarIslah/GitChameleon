# Add the parent directory to import sys
import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_149 import error404, safe_join_fail_404


class TestSafeJoinFail404(unittest.TestCase):
    def test_safe_join_success(self):
        """Test that safe_join_fail_404 returns the joined path when valid."""
        base_path = "/base/path"
        sub_path = "sub/path"

        # When the paths can be safely joined
        result = safe_join_fail_404(base_path, sub_path)

        # Then the result should be the joined path
        self.assertEqual(result, os.path.join(base_path, sub_path))

    def test_safe_join_fail_404(self):
        """Test that safe_join_fail_404 raises a 404 error when the path is outside the base path."""
        base_path = "/base/path"
        sub_path = "../outside"

        # The current implementation doesn't actually raise the error,
        # but according to the function's comments, it should.
        # This test will fail with the current implementation.

        with self.assertRaises(error404):
            safe_join_fail_404(base_path, sub_path)

    @patch("flask.safe_join")
    def test_flask_safe_join_called(self, mock_safe_join):
        """Test that flask.safe_join is called with the correct arguments."""
        base_path = "/base/path"
        sub_path = "sub/path"
        expected_result = "/base/path/sub/path"

        # Set up the mock to return a specific value
        mock_safe_join.return_value = expected_result

        # Call the function
        result = safe_join_fail_404(base_path, sub_path)

        # Verify that flask.safe_join was called with the correct arguments
        mock_safe_join.assert_called_once_with(base_path, sub_path)

        # Verify that the function returns the result from flask.safe_join
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
