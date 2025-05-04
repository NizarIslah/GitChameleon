import os
import sys
import unittest
from io import BytesIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_146 import app1, get_content_disp


class TestSample146(unittest.TestCase):
    def setUp(self):
        app1.testing = True
        self.client = app1.test_client()

    def test_download_route(self):
        """Test the download route returns the correct file with proper headers."""
        response = self.client.get("/download")

        # Check status code
        self.assertEqual(response.status_code, 200)

        # Check content
        self.assertEqual(response.data, b"Hello, World!")

        # Check Content-Disposition header
        self.assertIn(
            "attachment; filename=hello.txt",
            response.headers.get("Content-Disposition"),
        )

        # Check Content-Type (updated to match actual response)
        self.assertEqual(
            response.headers.get("Content-Type"), "text/plain; charset=utf-8"
        )

    def test_get_content_disp(self):
        """Test the get_content_disp helper function."""
        # Use the download function from our app
        content_disp = get_content_disp(app1, app1.view_functions["download"])

        # Check that the Content-Disposition header is set correctly
        self.assertIn("attachment; filename=hello.txt", content_disp)


if __name__ == "__main__":
    unittest.main()
