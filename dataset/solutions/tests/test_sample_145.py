# Add the parent directory to import sys
import os
import sys
import unittest
from io import BytesIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_145 import app1, download, get_content_disp


class TestSample145(unittest.TestCase):
    def setUp(self):
        self.app = app1
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_get_content_disp_function(self):
        """Test that get_content_disp correctly extracts Content-Disposition header"""
        content_disp = get_content_disp(self.app, download)
        self.assertIsNotNone(content_disp)
        self.assertIn("attachment", content_disp)
        self.assertIn("filename=hello.txt", content_disp)

    def test_download_route(self):
        """Test the download route returns the correct response"""
        response = self.client.get("/download")

        # Check status code
        self.assertEqual(response.status_code, 200)

        # Check Content-Disposition header
        self.assertIn("attachment", response.headers.get("Content-Disposition"))
        self.assertIn("filename=hello.txt", response.headers.get("Content-Disposition"))

        # Check content
        self.assertEqual(response.data, b"Hello, World!")

        # Check content type
        self.assertIn("text/plain", response.headers.get("Content-Type"))

    def test_download_content(self):
        """Test that the download contains the correct content"""
        response = self.client.get("/download")
        self.assertEqual(response.data, b"Hello, World!")


if __name__ == "__main__":
    unittest.main()
