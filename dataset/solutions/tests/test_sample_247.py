import os
import sys
import unittest

import falcon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_247 import custom_append_link


class TestCustomAppendLink(unittest.TestCase):
    """Test cases for the custom_append_link function."""

    def setUp(self):
        """Set up a new Response object for each test."""
        self.resp = falcon.Response()

    def test_append_link_adds_link_header(self):
        """Test that the function adds a link header to the response."""
        link = "https://example.com/resource"
        rel = "next"

        # Call the function
        result = custom_append_link(self.resp, link, rel)

        # Check that the function returns the response object
        self.assertIs(result, self.resp)

        # Check that the link header was added (lowercase 'link')
        self.assertIn("link", self.resp.headers)

        # Get the Link header value
        link_header = self.resp.headers["link"]

        # Check that the link header contains the expected values
        expected_link = f"<{link}>; rel={rel}; crossorigin"
        self.assertEqual(link_header, expected_link)

    def test_append_link_with_different_values(self):
        """Test the function with different link and rel values."""
        link = "https://api.example.org/users/123"
        rel = "self"

        # Call the function
        custom_append_link(self.resp, link, rel)

        # Check the link header
        link_header = self.resp.headers["link"]
        expected_link = f"<{link}>; rel={rel}; crossorigin"
        self.assertEqual(link_header, expected_link)

    def test_append_multiple_links(self):
        """Test appending multiple links to the same response."""
        # First link
        link1 = "https://example.com/page/1"
        rel1 = "prev"
        custom_append_link(self.resp, link1, rel1)

        # Second link
        link2 = "https://example.com/page/3"
        rel2 = "next"
        custom_append_link(self.resp, link2, rel2)

        # Check that both links are in the header
        # Falcon combines multiple Link headers with a comma
        link_header = self.resp.headers["link"]
        expected_link1 = f"<{link1}>; rel={rel1}; crossorigin"
        expected_link2 = f"<{link2}>; rel={rel2}; crossorigin"

        self.assertIn(expected_link1, link_header)
        self.assertIn(expected_link2, link_header)
        self.assertIn(",", link_header)  # Links should be comma-separated


if __name__ == "__main__":
    unittest.main()
