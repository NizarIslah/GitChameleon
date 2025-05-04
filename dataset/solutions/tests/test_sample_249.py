import os
import sys
import unittest
from unittest.mock import MagicMock

import falcon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_249 import custom_link


class TestCustomLink(unittest.TestCase):
    def test_custom_link_appends_link_to_response(self):
        # Arrange
        mock_resp = MagicMock(spec=falcon.Response)
        link_rel = "next"
        link_href = "https://example.com/next"

        # Act
        result = custom_link(mock_resp, link_rel, link_href)

        # Assert
        mock_resp.append_link.assert_called_once_with(link_href, link_rel)
        self.assertEqual(result, mock_resp)

    def test_custom_link_with_real_response(self):
        # Arrange
        resp = falcon.Response()
        link_rel = "next"
        link_href = "https://example.com/next"

        # Act
        result = custom_link(resp, link_rel, link_href)

        # Assert
        # Falcon lowercases header names, and does not quote rel value
        self.assertIn("link", result.headers)
        self.assertEqual(result.headers["link"], f"<{link_href}>; rel={link_rel}")
        self.assertEqual(result, resp)


if __name__ == "__main__":
    unittest.main()
