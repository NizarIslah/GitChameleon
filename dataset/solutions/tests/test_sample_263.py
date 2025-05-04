import os
import sys
import unittest

import tornado.testing

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_263 import DummyAuth


class TestDummyAuth(tornado.testing.AsyncTestCase):
    """Test cases for the DummyAuth class."""

    def setUp(self):
        """Set up the test case."""
        super().setUp()
        self.auth = DummyAuth()

    @tornado.testing.gen_test
    async def test_async_get_user_info(self):
        """Test that async_get_user_info returns the expected dictionary."""
        # Test with a sample access token
        access_token = "sample_token"
        result = await self.auth.async_get_user_info(access_token)

        # Verify the result contains the expected keys and values
        self.assertIn("user", result)
        self.assertIn("token", result)
        self.assertEqual(result["user"], "test")
        self.assertEqual(result["token"], access_token)

    @tornado.testing.gen_test
    async def test_async_get_user_info_empty_token(self):
        """Test that async_get_user_info works with an empty token."""
        # Test with an empty access token
        access_token = ""
        result = await self.auth.async_get_user_info(access_token)

        # Verify the result contains the expected keys and values
        self.assertIn("user", result)
        self.assertIn("token", result)
        self.assertEqual(result["user"], "test")
        self.assertEqual(result["token"], access_token)


if __name__ == "__main__":
    unittest.main()
