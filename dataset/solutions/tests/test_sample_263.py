import unittest
import asyncio
import tornado.testing
from dataset.solutions.sample_263 import DummyAuth


class TestDummyAuth(tornado.testing.AsyncTestCase):
    """Test cases for the DummyAuth class."""

    def setUp(self):
        """Set up the test case."""
        super().setUp()
        self.auth = DummyAuth()

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