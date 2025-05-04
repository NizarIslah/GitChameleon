import os
import sys
import unittest
from unittest.mock import MagicMock

import falcon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_245 import ExampleMiddleware, custom_middleware_variable


class TestExampleMiddleware(unittest.TestCase):
    def setUp(self):
        self.middleware = ExampleMiddleware()
        self.req = MagicMock()
        self.resp = MagicMock()

    def test_process_request(self):
        # Test that process_request runs without errors
        # Since it's an empty implementation, we're just verifying it can be called
        self.middleware.process_request(self.req, self.resp)
        # No assertions needed as the method doesn't do anything yet


class TestCustomMiddlewareVariable(unittest.TestCase):
    def test_returns_list_with_middleware(self):
        result = custom_middleware_variable()

        # Check that it returns a list
        self.assertIsInstance(result, list)

        # Check that the list has exactly one item
        self.assertEqual(len(result), 1)

        # Check that the item is an instance of ExampleMiddleware
        self.assertIsInstance(result[0], ExampleMiddleware)


if __name__ == "__main__":
    unittest.main()
