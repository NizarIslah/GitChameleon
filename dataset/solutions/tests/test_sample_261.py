import os
import sys
import unittest

import tornado.httpserver
import tornado.testing
import tornado.web

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_261 import COOKIE_SECRET, GetCookieHandler


class TestGetCookieHandler(tornado.testing.AsyncHTTPTestCase):
    def get_app(self):
        # Create a test application with the GetCookieHandler
        return tornado.web.Application(
            [("/", GetCookieHandler)], cookie_secret=COOKIE_SECRET
        )

    def test_get_with_cookie(self):
        # Test when a cookie is present
        cookie_value = "test_cookie_value"
        signed_cookie = tornado.web.create_signed_value(
            COOKIE_SECRET, "mycookie", cookie_value
        ).decode()
        response = self.fetch("/", headers={"Cookie": f"mycookie={signed_cookie}"})
        self.assertEqual(response.code, 200)
        self.assertEqual(response.body.decode(), cookie_value)

    def test_get_without_cookie(self):
        # Test when no cookie is present
        response = self.fetch("/")
        self.assertEqual(response.code, 200)
        # When no cookie is present, the handler doesn't write anything
        self.assertEqual(response.body.decode(), "")


if __name__ == "__main__":
    unittest.main()
