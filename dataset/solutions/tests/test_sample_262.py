# Add the parent directory to import sys
import os
import sys
import unittest

import tornado.httpserver
import tornado.testing
import tornado.web

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_262 import COOKIE_SECRET, SetCookieHandler


class TestSetCookieHandler(tornado.testing.AsyncHTTPTestCase):
    def get_app(self):
        # Create an application with our handler and the cookie secret
        return tornado.web.Application(
            [(r"/", SetCookieHandler)], cookie_secret=COOKIE_SECRET
        )

    def test_set_cookie(self):
        # Make a request to the handler
        response = self.fetch("/")

        # Check that the response code is 200 (OK)
        self.assertEqual(response.code, 200)

        # Check that the response body is correct
        self.assertEqual(response.body.decode(), "Cookie set")

        # Check that the cookie was set
        cookies = response.headers.get_list("Set-Cookie")
        self.assertTrue(any("mycookie" in cookie for cookie in cookies))

        # Verify the cookie is signed (will contain signature)
        self.assertTrue(
            any("|" in cookie for cookie in cookies if "mycookie" in cookie)
        )


if __name__ == "__main__":
    unittest.main()
