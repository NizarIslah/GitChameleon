import unittest
import tornado.testing
import tornado.web
import tornado.httpserver
from tornado.httpclient import HTTPResponse
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.solutions.sample_261 import GetCookieHandler, COOKIE_SECRET


class TestGetCookieHandler(tornado.testing.AsyncHTTPTestCase):
    def get_app(self):
        # Create a test application with the GetCookieHandler
        return tornado.web.Application(
            [("/", GetCookieHandler)],
            cookie_secret=COOKIE_SECRET
        )
    
    def test_get_with_cookie(self):
        # Test when a cookie is present
        # First, set a signed cookie
        cookie_value = "test_cookie_value"
        self.http_client.fetch(
            self.get_url("/"),
            method="GET",
            headers={"Cookie": f"mycookie={tornado.web.create_signed_value(COOKIE_SECRET, 'mycookie', cookie_value).decode()}"}
        )
        
        # Now make a request to get the cookie
        response = self.fetch("/")
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