import os
import sys
import unittest

import tornado.httpserver
import tornado.testing
import tornado.web

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_261 import COOKIE_SECRET, GetCookieHandler


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def make_app():
    return tornado.web.Application(
        [
            (r"/get", GetCookieHandler),
        ],
        cookie_secret=COOKIE_SECRET,
    )


def test_get_secure_cookie():
    port = find_free_port()
    app = make_app()
    server = tornado.httpserver.HTTPServer(app)
    server.listen(port)

    # Create a signed cookie value for "testvalue"
    signed_cookie = tornado.web.create_signed_value(
        COOKIE_SECRET, "mycookie", "testvalue"
    )
    cookie_header = "mycookie=" + signed_cookie.decode()

    client = tornado.httpclient.AsyncHTTPClient()
    url = f"http://localhost:{port}/get"

    # Include the signed cookie in the request headers.
    response = tornado.ioloop.IOLoop.current().run_sync(
        lambda: client.fetch(url, headers={"Cookie": cookie_header})
    )
    server.stop()
    return response.body.decode() == "testvalue"


result_get = test_get_secure_cookie()
assert result_get
