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


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def make_app():
    return tornado.web.Application(
        [
            (r"/set", SetCookieHandler),
        ],
        cookie_secret=COOKIE_SECRET,
    )


def test_set_secure_cookie():
    port = find_free_port()
    app = make_app()
    server = tornado.httpserver.HTTPServer(app)
    server.listen(port)

    client = tornado.httpclient.AsyncHTTPClient()
    url = f"http://localhost:{port}/set"

    response = tornado.ioloop.IOLoop.current().run_sync(lambda: client.fetch(url))
    server.stop()
    # Check that a Set-Cookie header is present with the cookie name "mycookie="
    set_cookie_headers = response.headers.get_list("Set-Cookie")
    return any("mycookie=" in header for header in set_cookie_headers)


result_set = test_set_secure_cookie()
assert result_set
