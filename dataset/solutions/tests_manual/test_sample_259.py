import concurrent.futures
import os
import socket

# Add the parent directory to import sys
import sys
import unittest

import tornado.httpclient
import tornado.httpserver
import tornado.ioloop
import tornado.wsgi

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_259 import custom_wsgi_container, find_free_port, simple_wsgi_app


def test_wsgi_container_executor():
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    container = custom_wsgi_container(simple_wsgi_app, executor)

    port = find_free_port()
    server = tornado.httpserver.HTTPServer(container)
    server.listen(port)

    client = tornado.httpclient.AsyncHTTPClient()
    url = f"http://localhost:{port}"

    response = tornado.ioloop.IOLoop.current().run_sync(lambda: client.fetch(url))

    server.stop()
    executor.shutdown(wait=True)

    return response.body == b"Hello World"


result = test_wsgi_container_executor()
assert result
