# library: tornado
# version: 6.3.0
# extra_dependencies: []
import tornado.wsgi
import tornado.httpserver
import tornado.ioloop
import tornado.httpclient
import concurrent.futures
import socket

from typing import Callable, Dict, List, Any, Iterable

WSGIAppType = Callable[
    [Dict[str, Any], Callable[[str, List[tuple[str, str]]], None]], Iterable[bytes]
]


# A simple WSGI application that returns "Hello World"
def simple_wsgi_app(environ, start_response):
    status = "200 OK"
    headers = [("Content-Type", "text/plain")]
    start_response(status, headers)
    return [b"Hello World"]


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def custom_wsgi_container(
    app: WSGIAppType, executor: concurrent.futures.Executor
) -> tornado.wsgi.WSGIContainer:
    return tornado.wsgi.WSGIContainer(app, executor=executor)
