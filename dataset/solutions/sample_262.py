# library: tornado
# version: 6.3.0
# extra_dependencies: []
import tornado.web
import tornado.ioloop
import tornado.httpserver
import tornado.httpclient
import socket

COOKIE_SECRET = "MY_SECRET_KEY"


class SetCookieHandler(tornado.web.RequestHandler):
    def get(self) -> None:
        self.set_signed_cookie("mycookie", "testvalue")
        self.write("Cookie set")
