# library: tornado
# version: 6.3.0
# extra_dependencies: []
import tornado.web
import tornado.ioloop
import tornado.httpserver
import tornado.httpclient
import socket

COOKIE_SECRET = "MY_SECRET_KEY"


class GetCookieHandler(tornado.web.RequestHandler):
    def get(self) -> None:
        cookie_value = self.get_signed_cookie("mycookie")
        if cookie_value:
            self.write(cookie_value.decode())
