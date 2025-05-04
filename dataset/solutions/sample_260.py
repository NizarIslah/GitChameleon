# library: tornado
# version: 6.3.0
# extra_dependencies: []
import tornado.ioloop
import tornado.web
import tornado.httpserver
import tornado.websocket
import tornado.httpclient
import socket


async def custom_websocket_connect(
    url: str, resolver: tornado.netutil.Resolver
) -> tornado.websocket.WebSocketClientConnection:
    return await tornado.websocket.websocket_connect(url, resolver=resolver)
