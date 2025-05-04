# library: tornado
# version: 6.0.0
# extra_dependencies: []
import tornado.httputil


class DummyConnection:
    def __init__(self):
        self.buffer = []

    def write(self, chunk):
        self.buffer.append(chunk)


req = tornado.httputil.HTTPServerRequest(method="GET", uri="/")
req.connection = DummyConnection()


def custom_write(request: tornado.httputil.HTTPServerRequest, text: str) -> list[str]:
    request.connection.write(text)
    return request.connection.buffer
