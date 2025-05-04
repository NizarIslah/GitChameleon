# library: tornado
# version: 5.0.0
# extra_dependencies: []
import tornado.ioloop


def custom_get_ioloop() -> tornado.ioloop.IOLoop:
    return tornado.ioloop.IOLoop.current()
