# library: mitmproxy
# version: 7.0.0
# extra_dependencies: []
import contextlib


class DummyServerConn:
    def __init__(self, sockname):
        self.sockname = sockname


class ConnectionLogger:
    pass


def solution() -> None:
    def server_connected(self, server_conn: DummyServerConn) -> None:
        print(server_conn.sockname)

    ConnectionLogger.server_connected = server_connected
