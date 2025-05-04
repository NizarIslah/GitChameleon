# library: mitmproxy
# version: 7.0.0
# extra_dependencies: []
import contextlib


class DummyClientConn:
    def __init__(self, peername):
        self.peername = peername


class ConnectionLogger:
    pass


def solution() -> None:
    def client_disconnected(self, client_conn) -> None:
        print(client_conn.peername)

    ConnectionLogger.client_disconnected = client_disconnected
