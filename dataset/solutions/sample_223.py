# library: mitmproxy
# version: 7.0.0
# extra_dependencies: []
import contextlib


class DummyLogEntry:
    def __init__(self, msg):
        self.msg = msg


class MyAddon:
    pass


def solution() -> None:
    def add_log(self, entry):
        print(f"{entry.msg}")

    MyAddon.add_log = add_log
