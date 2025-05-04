import unittest
import sys
import io
from contextlib import redirect_stdout
from unittest.mock import patch
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_218 import DummyServerConn, ConnectionLogger, solution


import unittest
import io


class TestConnectionLogger(unittest.TestCase):
    def test_server_connected(self):
        # Update the ConnectionLogger class with the new method.
        solution()
        logger = ConnectionLogger()
        dummy_conn = DummyServerConn(("127.0.0.1", 8080))

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            logger.server_connected(dummy_conn)
        print(output.getvalue())
        expect = "('127.0.0.1', 8080)"

        self.assertIn(expect, output.getvalue())


unittest.main()
