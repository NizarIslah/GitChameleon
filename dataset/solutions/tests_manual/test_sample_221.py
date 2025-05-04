import unittest
import io
import sys
from contextlib import redirect_stdout
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from sample_221 import DummyClientConn, ConnectionLogger, solution


import unittest
import io


class TestConnectionLogger(unittest.TestCase):
    def test_client_connected(self):
        logger = ConnectionLogger()
        solution()
        dummy_conn = DummyClientConn(("127.0.0.1", 8080))

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            logger.client_connected(dummy_conn)

        expect = "('127.0.0.1', 8080)"

        self.assertIn(expect, output.getvalue())


unittest.main()
