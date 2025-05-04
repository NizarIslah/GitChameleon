import unittest
import sys
import io
from unittest.mock import patch
from contextlib import redirect_stdout

# Import the module to test
sys.path.append("/repo/dataset/solutions")
from sample_222 import DummyClientConn, ConnectionLogger, solution


import unittest
import io


class TestConnectionLogger(unittest.TestCase):
    def test_client_disconnected(self):
        logger = ConnectionLogger()
        solution()
        dummy_conn = DummyClientConn(("127.0.0.1", 8080))

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            logger.client_disconnected(dummy_conn)

        expect = "('127.0.0.1', 8080)"

        self.assertIn(expect, output.getvalue())


unittest.main()
