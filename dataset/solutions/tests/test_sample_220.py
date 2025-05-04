import unittest
import sys
import io
from contextlib import redirect_stdout
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_220 import DummyServerConn, ConnectionLogger, solution


class TestSample220(unittest.TestCase):
    def setUp(self):
        # Reset ConnectionLogger class before each test
        ConnectionLogger.server_disconnected = None

    def test_solution_adds_server_disconnected_method(self):
        # Verify the method doesn't exist before calling solution
        self.assertFalse(
            hasattr(ConnectionLogger, "server_disconnected")
            and callable(getattr(ConnectionLogger, "server_disconnected"))
        )

        # Call solution function
        solution()

        # Verify the method exists after calling solution
        self.assertTrue(
            hasattr(ConnectionLogger, "server_disconnected")
            and callable(getattr(ConnectionLogger, "server_disconnected"))
        )

    def test_server_disconnected_prints_sockname(self):
        # Call solution to add the method
        solution()

        # Create a dummy server connection with a test sockname
        test_sockname = ("127.0.0.1", 8080)
        server_conn = DummyServerConn(test_sockname)

        # Create a logger instance
        logger = ConnectionLogger()

        # Capture stdout to verify the print output
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            logger.server_disconnected(server_conn)

        # Verify the output contains the sockname
        self.assertEqual(captured_output.getvalue().strip(), str(test_sockname))


if __name__ == "__main__":
    unittest.main()
