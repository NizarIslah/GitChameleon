import unittest
import io
import sys
from contextlib import redirect_stdout
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_219 import DummyServerConn, ConnectionLogger, solution


class TestSample219(unittest.TestCase):
    def test_solution(self):
        # Call the solution function to add server_connect method to ConnectionLogger
        solution()

        # Verify that server_connect method was added to ConnectionLogger
        self.assertTrue(
            hasattr(ConnectionLogger, "server_connect"),
            "server_connect method was not added to ConnectionLogger",
        )

        # Create a DummyServerConn instance with a test sockname
        test_sockname = ("127.0.0.1", 8080)
        server_conn = DummyServerConn(test_sockname)

        # Capture stdout to verify the print output
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            # Create a ConnectionLogger instance and call server_connect
            logger = ConnectionLogger()
            logger.server_connect(server_conn)

        # Verify that the sockname was printed correctly
        expected_output = str(test_sockname) + "\n"
        self.assertEqual(captured_output.getvalue(), expected_output)


if __name__ == "__main__":
    unittest.main()
