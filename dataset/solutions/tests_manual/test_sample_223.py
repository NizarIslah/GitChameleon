import unittest
import sys
import io
from contextlib import redirect_stdout

# Import the module to test
sys.path.append("/repo/dataset/solutions")
from sample_223 import DummyLogEntry, MyAddon, solution


import unittest
import io


class TestMyAddonLogging(unittest.TestCase):
    def test_logging_event(self):
        addon = MyAddon()
        solution()
        dummy_entry = DummyLogEntry("Test log message")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            addon.add_log(dummy_entry)
        print(output.getvalue())

        self.assertIn("Test log message", output.getvalue())


unittest.main()
