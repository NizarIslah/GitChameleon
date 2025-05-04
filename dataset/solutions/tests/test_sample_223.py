import unittest
import sys
import io
from contextlib import redirect_stdout

# Import the module to test
sys.path.append("/repo/dataset/solutions")
from sample_223 import DummyLogEntry, MyAddon, solution


class TestSample223(unittest.TestCase):
    def setUp(self):
        # Reset MyAddon class before each test
        if hasattr(MyAddon, "add_log"):
            delattr(MyAddon, "add_log")

    def test_dummy_log_entry_init(self):
        """Test that DummyLogEntry initializes with the correct message."""
        test_msg = "Test log message"
        log_entry = DummyLogEntry(test_msg)
        self.assertEqual(log_entry.msg, test_msg)

    def test_solution_adds_method(self):
        """Test that solution() adds add_log method to MyAddon."""
        # Verify method doesn't exist before calling solution
        self.assertFalse(
            hasattr(MyAddon, "add_log") and callable(getattr(MyAddon, "add_log"))
        )

        # Call solution
        solution()

        # Verify method exists after calling solution
        self.assertTrue(
            hasattr(MyAddon, "add_log") and callable(getattr(MyAddon, "add_log"))
        )

    def test_add_log_output(self):
        """Test that add_log prints the message."""
        # Call solution to add the method
        solution()

        # Create a test instance and log entry
        addon = MyAddon()
        test_msg = "Test log message"
        log_entry = DummyLogEntry(test_msg)

        # Capture stdout
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            addon.add_log(log_entry)

        # Check output
        self.assertEqual(captured_output.getvalue().strip(), test_msg)


if __name__ == "__main__":
    unittest.main()
