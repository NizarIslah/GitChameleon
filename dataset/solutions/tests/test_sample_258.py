import asyncio
import signal

# Import the function to test
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.append("/repo/dataset/solutions")
from sample_258 import custom_add_callback_from_signal


class TestCustomAddCallbackFromSignal(unittest.TestCase):
    """Test cases for custom_add_callback_from_signal function."""

    def setUp(self):
        """Set up test environment."""
        # Create a new event loop for each test
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Create a mock callback function
        self.callback_mock = MagicMock()

    def tearDown(self):
        """Clean up after each test."""
        # Close the event loop
        self.loop.close()
        asyncio.set_event_loop(None)

    @patch("asyncio.get_event_loop")
    def test_custom_add_callback_from_signal(self, mock_get_event_loop):
        """Test that the function correctly adds a signal handler to the event loop."""
        # Setup the mock
        mock_loop = MagicMock()
        mock_get_event_loop.return_value = mock_loop

        # Call the function with SIGINT (2)
        custom_add_callback_from_signal(self.callback_mock, signal.SIGINT)

        # Assert that add_signal_handler was called with the correct arguments
        mock_loop.add_signal_handler.assert_called_once_with(
            signal.SIGINT, self.callback_mock
        )

    def test_integration_with_real_loop(self):
        """Integration test with a real event loop."""
        # This test actually adds a signal handler to the event loop
        test_signal = signal.SIGUSR1

        # Define a callback that sets a flag
        callback_called = False

        def test_callback():
            nonlocal callback_called
            callback_called = True
            self.loop.stop()

        # Add the signal handler
        custom_add_callback_from_signal(test_callback, test_signal)

        # Verify the signal handler was added by checking if it's in the loop's signal handlers
        # This is implementation-specific and might not work on all platforms
        if hasattr(self.loop, "_signal_handlers"):
            self.assertIn(test_signal, self.loop._signal_handlers)

        # Note: We don't actually send the signal in the test as it could
        # interfere with the test runner, but we've verified the handler was registered


if __name__ == "__main__":
    unittest.main()
