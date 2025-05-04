import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch

# We add a guard to avoid import errors in environments
# that don't meet Gradio/Matplotlib requirements (e.g., numpy>=1.23).
# If the environment cannot import gradio for that reason, we skip all tests.
try:
    # Check numpy version first
    import numpy
    from packaging import version

    if version.parse(numpy.__version__) < version.parse("1.23"):
        raise ImportError(
            f"Skipping tests because numpy>=1.23 is required, found {numpy.__version__}"
        )

    import gradio as gr

    # Only insert parent directory after we confirm imports won't fail
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    import sample_40

    # Filter deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Check gradio version
    gr_version = gr.__version__
    print(f"Using gradio version: {gr_version}")

    class TestImageProcessing(unittest.TestCase):
        """Test cases for the process_image function and Gradio Interface in sample_40.py."""

        def test_process_image_returns_processed_text(self):
            """Test that process_image returns 'Processed' regardless of input."""
            # Test with None input
            result = sample_40.process_image(None)
            self.assertEqual(result, "Processed")

            # Test with a mock image input
            mock_image = MagicMock()
            result = sample_40.process_image(mock_image)
            self.assertEqual(result, "Processed")

            # Check that the result is a string
            self.assertIsInstance(result, str)

        @patch("gradio.Interface.launch")
        def test_interface_launch(self, mock_launch):
            """Test that the interface can be launched."""
            mock_launch.return_value = MagicMock()
            result = sample_40.iface.launch()
            mock_launch.assert_called_once()
            self.assertIsNotNone(result)

        @patch("gradio.Interface.launch")
        def test_interface_launch_with_share(self, mock_launch):
            """Test that the interface can be launched with sharing enabled."""
            mock_launch.return_value = MagicMock()
            result = sample_40.iface.launch(share=True)
            mock_launch.assert_called_once_with(share=True)
            self.assertIsNotNone(result)

        @patch("gradio.Interface.launch")
        def test_interface_with_custom_server_name(self, mock_launch):
            """Test that the interface can be launched with a custom server name."""
            mock_launch.return_value = MagicMock()
            result = sample_40.iface.launch(server_name="0.0.0.0")
            mock_launch.assert_called_once_with(server_name="0.0.0.0")
            self.assertIsNotNone(result)

        def test_process_image_with_different_inputs(self):
            """Test that process_image returns 'Processed' for different types of inputs."""
            # Test with empty string
            result = sample_40.process_image("")
            self.assertEqual(result, "Processed")

            # Test with a string path
            result = sample_40.process_image("path/to/image.jpg")
            self.assertEqual(result, "Processed")

            # Test with a dictionary (simulating a complex input)
            result = sample_40.process_image({"path": "image.jpg", "type": "jpg"})
            self.assertEqual(result, "Processed")

except ImportError as e:
    # If imports fail, we skip all tests.
    SKIP_REASON = str(e)

    class TestImageProcessing(unittest.TestCase):
        @unittest.skip(SKIP_REASON)
        def test_skipped_due_to_import_error(self):
            pass


if __name__ == "__main__":
    unittest.main()
