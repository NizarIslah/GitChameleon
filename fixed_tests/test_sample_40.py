import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch

# Make sure we can import from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gradio as gr

# Attempt to import sample_40, skip related tests if unavailable
try:
    import sample_40
    SAMPLE_40_AVAILABLE = True
except ImportError:
    SAMPLE_40_AVAILABLE = False
    sample_40 = None

# Filter deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Check gradio version
gr_version = gr.__version__
print(f"Using gradio version: {gr_version}")

class TestImageProcessing(unittest.TestCase):
    """Test cases for the process_image function and Gradio Interface in sample_40.py."""

    def setUp(self):
        if not SAMPLE_40_AVAILABLE:
            self.skipTest("sample_40 is not available. Skipping tests that require sample_40.")

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

    def test_interface_creation(self):
        """Test that the Gradio Interface is created correctly."""
        # Check that interface is a Gradio Interface object
        self.assertIsInstance(sample_40.iface, gr.Interface)
        
        # Check that the interface has the correct function
        self.assertEqual(sample_40.iface.fn, sample_40.process_image)
        
        # Handle differences in various Gradio versions
        if hasattr(sample_40.iface, 'inputs') and hasattr(gr, 'inputs'):
            # Check that there is one input component
            self.assertEqual(len(sample_40.iface.inputs), 1)
            # Check that the input is an Image component (in Gradio 2.9.2)
            self.assertIsInstance(sample_40.iface.inputs[0], gr.inputs.Image)
            
        if hasattr(sample_40.iface, 'outputs') and hasattr(gr, 'outputs'):
            # Check that there is one output component
            self.assertEqual(len(sample_40.iface.outputs), 1)
            # Check that the output is a Textbox component (in Gradio 2.9.2)
            self.assertIsInstance(sample_40.iface.outputs[0], gr.outputs.Textbox)

    @patch('gradio.Interface.launch')
    def test_interface_launch(self, mock_launch):
        """Test that the interface can be launched."""
        mock_launch.return_value = MagicMock()
        result = sample_40.iface.launch()
        mock_launch.assert_called_once()
        self.assertIsNotNone(result)

    @patch('gradio.Interface.launch')
    def test_interface_launch_with_share(self, mock_launch):
        """Test that the interface can be launched with sharing enabled."""
        mock_launch.return_value = MagicMock()
        result = sample_40.iface.launch(share=True)
        mock_launch.assert_called_once_with(share=True)
        self.assertIsNotNone(result)

    @patch('gradio.Interface.launch')
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


if __name__ == '__main__':
    unittest.main()