# Add the parent directory to import sys
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gradio as gr
import sample_40

# Filter deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Check gradio version
gr_version = gr.__version__
print(f"Using gradio version: {gr_version}")


class TestImageProcessing(unittest.TestCase):
    """Test cases for the process_image function in sample_40.py."""

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
