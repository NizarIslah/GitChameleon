# Add the parent directory to import sys
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gradio as gr
import sample_39

# Filter deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Check gradio version
gr_version = gr.__version__
print(f"Using gradio version: {gr_version}")


class TestImageDisplay(unittest.TestCase):
    """Test cases for the display_image function and Gradio Interface in sample_39.py."""

    def test_display_image_returns_correct_url(self):
        """Test that display_image returns the correct image URL."""
        expected_url = "https://image_placeholder.com/42"
        result = sample_39.display_image()
        self.assertEqual(result, expected_url)
        # Check that the result is a string
        self.assertIsInstance(result, str)
        # Check that the result contains a valid URL format
        self.assertTrue(result.startswith("http"))

    def test_interface_creation(self):
        """Test that the Gradio Interface is created correctly."""
        # Check that interface is a Gradio Interface object
        self.assertIsInstance(sample_39.iface, gr.Interface)
        
        # Check that the interface has the correct function
        self.assertEqual(sample_39.iface.fn, sample_39.display_image)
        
        # Check that the interface has the expected configuration
        if hasattr(sample_39.iface, 'input_components'):
            # Check that there are no input components
            self.assertEqual(len(sample_39.iface.input_components), 0)
            
        if hasattr(sample_39.iface, 'output_components'):
            # Check that there is one output component
            self.assertEqual(len(sample_39.iface.output_components), 1)
            # Check that the output is an Image component
            self.assertIsInstance(sample_39.iface.output_components[0], gr.components.Image)


if __name__ == '__main__':
    unittest.main()
