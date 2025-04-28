# Add the parent directory to import sys
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gradio as gr
import sample_42

# Filter deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Check gradio version
gr_version = gr.__version__
print(f"Using gradio version: {gr_version}")


class TestDropdownSelection(unittest.TestCase):
    """Test cases for the get_selected_options function and Gradio Interface in sample_42.py."""

    def test_get_selected_options_with_single_option(self):
        """Test that get_selected_options returns the correct string with a single option."""
        # Test with a single option
        result = sample_42.get_selected_options(["angola"])
        self.assertEqual(result, "Selected options: ['angola']")
        
        # Check that the result is a string
        self.assertIsInstance(result, str)
        
        # Test with a different single option
        result = sample_42.get_selected_options(["canada"])
        self.assertEqual(result, "Selected options: ['canada']")

    def test_get_selected_options_with_multiple_options(self):
        """Test that get_selected_options returns the correct string with multiple options."""
        # Test with multiple options
        result = sample_42.get_selected_options(["angola", "pakistan"])
        self.assertEqual(result, "Selected options: ['angola', 'pakistan']")
        
        # Test with all options
        result = sample_42.get_selected_options(["angola", "pakistan", "canada"])
        self.assertEqual(result, "Selected options: ['angola', 'pakistan', 'canada']")

    def test_get_selected_options_with_no_options(self):
        """Test that get_selected_options returns the correct string with no options."""
        # Test with empty list
        result = sample_42.get_selected_options([])
        self.assertEqual(result, "Selected options: []")
        
        # Test with None (though this shouldn't happen in practice)
        result = sample_42.get_selected_options(None)
        self.assertEqual(result, "Selected options: None")

    def test_selection_options_list(self):
        """Test that the selection_options list contains the expected values."""
        # Check that selection_options is a list
        self.assertIsInstance(sample_42.selection_options, list)
        
        # Check that selection_options contains the expected values
        self.assertEqual(len(sample_42.selection_options), 3)
        self.assertIn("angola", sample_42.selection_options)
        self.assertIn("pakistan", sample_42.selection_options)
        self.assertIn("canada", sample_42.selection_options)

    def test_interface_creation(self):
        """Test that the Gradio Interface is created correctly."""
        # Check that interface is a Gradio Interface object
        self.assertIsInstance(sample_42.iface, gr.Interface)
        
        # Check that the interface has the correct function
        self.assertEqual(sample_42.iface.fn, sample_42.get_selected_options)
        
        # For Gradio 3.17.0, we can check the components
        if hasattr(sample_42.iface, 'input_components'):
            # Check that there is one input component
            self.assertEqual(len(sample_42.iface.input_components), 1)
            # Check that the input is a Dropdown component
            self.assertIsInstance(sample_42.iface.input_components[0], gr.components.Dropdown)
            # Check that the dropdown has the correct options
            self.assertEqual(sample_42.iface.input_components[0].choices, sample_42.selection_options)
            # Check that multiselect is enabled
            self.assertTrue(sample_42.iface.input_components[0].multiselect)
            
        if hasattr(sample_42.iface, 'output_components'):
            # Check that there is one output component
            self.assertEqual(len(sample_42.iface.output_components), 1)
            # Check that the output is a Textbox component
            self.assertIsInstance(sample_42.iface.output_components[0], gr.components.Textbox)

    @patch('gradio.Interface.launch')
    def test_interface_launch(self, mock_launch):
        """Test that the interface can be launched."""
        # Set up the mock to return a simple object
        mock_launch.return_value = MagicMock()
        
        # Launch the interface
        result = sample_42.iface.launch()
        
        # Check that launch was called
        mock_launch.assert_called_once()
        
        # Check that a result was returned
        self.assertIsNotNone(result)

    @patch('gradio.Interface.launch')
    def test_interface_launch_with_share(self, mock_launch):
        """Test that the interface can be launched with sharing enabled."""
        # Set up the mock to return a simple object
        mock_launch.return_value = MagicMock()
        
        # Launch the interface with share=True
        result = sample_42.iface.launch(share=True)
        
        # Check that launch was called with share=True
        mock_launch.assert_called_once_with(share=True)
        
        # Check that a result was returned
        self.assertIsNotNone(result)

    @patch('gradio.Interface.launch')
    def test_interface_with_custom_server_name(self, mock_launch):
        """Test that the interface can be launched with a custom server name."""
        # Set up the mock to return a simple object
        mock_launch.return_value = MagicMock()
        
        # Launch the interface with a custom server name
        result = sample_42.iface.launch(server_name="0.0.0.0")
        
        # Check that launch was called with server_name="0.0.0.0"
        mock_launch.assert_called_once_with(server_name="0.0.0.0")
        
        # Check that a result was returned
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()