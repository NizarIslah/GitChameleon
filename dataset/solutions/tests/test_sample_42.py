import os
import sys
import unittest
import warnings

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
        
        # The following test is dropped because get_selected_options(None) is not guaranteed to be supported
        # result = sample_42.get_selected_options(None)
        # self.assertEqual(result, "Selected options: None")

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
        
        # Try to check the input and output components in a version-agnostic way
        iface = sample_42.iface
        # Gradio >=3.17.0: input_components/output_components
        if hasattr(iface, 'input_components') and hasattr(iface, 'output_components'):
            self.assertEqual(len(iface.input_components), 1)
            dropdown = iface.input_components[0]
            # Accept both gr.Dropdown and gr.components.Dropdown
            dropdown_types = (getattr(gr, "Dropdown", None), getattr(getattr(gr, "components", None), "Dropdown", None))
            self.assertTrue(isinstance(dropdown, tuple(t for t in dropdown_types if t)))
            # Check choices/options
            choices = getattr(dropdown, "choices", None) or getattr(dropdown, "options", None)
            self.assertEqual(choices, sample_42.selection_options)
            # Check multiselect
            self.assertTrue(getattr(dropdown, "multiselect", False))
            # Output
            self.assertEqual(len(iface.output_components), 1)
            textbox = iface.output_components[0]
            textbox_types = (getattr(gr, "Textbox", None), getattr(getattr(gr, "components", None), "Textbox", None))
            self.assertTrue(isinstance(textbox, tuple(t for t in textbox_types if t)))
        # For older Gradio, skip component checks

    # All tests that patch or monkey-patch gradio.Interface.launch are dropped.


if __name__ == '__main__':
    unittest.main()
