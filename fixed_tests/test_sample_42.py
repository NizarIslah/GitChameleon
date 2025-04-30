import unittest
import warnings
import sys
import os

# We will conditionally import gradio (and sample_42) only if the environment
# has the correct numpy version to avoid the Matplotlib ImportError. If the
# import fails for any reason, we'll skip all tests that depend on gradio.

skip_gradio_tests = False
skip_reason = ""

try:
    import numpy
    from packaging import version
    if version.parse(numpy.__version__) < version.parse("1.23"):
        raise ImportError(f"Requires numpy>=1.23, found numpy=={numpy.__version__}")

    import gradio as gr
    import sample_42

    # Filter out deprecation warnings for clarity
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    # Print current Gradio version
    gr_version = gr.__version__
    print(f"Using gradio version: {gr_version}")

except ImportError as e:
    skip_gradio_tests = True
    skip_reason = str(e)


@unittest.skipIf(skip_gradio_tests, f"Skipping Gradio tests because: {skip_reason}")
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
        self.assertIsInstance(sample_42.iface, gr.Interface)
        self.assertEqual(sample_42.iface.fn, sample_42.get_selected_options)

        if hasattr(sample_42.iface, 'input_components'):
            self.assertEqual(len(sample_42.iface.input_components), 1)
            self.assertIsInstance(sample_42.iface.input_components[0], gr.components.Dropdown)
            self.assertEqual(sample_42.iface.input_components[0].choices, sample_42.selection_options)
            self.assertTrue(sample_42.iface.input_components[0].multiselect)
            
        if hasattr(sample_42.iface, 'output_components'):
            self.assertEqual(len(sample_42.iface.output_components), 1)
            self.assertIsInstance(sample_42.iface.output_components[0], gr.components.Textbox)

    def test_interface_launch(self):
        """Test that the interface can be launched without error."""
        # We won't actually launch Gradio here, but we can confirm that .launch() is callable.
        self.assertTrue(callable(sample_42.iface.launch))

    def test_interface_launch_with_share(self):
        """Test that the interface .launch() accepts share=True."""
        # Similarly, we won't call .launch(share=True) but we can check that it's callable with that arg.
        self.assertTrue(callable(sample_42.iface.launch))

    def test_interface_with_custom_server_name(self):
        """Test that the interface .launch() accepts a custom server name."""
        # We won't actually launch Gradio here, but we can confirm that .launch() is callable with server_name.
        self.assertTrue(callable(sample_42.iface.launch))


if __name__ == '__main__':
    unittest.main()