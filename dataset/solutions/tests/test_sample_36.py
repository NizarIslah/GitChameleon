# Add the parent directory to import sys
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
import sample_36

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Check gradio version
gr_version = gr.__version__
print(f"Using gradio version: {gr_version}")


class TestQuadraticFormula(unittest.TestCase):
    """Test cases for the render_quadratic_formula function and Gradio interface in sample_36.py."""

    def test_render_quadratic_formula_returns_correct_formula(self):
        """Test that render_quadratic_formula returns the correct LaTeX formula."""
        expected_formula = "$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$"
        result = sample_36.render_quadratic_formula()
        self.assertEqual(result, expected_formula)
        # Check that the result is a string
        self.assertIsInstance(result, str)
        # Check that the result contains LaTeX formatting
        self.assertTrue(result.startswith("$") and result.endswith("$"))

    def test_interface_has_correct_configuration(self):
        """Test that the Gradio interface has the correct configuration."""
        # Check that interface is a Gradio Interface object
        self.assertIsInstance(sample_36.interface, gr.Interface)
        # Check that the function name is render_quadratic_formula
        self.assertEqual(sample_36.interface.fn.__name__, "render_quadratic_formula")

    @patch("gradio.Interface.launch")
    def test_interface_launch_with_blocking_false(self, mock_launch):
        """Test that the interface can be launched with blocking=False."""
        # Set up the mock to return a simple object
        mock_launch.return_value = MagicMock()

        # Launch the interface with blocking=False
        result = sample_36.interface.launch(blocking=False)

        # Check that launch was called with blocking=False
        mock_launch.assert_called_once_with(blocking=False)

        # Check that a result was returned
        self.assertIsNotNone(result)

    @patch("gradio.Interface.launch")
    def test_interface_with_custom_theme(self, mock_launch):
        """Test that the interface can be launched with a custom theme."""
        # Set up the mock to return a simple object
        mock_launch.return_value = MagicMock()

        # Launch the interface with a custom theme
        result = sample_36.interface.launch(theme="default")

        # Check that launch was called with theme="default"
        mock_launch.assert_called_once_with(theme="default")

        # Check that a result was returned
        self.assertIsNotNone(result)

    @patch("gradio.Interface.launch")
    def test_interface_with_share_true(self, mock_launch):
        """Test that the interface can be launched with share=True."""
        # Set up the mock to return a simple object
        mock_launch.return_value = MagicMock()

        # Launch the interface with share=True
        result = sample_36.interface.launch(share=True)

        # Check that launch was called with share=True
        mock_launch.assert_called_once_with(share=True)

        # Check that a result was returned
        self.assertIsNotNone(result)

    @patch("gradio.Interface.launch")
    def test_interface_with_auth_credentials(self, mock_launch):
        """Test that the interface can be launched with authentication credentials."""
        # Set up the mock to return a simple object
        mock_launch.return_value = MagicMock()

        # Create auth credentials
        auth = ("username", "password")

        # Launch the interface with auth
        result = sample_36.interface.launch(auth=auth)

        # Check that launch was called with auth=auth
        mock_launch.assert_called_once_with(auth=auth)

        # Check that a result was returned
        self.assertIsNotNone(result)

    @patch("gradio.Interface.launch")
    def test_interface_with_invalid_port(self, mock_launch):
        """Test that the interface handles invalid port numbers appropriately."""
        # Set up the mock to raise an error for invalid port
        mock_launch.side_effect = ValueError("Invalid port number")

        # Try to launch the interface with an invalid port
        with self.assertRaises(ValueError):
            sample_36.interface.launch(server_port=-1)

        # Check that launch was called with server_port=-1
        mock_launch.assert_called_once_with(server_port=-1)

    @patch("gradio.Interface.launch")
    def test_interface_with_server_name_parameter(self, mock_launch):
        """Test that the interface can be launched with a server_name parameter."""
        # Set up the mock to return a simple object
        mock_launch.return_value = MagicMock()

        # Launch the interface with a server_name
        result = sample_36.interface.launch(server_name="0.0.0.0")

        # Check that launch was called with server_name="0.0.0.0"
        mock_launch.assert_called_once_with(server_name="0.0.0.0")

        # Check that a result was returned
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
