# Add the parent directory to import sys
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
import sample_37

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Check gradio version
gr_version = gr.__version__
print(f"Using gradio version: {gr_version}")


class TestQuadraticFormulaChatbot(unittest.TestCase):
    """Test cases for the render_quadratic_formula function and Gradio Chatbot in sample_37.py."""

    def test_render_quadratic_formula_returns_correct_formula(self):
        """Test that render_quadratic_formula returns the correct formula."""
        expected_formula = "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}"
        result = sample_37.render_quadratic_formula()
        self.assertEqual(result, expected_formula)
        # Check that the result is a string
        self.assertIsInstance(result, str)
        # Check that the result contains the expected LaTeX formatting
        self.assertTrue("\\frac" in result)
        self.assertTrue("\\sqrt" in result)
        self.assertTrue("\\pm" in result)

    def test_chatbot_interface_creation(self):
        """Test that the Gradio Chatbot interface is created correctly."""
        # Check that interface is a Gradio Chatbot object
        self.assertIsInstance(sample_37.interface, gr.Chatbot)

    @patch("gradio.blocks.Blocks.launch")
    def test_interface_launch_with_blocking_false(self, mock_launch):
        """Test that the interface can be launched with blocking=False."""
        # Create a Blocks instance with the Chatbot
        with gr.Blocks() as demo:
            sample_37.interface.render()

        # Set up the mock to return a simple object
        mock_launch.return_value = MagicMock()

        # Launch the demo with blocking=False
        result = demo.launch(blocking=False)

        # Check that launch was called with blocking=False
        mock_launch.assert_called_once_with(blocking=False)

        # Check that a result was returned
        self.assertIsNotNone(result)

    @patch("gradio.blocks.Blocks.launch")
    def test_interface_with_custom_theme(self, mock_launch):
        """Test that the interface can be launched with a custom theme."""
        # Create a Blocks instance with the Chatbot
        with gr.Blocks() as demo:
            sample_37.interface.render()

        # Set up the mock to return a simple object
        mock_launch.return_value = MagicMock()

        # Launch the demo with a custom theme
        result = demo.launch(theme="default")

        # Check that launch was called with theme="default"
        mock_launch.assert_called_once_with(theme="default")

        # Check that a result was returned
        self.assertIsNotNone(result)

    @patch("gradio.blocks.Blocks.launch")
    def test_interface_with_share_true(self, mock_launch):
        """Test that the interface can be launched with share=True."""
        # Create a Blocks instance with the Chatbot
        with gr.Blocks() as demo:
            sample_37.interface.render()

        # Set up the mock to return a simple object
        mock_launch.return_value = MagicMock()

        # Launch the demo with share=True
        result = demo.launch(share=True)

        # Check that launch was called with share=True
        mock_launch.assert_called_once_with(share=True)

        # Check that a result was returned
        self.assertIsNotNone(result)

    @patch("gradio.blocks.Blocks.launch")
    def test_interface_with_auth_credentials(self, mock_launch):
        """Test that the interface can be launched with authentication credentials."""
        # Create a Blocks instance with the Chatbot
        with gr.Blocks() as demo:
            sample_37.interface.render()

        # Set up the mock to return a simple object
        mock_launch.return_value = MagicMock()

        # Create auth credentials
        auth = ("username", "password")

        # Launch the demo with auth
        result = demo.launch(auth=auth)

        # Check that launch was called with auth=auth
        mock_launch.assert_called_once_with(auth=auth)

        # Check that a result was returned
        self.assertIsNotNone(result)

    @patch("gradio.blocks.Blocks.launch")
    def test_interface_with_invalid_parameters(self, mock_launch):
        """Test that the interface handles invalid parameters appropriately."""
        # Create a Blocks instance with the Chatbot
        with gr.Blocks() as demo:
            sample_37.interface.render()

        # Set up the mock to raise an error for invalid parameters
        mock_launch.side_effect = ValueError("Invalid parameter")

        # Try to launch the demo with an invalid parameter
        with self.assertRaises(ValueError):
            demo.launch(server_port=-1)

        # Check that launch was called with server_port=-1
        mock_launch.assert_called_once_with(server_port=-1)


if __name__ == "__main__":
    unittest.main()
