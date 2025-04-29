# Add the parent directory to import sys
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gradio as gr
import sample_37

# Filter deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

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


if __name__ == '__main__':
    unittest.main()
