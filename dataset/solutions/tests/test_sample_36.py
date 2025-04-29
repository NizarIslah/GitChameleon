import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gradio as gr
import sample_36

# Filter deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

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

if __name__ == '__main__':
    unittest.main()
