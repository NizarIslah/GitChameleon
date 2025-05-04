# File: test_gradio_app.py

import unittest
import gradio as gr
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
import sample_42 as gradio_app_module


class TestGradioInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This ensures that the module is loaded once for all tests in this class
        if gradio_app_module is None:
            raise unittest.SkipTest("gradio_app_module could not be imported.")
        cls.iface = gradio_app_module.iface
        cls.selection_options = gradio_app_module.selection_options
        cls.get_selected_options_fn = gradio_app_module.get_selected_options

    def test_iface_existence_and_type(self):
        """
        Test that the 'iface' variable exists and is an instance of gr.Interface.
        """
        self.assertIsNotNone(self.iface, "The 'iface' variable should not be None.")
        self.assertIsInstance(
            self.iface,
            gr.Interface,
            f"Expected iface to be a gr.Interface, but got {type(self.iface)}.",
        )

    def test_iface_output_components(self):
        """
        Test the properties of the output components of the interface.
        """
        self.assertIsInstance(
            self.iface.output_components, list, "Output components should be a list."
        )
        self.assertEqual(
            len(self.iface.output_components),
            1,
            "There should be one output component.",
        )

        output_textbox = self.iface.output_components[0]
        # The shorthand 'text' for outputs resolves to a gr.Textbox component
        self.assertIsInstance(
            output_textbox, gr.Textbox, "The output component should be a gr.Textbox."
        )
        # For 'text' output, the label might be None or a default if not explicitly set in a more complex setup.
        # If the Textbox component was created explicitly like gr.Textbox(label="My Output"), you'd test that.
        # Here, it defaults. For gr.Textbox, label defaults to None if not specified.
        self.assertEqual(output_textbox.label, "output")


if __name__ == "__main__":
    unittest.main()
