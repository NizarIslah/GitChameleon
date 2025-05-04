# Add the parent directory to import sys
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_104 import SampleForm, get_template_string, render_output


class TestSample104(unittest.TestCase):
    def setUp(self):
        # Get the template string from the function
        self.template_string = get_template_string()

    def test_form_instance(self):
        """Test that SampleForm is properly initialized with expected fields"""
        form = SampleForm()
        self.assertTrue(hasattr(form, "fields"))
        self.assertIn("name", form.fields)
        self.assertEqual(form.fields["name"].label, "Name")
        self.assertEqual(form.fields["name"].help_text, "Enter your name")

    def test_template_string(self):
        """Test that the template string contains expected Django template tags"""
        self.assertIn("{{ form.name.label_tag }}", self.template_string)
        self.assertIn("{{ form.name.help_text|safe }}", self.template_string)
        self.assertIn("{{ form.name }}", self.template_string)
        self.assertIn("{% if form.name.help_text %}", self.template_string)
        self.assertIn("{% endif %}", self.template_string)

    def test_render_output(self):
        """Test that render_output produces the expected HTML"""
        rendered_html = render_output(self.template_string)

        # Check for label
        self.assertIn('<label for="id_name">Name:</label>', rendered_html)

        # Check for help text div
        self.assertIn('<div class="helptext" id="id_name_helptext">', rendered_html)
        self.assertIn("Enter your name", rendered_html)

        # Check for input field
        self.assertIn('<input type="text" name="name"', rendered_html)
        self.assertIn('id="id_name"', rendered_html)
        self.assertIn("required", rendered_html)

        # Check overall structure
        self.assertTrue(rendered_html.strip().startswith("<form>"))
        self.assertTrue(rendered_html.strip().endswith("</form>"))

    def test_rendered_output_matches_target(self):
        """Test that the rendered output matches the target HTML structure"""
        rendered_html = render_output(self.template_string)

        # Normalize whitespace for comparison
        def normalize_whitespace(text):
            # Replace multiple whitespace with a single space
            return re.sub(r"\s+", " ", text.strip())

        # Check that all expected elements are present in the correct order
        normalized_html = normalize_whitespace(rendered_html)

        # Check for the basic structure
        self.assertIn("<form>", normalized_html)
        self.assertIn("<div>", normalized_html)
        self.assertIn("</div>", normalized_html)
        self.assertIn("</form>", normalized_html)

        # Check for the correct order of elements
        label_pos = normalized_html.find('<label for="id_name">Name:</label>')
        helptext_pos = normalized_html.find(
            '<div class="helptext" id="id_name_helptext">'
        )
        input_pos = normalized_html.find('<input type="text" name="name"')

        self.assertGreater(label_pos, 0, "Label tag not found")
        self.assertGreater(
            helptext_pos, label_pos, "Help text div not found or in wrong order"
        )
        self.assertGreater(
            input_pos, helptext_pos, "Input field not found or in wrong order"
        )


if __name__ == "__main__":
    unittest.main()
