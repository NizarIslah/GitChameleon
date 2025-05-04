# Add the parent directory to import sys
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_103 import SampleForm, get_template_string, render_output


class TestSample103(unittest.TestCase):
    def setUp(self):
        # Setup runs before each test method
        self.template_string = get_template_string()
        self.expected_elements = [
            r"<form>",
            r"<div>",
            r'<label for=[\'"]id_name[\'"]>Name:</label>',
            r'<div class=[\'"]helptext[\'"] id=[\'"]id_name_helptext[\'"]>Enter your name</div>',
            r'<input type=[\'"]text[\'"] name=[\'"]name[\'"] required.*id=[\'"]id_name[\'"]',
            r"</div>",
            r"</form>",
        ]

    def test_form_field_definition(self):
        """Test that the SampleForm has the correct field definition"""
        form = SampleForm()
        self.assertTrue(hasattr(form, "fields"))
        self.assertIn("name", form.fields)
        name_field = form.fields["name"]
        self.assertEqual(name_field.label, "Name")
        self.assertEqual(name_field.help_text, "Enter your name")

    def test_get_template_string(self):
        """Test that get_template_string returns the expected template"""
        template = get_template_string()
        self.assertIsInstance(template, str)
        self.assertIn("{{ form.name.as_field_group }}", template)

    def test_render_output(self):
        """Test that render_output produces the expected HTML"""
        rendered_html = render_output(self.template_string)

        # Remove whitespace for easier comparison
        cleaned_html = re.sub(r"\s+", " ", rendered_html).strip()

        # Check that all expected elements are in the rendered HTML
        for pattern in self.expected_elements:
            with self.subTest(pattern=pattern):
                self.assertTrue(
                    re.search(pattern, cleaned_html, re.IGNORECASE),
                    f"Pattern '{pattern}' not found in rendered HTML: {cleaned_html}",
                )

    def test_form_rendering_with_as_field_group(self):
        """Test that the form field renders with as_field_group template tag"""
        rendered_html = render_output(self.template_string)

        # Check that the rendered HTML contains the label, help text, and input field
        self.assertIn('<label for="id_name">Name:</label>', rendered_html)
        self.assertIn(
            '<div class="helptext" id="id_name_helptext">Enter your name</div>',
            rendered_html,
        )
        self.assertIn('input type="text" name="name" required', rendered_html.lower())
        self.assertIn('id="id_name"', rendered_html.lower())

        # Check that the aria-describedby attribute is present and points to the help text
        self.assertIn('aria-describedby="id_name_helptext"', rendered_html.lower())


if __name__ == "__main__":
    unittest.main()
