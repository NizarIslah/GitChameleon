# Add the parent directory to import sys
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_108 import MyModel, color


class TestSample108(unittest.TestCase):
    def test_model_structure(self):
        """Test that the model has the expected structure."""
        # Check that the model has a color field
        self.assertTrue(hasattr(MyModel, "color"))

        # Check that the color field is a CharField
        color_field = MyModel._meta.get_field("color")
        self.assertEqual(color_field.__class__.__name__, "CharField")

        # Check that the color field has the correct max_length
        self.assertEqual(color_field.max_length, 5)

        # Check that the color field has choices
        self.assertTrue(hasattr(color_field, "choices"))
        self.assertTrue(color_field.choices)

    def test_color_choices(self):
        """Test that the color choices are correctly defined."""
        # Check that the color choices are correctly defined
        self.assertEqual(color.RED, "RED")
        self.assertEqual(color.GREEN, "GREEN")
        self.assertEqual(color.BLUE, "BLUE")

        # Check that the choices are available in the model
        choices_dict = dict(MyModel._meta.get_field("color").choices)
        self.assertIn("RED", choices_dict)
        self.assertIn("GREEN", choices_dict)
        self.assertIn("BLUE", choices_dict)

    def test_model_creation(self):
        """Test that the model can be created with valid color values."""
        # Create a model instance with each valid color
        model_red = MyModel(color=color.RED)
        model_green = MyModel(color=color.GREEN)
        model_blue = MyModel(color=color.BLUE)

        # Check that the color values are correctly stored
        self.assertEqual(model_red.color, "RED")
        self.assertEqual(model_green.color, "GREEN")
        self.assertEqual(model_blue.color, "BLUE")

    def test_model_validation(self):
        """Test that the model validates color values."""
        from django.core.exceptions import ValidationError

        # Create a model with an invalid color
        model_invalid = MyModel(color="YELLOW")

        # Check that validation raises an error
        with self.assertRaises(ValidationError):
            model_invalid.full_clean()


if __name__ == "__main__":
    unittest.main()
