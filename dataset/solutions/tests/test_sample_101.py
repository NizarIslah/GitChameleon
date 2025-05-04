# Add the parent directory to import sys
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from django.forms import Form
from django.forms.models import BaseModelFormSet

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_101 import save_existing


class TestSample101(unittest.TestCase):
    def setUp(self):
        # Create mock objects for testing
        self.mock_form = MagicMock(spec=Form)
        self.mock_obj = "test_instance"

        # Create a mock formset with a save_existing method
        self.mock_formset = MagicMock(spec=BaseModelFormSet)
        self.mock_formset.save_existing.return_value = None

    def test_save_existing(self):
        """Test that save_existing correctly calls the formset's save_existing method."""
        # Call the function under test
        result = save_existing(
            formset=self.mock_formset, form=self.mock_form, obj=self.mock_obj
        )

        # Assert that the formset's save_existing method was called with the correct parameters
        self.mock_formset.save_existing.assert_called_once_with(
            form=self.mock_form, instance=self.mock_obj
        )

        # Assert that our function returns the result of the formset's save_existing method
        self.assertEqual(result, self.mock_formset.save_existing.return_value)


if __name__ == "__main__":
    unittest.main()
