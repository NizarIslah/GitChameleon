# Add the parent directory to import sys
import os
import sys
import unittest
from unittest.mock import Mock, patch

from django.forms import Form
from django.forms.models import BaseModelFormSet

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_102 import save_existing


class TestSaveExisting(unittest.TestCase):
    def setUp(self):
        # Create mock objects for testing
        self.mock_formset = Mock(spec=BaseModelFormSet)
        self.mock_form = Mock(spec=Form)
        self.instance = "test_instance"

        # Set up the mock return value for the save_existing method
        self.mock_formset.save_existing.return_value = None

    def test_save_existing_calls_formset_method(self):
        """Test that save_existing correctly calls the formset's save_existing method with the right parameters."""
        # Call the function under test
        result = save_existing(
            formset=self.mock_formset, form=self.mock_form, instance=self.instance
        )

        # Assert that the formset's save_existing method was called with the correct parameters
        self.mock_formset.save_existing.assert_called_once_with(
            form=self.mock_form, obj=self.instance
        )

        # Assert that the function returns the result of the formset's save_existing method
        self.assertEqual(result, self.mock_formset.save_existing.return_value)

    def test_save_existing_with_different_instance(self):
        """Test save_existing with a different instance value."""
        different_instance = "different_instance"

        # Call the function under test with a different instance
        save_existing(
            formset=self.mock_formset, form=self.mock_form, instance=different_instance
        )

        # Assert that the formset's save_existing method was called with the correct parameters
        self.mock_formset.save_existing.assert_called_once_with(
            form=self.mock_form, obj=different_instance
        )


if __name__ == "__main__":
    unittest.main()
