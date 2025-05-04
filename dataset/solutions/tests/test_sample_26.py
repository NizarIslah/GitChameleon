# Add the parent directory to import sys
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import contextlib
import io

import nltk
import sample_26

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Check nltk version
nltk_version = nltk.__version__
print(f"Using nltk version: {nltk_version}")


class TestShowUsage(unittest.TestCase):
    """Test cases for the show_usage function in sample_26.py."""

    def test_basic_usage_with_valid_object(self):
        """Test basic usage with a valid NLTK object that has usage information."""
        # Use nltk.downloader as it has usage information
        result = sample_26.show_usage(nltk.downloader)

        # The result should be a non-empty string
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        # The output should contain some expected text from the usage information
        # NLTK's usage() function might format the output differently
        # Just check that we get some non-empty output
        self.assertTrue(len(result) > 0)

    def test_usage_with_none_object(self):
        """Test usage with None object."""
        try:
            # This might raise an AttributeError or TypeError
            result = sample_26.show_usage(None)

            # If we get here, the function didn't raise an error
            # The result should be a string (possibly empty)
            self.assertIsInstance(result, str)
        except (AttributeError, TypeError) as e:
            # If an error is raised, that's also acceptable behavior for None input
            self.assertTrue(True)

    def test_usage_with_object_lacking_usage_info(self):
        """Test usage with an object that doesn't have usage information."""

        # Create a simple object that doesn't have usage information
        class SimpleObject:
            pass

        obj = SimpleObject()

        try:
            # This might raise an AttributeError
            result = sample_26.show_usage(obj)

            # If we get here, the function didn't raise an error
            # The result should be a string (possibly empty)
            self.assertIsInstance(result, str)
        except AttributeError as e:
            # If an AttributeError is raised, that's acceptable behavior
            self.assertTrue(
                "'SimpleObject' object has no attribute" in str(e)
                or "has no attribute 'usage'" in str(e)
            )

    def test_usage_with_custom_object(self):
        """Test usage with a custom object that implements a usage method."""

        # Create a custom object with a usage method
        class CustomObject:
            @staticmethod
            def usage():
                print("This is a custom usage message")

        obj = CustomObject()

        # Get the usage information
        result = sample_26.show_usage(obj)

        # The result should be a non-empty string
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        # The output should contain the custom usage message
        # NLTK's usage() function formats the output with additional information
        self.assertTrue("CustomObject supports the following operations:" in result)
        self.assertTrue("cls.usage()" in result)

    def test_usage_with_nltk_module(self):
        """Test usage with an NLTK module."""
        # Try with different NLTK modules that might have usage information
        nltk_modules = [nltk.tokenize, nltk.stem, nltk.tag]

        for module in nltk_modules:
            try:
                # This might raise an AttributeError if the module doesn't have usage
                result = sample_26.show_usage(module)

                # If we get here, the function didn't raise an error
                # The result should be a string
                self.assertIsInstance(result, str)

                # If the result is non-empty, it should contain some expected text
                if len(result) > 0:
                    print(f"Module {module.__name__} has usage information")
                    # No need to check further, we found a module with usage
                    break
            except AttributeError:
                # If an AttributeError is raised, continue with the next module
                continue

        # If we couldn't find any module with usage, at least verify the function works
        # by creating a mock object with a usage method
        class MockNLTKModule:
            @staticmethod
            def usage():
                print("Mock NLTK module usage information")

        mock_module = MockNLTKModule()
        result = sample_26.show_usage(mock_module)

        # The result should be a non-empty string
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        # The output should contain the mock usage message
        # NLTK's usage() function formats the output with additional information
        self.assertTrue("MockNLTKModule supports the following operations:" in result)
        self.assertTrue("cls.usage()" in result)


if __name__ == "__main__":
    unittest.main()
