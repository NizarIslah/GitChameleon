import unittest
import sys
import os
import nltk
from nltk.corpus import wordnet

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_95 import get_synset_examples


class TestGetSynsetExamples(unittest.TestCase):
    def setUp(self):
        # Ensure the required NLTK data is downloaded
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    def test_get_synset_examples_returns_list(self):
        """Test that get_synset_examples returns a list."""
        examples = get_synset_examples("dog.n.01")
        self.assertIsInstance(examples, list)

    def test_get_synset_examples_content(self):
        """Test that get_synset_examples returns expected examples for a known synset."""
        # 'dog.n.01' is a common synset with known examples
        examples = get_synset_examples("dog.n.01")
        self.assertTrue(
            len(examples) > 0, "Expected at least one example for 'dog.n.01'"
        )

        # Verify that the examples are strings
        for example in examples:
            self.assertIsInstance(example, str)

    def test_invalid_synset(self):
        """Test that an invalid synset raises an appropriate exception."""
        with self.assertRaises(ValueError):
            get_synset_examples("nonexistent_synset")

    def test_different_synsets(self):
        """Test that different synsets return different examples."""
        dog_examples = get_synset_examples("dog.n.01")
        cat_examples = get_synset_examples("cat.n.01")

        # Different synsets should have different examples
        self.assertNotEqual(dog_examples, cat_examples)


if __name__ == "__main__":
    unittest.main()
