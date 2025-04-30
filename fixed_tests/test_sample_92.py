import unittest
import spacy
from spacy.tokens import Example
import sys
import os

# Ensure we can import create_whitespace_variant from sample_92.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sample_92 import create_whitespace_variant


class TestCreateWhitespaceVariant(unittest.TestCase):
    def setUp(self):
        # Use a blank English model
        self.nlp = spacy.blank("en")

        # Create a small Doc and an Example for testing
        text = "This is a test sentence."
        doc = self.nlp(text)
        # Example.from_dict() produces a valid spacy.training.Example object
        # which supports example.to_dict() internally
        self.example = Example.from_dict(
            doc,
            {
                "token_annotation": {"ORTH": [t.text for t in doc]},
                "doc_annotation": {
                    "entities": ["O"] * len(doc),
                    "links": {},
                    "spans": {}
                },
            },
        )

    def test_create_whitespace_variant_end(self):
        """Test adding whitespace at the end of the token stream."""
        whitespace = "  "  # two spaces
        # position is after the last token
        position = len(self.example.reference)
        augmented = create_whitespace_variant(self.nlp, self.example, whitespace, position)
        aug_text = augmented.text
        orig_text = self.example.text

        # should simply append the whitespace
        self.assertTrue(aug_text.endswith(whitespace))
        self.assertEqual(len(aug_text), len(orig_text) + len(whitespace))

    def test_create_whitespace_variant_newline(self):
        """Test adding a newline at the end of the token stream."""
        whitespace = "\n"
        # position is after the last token
        position = len(self.example.reference)
        augmented = create_whitespace_variant(self.nlp, self.example, whitespace, position)
        aug_text = augmented.text
        orig_text = self.example.text

        # newline should be appended
        self.assertTrue(aug_text.endswith(whitespace))
        self.assertEqual(len(aug_text), len(orig_text) + len(whitespace))

    def test_create_whitespace_variant_space(self):
        """Test adding a single space before the 'sentence' token."""
        whitespace = " "
        tokens = [t.text for t in self.example.reference]
        position = tokens.index("sentence")
        augmented = create_whitespace_variant(self.nlp, self.example, whitespace, position)
        aug_text = augmented.text
        orig_text = self.example.text

        # length increased by exactly one space
        self.assertEqual(len(aug_text), len(orig_text) + 1)
        # check that space was inserted immediately before "sentence"
        self.assertIn("test" + whitespace + "sentence", aug_text)

    def test_create_whitespace_variant_tab(self):
        """Test adding a tab character before the 'sentence' token."""
        whitespace = "\t"
        tokens = [t.text for t in self.example.reference]
        position = tokens.index("sentence")
        augmented = create_whitespace_variant(self.nlp, self.example, whitespace, position)
        aug_text = augmented.text
        orig_text = self.example.text

        # length increased by exactly one tab
        self.assertEqual(len(aug_text), len(orig_text) + 1)
        # check that tab was inserted immediately before "sentence"
        self.assertIn("test" + whitespace + "sentence", aug_text)


if __name__ == "__main__":
    unittest.main()