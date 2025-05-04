import unittest
import spacy
from spacy.training import Example
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_92 import create_whitespace_variant


class TestCreateWhitespaceVariant(unittest.TestCase):
    def setUp(self):
        # blank English model
        self.nlp = spacy.blank("en")
        text = "This is a test sentence."
        # build a minimal Example with just tokenization
        doc = self.nlp(text)
        self.example = Example.from_dict(
            doc,
            {
                "token_annotation": {"ORTH": [t.text for t in doc]},
                "doc_annotation": {
                    "entities": ["O"] * len(doc),
                    "links": {},
                    "spans": {},
                },
            },
        )

    def test_create_whitespace_variant_end(self):
        """Test adding whitespace at the end of the token stream."""
        whitespace = "  "  # two spaces
        # insert at the end (after last token)
        position = len(self.example.reference)
        augmented = create_whitespace_variant(
            self.nlp, self.example, whitespace, position
        )
        aug_text = augmented.text
        orig_text = self.example.text

        # should simply append the whitespace
        self.assertTrue(aug_text.endswith(whitespace))


if __name__ == "__main__":
    unittest.main()
