import unittest
import spacy
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sample_92 import create_whitespace_variant

class TestCreateWhitespaceVariant(unittest.TestCase):
    def setUp(self):
        # blank English model
        self.nlp = spacy.blank("en")
        text = "This is a test sentence."
        # build a minimal Doc object
        self.doc = self.nlp(text)

    def create_example(self, doc):
        """Mock function to create an example-like structure."""
        return {
            "text": doc.text,
            "reference": doc,
            "token_annotation": {"ORTH": [t.text for t in doc]},
            "doc_annotation": {"entities": ["O"] * len(doc), "links": {}, "spans": {}},
        }

    def test_create_whitespace_variant_end(self):
        """Test adding whitespace at the end of the token stream."""
        whitespace = "  "  # two spaces
        example = self.create_example(self.doc)
        position = len(example["reference"])
        augmented = create_whitespace_variant(self.nlp, example, whitespace, position)
        aug_text = augmented.text
        orig_text = example["text"]

        # should simply append the whitespace
        self.assertTrue(aug_text.endswith(whitespace))
        self.assertEqual(len(aug_text), len(orig_text) + len(whitespace))

    def test_create_whitespace_variant_newline(self):
        """Test adding a newline at the end of the token stream."""
        whitespace = "\n"
        example = self.create_example(self.doc)
        position = len(example["reference"])
        augmented = create_whitespace_variant(self.nlp, example, whitespace, position)
        aug_text = augmented.text
        orig_text = example["text"]

        # newline should be appended
        self.assertTrue(aug_text.endswith(whitespace))
        self.assertEqual(len(aug_text), len(orig_text) + len(whitespace))

    def test_create_whitespace_variant_space(self):
        """Test adding a single space before the 'sentence' token."""
        whitespace = " "
        example = self.create_example(self.doc)
        tokens = [t.text for t in example["reference"]]
        position = tokens.index("sentence")
        augmented = create_whitespace_variant(self.nlp, example, whitespace, position)
        aug_text = augmented.text
        orig_text = example["text"]

        # length increased by exactly one space
        self.assertEqual(len(aug_text), len(orig_text) + 1)
        # check that space was inserted immediately before "sentence"
        self.assertIn("test" + whitespace + "sentence", aug_text)

    def test_create_whitespace_variant_tab(self):
        """Test adding a tab character before the 'sentence' token."""
        whitespace = "\t"
        example = self.create_example(self.doc)
        tokens = [t.text for t in example["reference"]]
        position = tokens.index("sentence")
        augmented = create_whitespace_variant(self.nlp, example, whitespace, position)
        aug_text = augmented.text
        orig_text = example["text"]

        self.assertEqual(len(aug_text), len(orig_text) + 1)
        self.assertIn("test" + whitespace + "sentence", aug_text)

if __name__ == "__main__":
    unittest.main()
